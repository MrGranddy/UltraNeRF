import os
import time
import imageio
import numpy as np

from run_nerf_helpers import get_embedder, NeRF
from load_us import load_us_data
from get_rays import get_rays_us_linear  # Assuming this function is defined in get_rays.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser

# Install pytorch-msssim if not already installed
try:
    from pytorch_msssim import ms_ssim
except ImportError:
    !pip install pytorch-msssim
    from pytorch_msssim import ms_ssim


def gaussian_kernel(size: int, mean: float, std: float) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel using two normal distributions.
    
    Args:
        size (int): Half-size of the kernel (total size will be 2*size + 1)
        mean (float): Mean of the Gaussian distributions
        std (float): Standard deviation of the Gaussian distributions
    
    Returns:
        torch.Tensor: Normalized 2D Gaussian kernel
    """
    delta_t = 1.0

    # Create range and scale by delta_t
    x = torch.linspace(-size, size, 2 * size + 1) * delta_t

    # Create normal distributions
    d1 = torch.distributions.Normal(mean, std * 2.0)
    d2 = torch.distributions.Normal(mean, std)

    # Compute probabilities
    vals_x = torch.exp(d1.log_prob(x))
    vals_y = torch.exp(d2.log_prob(x))

    # Outer product
    gauss_kernel = torch.ger(vals_x, vals_y)

    # Normalize
    gauss_kernel /= gauss_kernel.sum()
    return gauss_kernel


# Precompute the Gaussian kernel
g_size = 3
g_mean = 0.0
g_variance = 1.0
g_kernel = gaussian_kernel(g_size, g_mean, g_variance)
g_kernel = g_kernel.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, K_H, K_W]


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, fn, embed_fn, netchunk=512 * 32):
    """Prepares inputs and applies network 'fn'."""
    inputs_flat = inputs.view(-1, inputs.shape[-1])
    embedded = embed_fn(inputs_flat)
    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = outputs_flat.view(*inputs.shape[:-1], outputs_flat.shape[-1])
    return outputs


def render_method_convolutional_ultrasound(raw, z_vals):
    """Render method for ultrasound converted to PyTorch."""
    # Compute distance between points
    dists = torch.abs(z_vals[..., :-1] - z_vals[..., 1:])
    dists = torch.cat([dists, dists[..., -1:]], dim=-1)

    # ATTENUATION
    attenuation_coeff = torch.abs(raw[..., 0])
    attenuation = -attenuation_coeff * dists
    attenuation_transmission = torch.exp(torch.cumsum(attenuation, dim=-1))

    # REFLECTION
    prob_border = torch.sigmoid(raw[..., 2])
    border_indicator = (torch.rand_like(prob_border) < prob_border).float()
    reflection_coeff = torch.sigmoid(raw[..., 1])
    reflection_transmission = torch.log((1.0 - reflection_coeff) * border_indicator + 1e-8)
    reflection_transmission = torch.exp(torch.cumsum(reflection_transmission, dim=-1))

    # Border convolution
    border_indicator = border_indicator.unsqueeze(1).unsqueeze(1)  # Add channel dimensions
    border_convolution = F.conv2d(
        border_indicator, g_kernel.to(border_indicator.device), padding='same'
    ).squeeze(1).squeeze(1)  # Remove channel dimensions

    # BACKSCATTERING
    density_coeff = torch.sigmoid(raw[..., 3])
    scatterers_density = (torch.rand_like(density_coeff) < density_coeff).float()
    amplitude = torch.sigmoid(raw[..., 4])
    scatterers_map = scatterers_density * amplitude

    # Scattering convolution
    scatterers_map = scatterers_map.unsqueeze(1).unsqueeze(1)
    psf_scatter = F.conv2d(
        scatterers_map, g_kernel.to(scatterers_map.device), padding='same'
    ).squeeze(1).squeeze(1)

    # Compute transmission
    transmission = attenuation_transmission * reflection_transmission

    # Final echo components
    b = transmission * psf_scatter
    r = transmission * reflection_coeff * border_convolution
    intensity_map = b + r

    return {
        'intensity_map': intensity_map,
        'attenuation_coeff': attenuation_coeff,
        'reflection_coeff': reflection_coeff,
        'attenuation_transmission': attenuation_transmission,
        'reflection_transmission': reflection_transmission,
        'scatterers_density': scatterers_density,
        'scatterers_density_coeff': density_coeff,
        'scatter_amplitude': amplitude,
        'b': b,
        'r': r,
        'transmission': transmission,
    }


def render_rays_us(
    ray_batch,
    network_fn,
    network_query_fn,
    N_samples,
    retraw=False,
    lindisp=False,
    args=None,
):
    """Volumetric rendering for ultrasound rays."""
    def raw2outputs(raw, z_vals, rays_d):
        return render_method_convolutional_ultrasound(raw, z_vals)

    # Validate input
    if not isinstance(ray_batch, torch.Tensor):
        raise TypeError("ray_batch must be a torch.Tensor")

    # Batch size
    N_rays = ray_batch.shape[0]

    # Extract ray origin, direction
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]

    # Extract lower, upper bound for ray distance
    bounds = ray_batch[..., 6:8].view(-1, 1, 2)
    near, far = bounds[..., 0], bounds[..., 1]

    # Create sample points along rays
    t_vals = torch.linspace(0.0, 1.0, steps=N_samples, device=ray_batch.device)

    if not lindisp:
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)

    z_vals = z_vals.expand(N_rays, N_samples)

    # Compute points in space to evaluate model
    pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)

    # Evaluate model at each point
    raw = network_query_fn(pts, network_fn)  # [N_rays, N_samples, 5]

    # Transform raw output
    ret = raw2outputs(raw, z_vals, rays_d)

    # Optionally include raw output
    if retraw:
        ret['raw'] = raw

    # Numerical stability check
    for k, v in ret.items():
        if torch.isnan(v).any():
            raise ValueError(f"NaN detected in output {k}")
        if torch.isinf(v).any():
            raise ValueError(f"Inf detected in output {k}")

    return ret


def batchify_rays(rays_flat, chunk=32 * 256, **kwargs):
    """Render rays in smaller minibatches to avoid OOM errors."""
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays_us(rays_flat[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], dim=0) for k in all_ret}
    return all_ret


def render_us(
    H,
    W,
    sw,
    sh,
    chunk=1024 * 32,
    rays=None,
    c2w=None,
    near=0.0,
    far=0.055,
    **kwargs,
):
    """Render ultrasound rays."""
    if c2w is not None:
        rays_o, rays_d = get_rays_us_linear(H, W, sw, sh, c2w)
    else:
        rays_o, rays_d = rays

    sh = rays_d.shape

    # Cast and reshape rays
    rays_o = rays_o.reshape(-1, 3).float()
    rays_d = rays_d.reshape(-1, 3).float()

    # Create near and far tensors
    near = near * torch.ones_like(rays_d[..., :1])
    far = far * torch.ones_like(rays_d[..., :1])

    # Concatenate ray information
    rays = torch.cat([rays_o, rays_d, near, far], dim=-1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk=chunk, **kwargs)

    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = all_ret[k].reshape(k_sh)

    return all_ret


def to8b(x):
    """Clip and convert to 8-bit image."""
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def img2mse(x, y):
    """Compute Mean Squared Error."""
    return torch.mean((x - y) ** 2)


def create_nerf(args):
    """Instantiate NeRF's MLP model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embed_fn, input_ch = get_embedder(args.multires, args.i_embed, args.i_embed_gauss)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = args.output_ch
    skips = [4]

    model = NeRF(
        D=args.netdepth,
        W=args.netwidth,
        input_ch=input_ch,
        input_ch_views=input_ch_views,
        output_ch=output_ch,
        skips=skips,
        use_viewdirs=args.use_viewdirs,
    ).to(device)

    grad_vars = list(model.parameters())
    models = {'model': model}

    def network_query_fn(inputs, network_fn):
        return run_network(
            inputs, network_fn, embed_fn=embed_fn, netchunk=args.netchunk
        )

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'N_samples': args.N_samples,
        'network_fn': model,
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}

    start = 0
    basedir = args.basedir
    expname = args.expname

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpt_dir = os.path.join(basedir, expname)
        ckpts = [
            os.path.join(ckpt_dir, f)
            for f in sorted(os.listdir(ckpt_dir))
            if ('model_' in f and 'fine' not in f and 'optimizer' not in f)
        ]
    print('Found ckpts', ckpts)

    if len(ckpts) > 0 and not args.no_reload:
        ft_weights = ckpts[-1]
        print('Reloading from', ft_weights)
        model.load_state_dict(torch.load(ft_weights, map_location=device))
        start = int(ft_weights[-10:-4]) + 1
        print('Resetting step to', start)

    return render_kwargs_train, render_kwargs_test, start, grad_vars, models


def config_parser():
    parser = ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path', default='config_fern.txt')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', help='input data directory')

    # Training options
    parser.add_argument('--n_iters', type=int, default=10000)
    parser.add_argument("--ssim_filter_size", type=int, default=7)
    parser.add_argument("--ssim_lambda", type=float, default=0.75)
    parser.add_argument("--loss", type=str, default='l2')
    parser.add_argument('--probe_depth', type=int, default=140)
    parser.add_argument('--probe_width', type=int, default=80)
    parser.add_argument("--output_ch", type=int, default=5)
    parser.add_argument("--netdepth", type=int, default=8, help='layers in network')
    parser.add_argument("--netwidth", type=int, default=128, help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=128, help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4, help='batch size')
    parser.add_argument("--lrate", type=float, default=1e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=float, default=0.1, help='exponential learning rate decay factor')
    parser.add_argument("--chunk", type=int, default=4096 * 16, help='number of rays processed in parallel')
    parser.add_argument("--netchunk", type=int, default=4096 * 16, help='number of pts sent through network in parallel')
    parser.add_argument("--no_batching", action='store_true', help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, help='specific weights file to reload')
    parser.add_argument("--random_seed", type=int, default=None, help='fix random seed for repeatability')

    # Pre-crop options
    parser.add_argument("--precrop_iters", type=int, default=0, help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float, default=0.5, help='fraction of img taken for central crops')

    # Rendering options
    parser.add_argument("--N_samples", type=int, default=64, help='number of coarse samples per ray')
    parser.add_argument("--use_viewdirs", action='store_true', help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, help='positional encoding type')
    parser.add_argument("--i_embed_gauss", type=int, default=0, help='Gaussian positional encoding size, 0 for none')

    parser.add_argument("--multires", type=int, default=10, help='max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, help='max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.0, help='std dev of noise added to regularize sigma_a output')

    parser.add_argument("--render_only", action='store_true', help='do not optimize, reload weights and render')
    parser.add_argument("--render_test", action='store_true', help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, help='downsampling factor to speed up rendering')

    # Dataset options
    parser.add_argument("--dataset_type", type=str, default='us', help='options: us')
    parser.add_argument("--testskip", type=int, default=8, help='load 1/N images from test/val sets')

    # Logging/saving options
    parser.add_argument("--i_print", type=int, default=50, help='frequency of console printout and metric logging')
    parser.add_argument("--i_img", type=int, default=100, help='frequency of image logging')
    parser.add_argument("--i_weights", type=int, default=100, help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=5000000, help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=5000000, help='frequency of render_poses video saving')

    parser.add_argument("--log_compression", action='store_true')
    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    # Load data
    if args.dataset_type == 'us':
        images, poses, i_test = load_us_data(args.datadir)

        if not isinstance(i_test, list):
            i_test = [i_test]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print("Test {}, train {}".format(len(i_test), len(i_train)))

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Scaling factors
    scaling = 0.001
    near = 0
    probe_depth = args.probe_depth * scaling
    probe_width = args.probe_width * scaling
    far = probe_depth
    H, W = images.shape[1], images.shape[2]
    sy = probe_depth / float(H)
    sx = probe_width / float(W)
    sh = sy
    sw = sx

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, models = create_nerf(args)

    bds_dict = {
        'near': torch.tensor(near, dtype=torch.float32).to(device),
        'far': torch.tensor(far, dtype=torch.float32).to(device),
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    render_kwargs_train["args"] = args

    model = render_kwargs_train["network_fn"]
    model.to(device)

    # Create optimizer
    lrate = args.lrate
    optimizer = torch.optim.Adam(params=grad_vars, lr=lrate)

    if args.lrate_decay > 0:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lrate_decay)

    global_step = start

    N_iters = args.n_iters
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    for i in range(start, N_iters):
        time0 = time.time()
        # Sample random ray batch
        img_i = np.random.choice(i_train)
        target = torch.from_numpy(images[img_i].T).to(device).float()
        pose = torch.from_numpy(poses[img_i, :3, :4]).to(device).float()

        ssim_weight = args.ssim_lambda
        l2_weight = 1.0 - ssim_weight

        rays_o, rays_d = get_rays_us_linear(H, W, sw, sh, pose)
        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)
        batch_rays = torch.stack([rays_o, rays_d], dim=0)
        loss_dict = {}

        optimizer.zero_grad()

        # Make predictions
        rendering_output = render_us(
            H, W, sw, sh, c2w=pose, chunk=args.chunk, rays=batch_rays,
            retraw=True, **render_kwargs_train)

        output_image = rendering_output['intensity_map']

        # Compute loss
        if args.loss == 'l2':
            l2_intensity_loss = img2mse(output_image, target)
            loss_dict["l2"] = (1.0, l2_intensity_loss)
        elif args.loss == 'ssim':
            ssim_intensity_loss = 1.0 - ms_ssim(
                output_image.unsqueeze(0).unsqueeze(0),
                target.unsqueeze(0).unsqueeze(0),
                data_range=1.0,
            )
            loss_dict["ssim"] = (ssim_weight, ssim_intensity_loss)
            l2_intensity_loss = img2mse(output_image, target)
            loss_dict["l2"] = (l2_weight, l2_intensity_loss)

        total_loss = sum(weight * value for weight, value in loss_dict.values())

        total_loss.backward()
        optimizer.step()
        dt = time.time() - time0

        # Update learning rate
        if args.lrate_decay > 0:
            scheduler.step()

        # Logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, 'model_{:06d}.pt'.format(i))
            torch.save(model.state_dict(), path)
            print('Saved weights at', path)

        if i % args.i_print == 0 or i < 10:
            print(expname, i, total_loss.item(), global_step)
            print('iter time {:.05f}'.format(dt))

            # Log scalars to TensorBoard
            for k, (weight, value) in loss_dict.items():
                writer.add_scalar(f'Loss/{k}', value.item(), i)
            writer.add_scalar('Loss/Total', total_loss.item(), i)
            writer.add_scalar('Misc/LearningRate', optimizer.param_groups[0]['lr'], i)

            if i % args.i_img == 0:
                # Render and log validation image
                img_i = np.random.choice(i_val)
                target = torch.from_numpy(images[img_i].T).to(device).float()
                pose = torch.from_numpy(poses[img_i, :3, :4]).to(device).float()

                rendering_output_test = render_us(
                    H, W, sw, sh, chunk=args.chunk, c2w=pose, **render_kwargs_test)

                output_image_test = rendering_output_test['intensity_map']

                # Compute validation loss
                if args.loss == 'l2':
                    l2_intensity_loss_test = img2mse(output_image_test, target)
                    loss_dict_test = {"l2": (1.0, l2_intensity_loss_test)}
                elif args.loss == 'ssim':
                    ssim_intensity_loss_test = 1.0 - ms_ssim(
                        output_image_test.unsqueeze(0).unsqueeze(0),
                        target.unsqueeze(0).unsqueeze(0),
                        data_range=1.0,
                    )
                    l2_intensity_loss_test = img2mse(output_image_test, target)
                    loss_dict_test = {
                        "ssim": (ssim_weight, ssim_intensity_loss_test),
                        "l2": (l2_weight, l2_intensity_loss_test),
                    }

                total_loss_test = sum(weight * value for weight, value in loss_dict_test.values())

                # Save validation images
                testimgdir = os.path.join(basedir, expname, 'tboard_val_imgs')
                os.makedirs(testimgdir, exist_ok=True)
                imageio.imwrite(
                    os.path.join(testimgdir, '{:06d}.png'.format(i)),
                    to8b(output_image_test.cpu().numpy().T),
                )

                # Log images to TensorBoard
                writer.add_image(
                    'Validation/Output',
                    to8b(output_image_test.cpu().numpy().T),
                    i, dataformats='HW',
                )
                writer.add_image(
                    'Validation/Target',
                    to8b(target.cpu().numpy()),
                    i, dataformats='HW',
                )

                # Log validation losses
                for k, (weight, value) in loss_dict_test.items():
                    writer.add_scalar(f'Validation/{k}', value.item(), i)
                writer.add_scalar('Validation/Total', total_loss_test.item(), i)

        global_step += 1

    writer.close()


if __name__ == '__main__':
    train()
