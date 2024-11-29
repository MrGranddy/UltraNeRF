import os

import imageio
import time

from run_nerf_helpers import *
from load_us import load_us_data

import torch
import torch.nn as nn
import torch.nn.functional as F

def gaussian_kernel(
    size: int, 
    mean: float, 
    std: float
) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel using two normal distributions.
    
    Args:
        size (int): Half-size of the kernel (total size will be 2*size + 1)
        mean (float): Mean of the Gaussian distributions
        std (float): Standard deviation of the Gaussian distributions
    
    Returns:
        torch.Tensor: Normalized 2D Gaussian kernel
    """
    delta_t = 1.0  # 9.197324e-01

    # Create range and scale by delta_t
    x_cos = torch.tensor(list(range(-size, size + 1)), dtype=torch.float32)
    x_cos *= delta_t

    # Create normal distributions
    d1 = torch.distributions.Normal(mean, std * 2.0)
    d2 = torch.distributions.Normal(mean, std)

    # Compute probabilities 
    vals_x = d1.log_prob(torch.linspace(start=-size, end=size, steps=2*size+1, dtype=torch.float32) * delta_t).exp()
    vals_y = d2.log_prob(torch.linspace(start=-size, end=size, steps=2*size+1, dtype=torch.float32) * delta_t).exp()

    # Outer product (equivalent to tf.einsum)
    gauss_kernel = torch.outer(vals_x, vals_y)

    # Normalize
    return gauss_kernel / gauss_kernel.sum()


g_size = 3
g_mean = 0.
g_variance = 1.
g_kernel = gaussian_kernel(g_size, g_mean, g_variance)
g_kernel = g_kernel.unsqueeze(0).unsqueeze(0)


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    
    Args:
        fn (Callable): Function to apply
        chunk (int): Size of the batch

    Returns:
        Callable: Batchified function
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret

def run_network(inputs, fn, embed_fn, netchunk=512 * 32):
    """Prepares inputs and applies network 'fn'.

    Args:
        inputs (torch.Tensor): Input tensor
        fn (Callable): Network function
        embed_fn (Callable): Embedding function
        netchunk (int): Network chunk size

    Returns:
        torch.Tensor: Output tensor
    """
    inputs_flat = inputs.view(-1, inputs.shape[-1])

    embedded = embed_fn(inputs_flat)
    outputs_flat = batchify(fn, netchunk)(embedded)

    outputs = outputs_flat.view(list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])

    return outputs

def render_method_convolutional_ultrasound(raw, z_vals):
    """ Render method for ultrasound converted to PyTorch

    Args:
        raw (torch.Tensor): Raw output tensor
        z_vals (torch.Tensor): Z values tensor

    Returns:
        dict: Dictionary of rendering components
    """

    # Compute distance between points
    dists = torch.abs(z_vals[..., :-1] - z_vals[..., 1:])
    dists = torch.cat([dists, dists[:, -1, None]], dim=-1)

    # ATTENUATION
    # Predict attenuation coefficient for each sampled point. This value is positive.
    attenuation_coeff = torch.abs(raw[..., 0])
    attenuation = -attenuation_coeff * dists
    # Compute total attenuation at each pixel location as exp{-sum[a_n*d_n]}
    attenuation_transmission = np.exp(torch.cumsum(attenuation, dim=1))
    
    # REFLECTION
    prob_border = torch.sigmoid(raw[..., 2])

    # Replace TensorFlow's Bernoulli distribution sampling
    border_indicator = (torch.rand_like(prob_border) < prob_border).float()
    
    # Predict reflection coefficient. This value is between (0, 1).
    reflection_coeff = torch.sigmoid(raw[..., 1])
    
    # Compute reflection transmission
    reflection_transmission = torch.log((1. - reflection_coeff) * border_indicator + 1.e-8)
    reflection_transmission = np.exp(torch.cumsum(reflection_transmission, dim=1))
    
    # Border convolution using the pre-defined Gaussian kernel
    border_convolution = F.conv2d(
        border_indicator, 
        g_kernel, 
        padding='same'
    )

    # BACKSCATTERING
    density_coeff_value = torch.sigmoid(raw[..., 3])
    density_coeff = torch.ones_like(reflection_coeff) * density_coeff_value
    
    # Replace TensorFlow's Bernoulli distribution sampling
    scatterers_density = (torch.rand_like(density_coeff) < density_coeff).float()
    
    # Predict scattering amplitude
    amplitude = torch.sigmoid(raw[..., 4])
    
    # Compute scattering template
    scatterers_map = scatterers_density * amplitude
    
    # Scattering point spread function convolution
    psf_scatter = F.conv2d(
        scatterers_map[:, :], 
        g_kernel, 
        padding='same'
    )

    # Compute remaining intensity at a point n
    transmission = attenuation_transmission * reflection_transmission
    
    # Compute backscattering part of the final echo
    b = transmission * psf_scatter
    
    # Compute reflection part of the final echo
    r = transmission * reflection_coeff * border_convolution
    
    # Compute the final echo
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
        "transmission": transmission
    }

def render_rays_us(
    ray_batch,
    network_fn,
    network_query_fn,
    N_samples,
    retraw=False,
    lindisp=False,
    args=None
):
    """Volumetric rendering for ultrasound rays.

    Args:
        ray_batch (torch.Tensor): Tensor of shape [batch_size, ...] representing rays
        network_fn (callable): Network function to process points
        network_query_fn (callable): Function to query network with points
        N_samples (int): Number of samples along each ray
        retraw (bool, optional): Whether to return raw network output. Defaults to False.
        lindisp (bool, optional): Whether to sample linearly in inverse depth. Defaults to False.
        args (object, optional): Additional arguments. Defaults to None.

    Returns:
        dict: Rendering results
    """
    def raw2outputs(raw, z_vals, rays_d):
        """Transforms model's predictions to semantically meaningful values."""
        return render_method_convolutional_ultrasound(raw, z_vals)

    # Validate input
    if not isinstance(ray_batch, torch.Tensor):
        raise TypeError("ray_batch must be a torch.Tensor")

    # Batch size
    N_rays = ray_batch.shape[0]

    # Extract ray origin, direction
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each

    # Extract unit-normalized viewing direction
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

    # Extract lower, upper bound for ray distance
    bounds = ray_batch[..., 6:8].reshape(-1, 1, 2)
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    # Create sample points along rays
    t_vals = torch.linspace(0., 1., N_samples, device=ray_batch.device)

    if not lindisp:
        # Space integration times linearly between 'near' and 'far'
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity)
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    # Broadcast z_vals to match ray batch
    z_vals = z_vals.broadcast_to(N_rays, N_samples)

    # Compute points in space to evaluate model
    origin = rays_o[..., None, :]
    step = rays_d[..., None, :] * z_vals[..., :, None]
    pts = step + origin

    try:
        # Evaluate model at each point
        raw = network_query_fn(pts, network_fn)  # [N_rays, N_samples, 5]

        # Transform raw output
        ret = raw2outputs(raw, z_vals, rays_d)

        # Optionally include raw output
        if retraw:
            ret['raw'] = raw

        # Numerical stability check (PyTorch equivalent)
        for k, v in ret.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in output {k}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in output {k}")

        return ret

    except Exception as e:
        raise RuntimeError(f"Error in render_rays_us: {str(e)}")


def batchify_rays(rays_flat, c2w=None, chunk=32 * 256, **kwargs):
    """Render rays in smaller minibatches to avoid Out of Memory errors.
    
    Args:
        rays_flat (torch.Tensor): Flattened ray tensor
        c2w (torch.Tensor, optional): Camera to world transformation matrix
        chunk (int, optional): Size of mini-batches
        **kwargs: Additional arguments for render_rays_us
    
    Returns:
        dict: Concatenated rendering results
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays_us(rays_flat[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    # Concatenate results along the first dimension
    all_ret = {k: torch.cat(all_ret[k], dim=0) for k in all_ret}
    return all_ret


def render_us(H, W, sw, sh,
              chunk=1024 * 32, rays=None, c2w=None,
              near=0., far=55. * 0.001,
              **kwargs):
    """Render ultrasound rays
    
    Args:
        H (int): Image height
        W (int): Image width
        sw (float): Sensor width
        sh (float): Sensor height
        chunk (int, optional): Rendering chunk size
        rays (tuple, optional): Precomputed rays
        c2w (torch.Tensor, optional): Camera to world transformation matrix
        near (float, optional): Near clipping plane
        far (float, optional): Far clipping plane
    
    Returns:
        dict: Rendering results
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays_us_linear(H, W, sw, sh, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    sh = rays_d.shape  # [..., 3]

    # Cast and reshape rays
    rays_o = rays_o.reshape(-1, 3).float()
    rays_d = rays_d.reshape(-1, 3).float()
    
    # Create near and far tensors
    near = torch.ones_like(rays_d[..., :1]) * near
    far = torch.ones_like(rays_d[..., :1]) * far

    # Concatenate ray information
    rays = torch.cat([rays_o, rays_d, near, far], dim=-1)

    # Render and reshape
    all_ret = batchify_rays(rays, c2w=c2w, chunk=chunk, **kwargs)
    
    # Reshape results to match original image shape
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = all_ret[k].reshape(k_sh)
    
    return all_ret





def create_nerf(args):
    """Instantiate NeRF's MLP model."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    embed_fn, input_ch = get_embedder(args.multires, args.i_embed, args.i_embed_gauss)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = args.output_ch
    skips = [4]

    model = NeRF(
        D=args.netdepth, W=args.netwidth,
        input_ch=input_ch, input_ch_views=input_ch_views,
        output_ch=output_ch, skips=skips,
        use_viewdirs=args.use_viewdirs).to(device)


    grad_vars = model.trainable_variables
    models = {'model': model}

    def network_query_fn(inputs, network_fn):
        return run_network(
            inputs, network_fn,
            embed_fn=embed_fn,
            netchunk=args.netchunk)

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'N_samples': args.N_samples,
        'network_fn': model
    }

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}

    start = 0
    basedir = args.basedir
    expname = args.expname

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 ('model_' in f and 'fine' not in f and 'optimizer' not in f)]
    print('Found ckpts', ckpts)

    if len(ckpts) > 0 and not args.no_reload:
        ft_weights = ckpts[-1]
        print('Reloading from', ft_weights)
        model.load_state_dict(torch.load(ft_weights, map_location=torch.device(device)))
        start = int(ft_weights[-10:-4]) + 1
        print('Resetting step to', start)
        #
        # if model_fine is not None:
        #     ft_weights_fine = '{}_fine_{}'.format(
        #         ft_weights[:-11], ft_weights[-10:])
        #     print('Reloading fine from', ft_weights_fine)
        #     model_fine.set_weights(np.load(ft_weights_fine, allow_pickle=True))

    return render_kwargs_train, render_kwargs_test, start, grad_vars, models


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path', default='config_fern.txt')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str,
                        default='./data/llff/fern', help='input data directory')

    # training options
    parser.add_argument('--n_iters', type=int, default=10000)
    parser.add_argument("--ssim_filter_size", type=int, default=7)
    parser.add_argument("--ssim_lambda", type=float, default=0.75)
    parser.add_argument("--loss", type=str, default='l2')
    parser.add_argument('--probe_depth', type=int, default=140)
    parser.add_argument('--probe_width', type=int, default=80)
    parser.add_argument("--output_ch", type=int, default=5),
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=128,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int,
                        default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=128,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000s)')
    parser.add_argument("--chunk", type=int, default=4096 * 16,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=4096 * 16,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--random_seed", type=int, default=None,
                        help='fix random seed for repeatability')

    # pre-crop options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--i_embed_gauss", type=int, default=0,
                        help='mapping size for Gaussian positional encoding, 0 for none')

    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='us',
                        help='options: us')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=50,
                        help='frequency of console printout and metric logging')
    parser.add_argument("--i_img", type=int, default=100,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=100,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=5000000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=5000000,
                        help='frequency of render_poses video saving')

    parser.add_argument("--log_compression", action='store_true')
    return parser


def main():
    args = parse_arguments()
    set_random_seed(args)
    data = load_data(args)
    if data is None:
        return
    create_logging_dirs(args)
    render_kwargs_train, render_kwargs_test, start, grad_vars, models = create_model(args, data)
    optimizer, global_step = setup_optimizer(args, grad_vars, start, models)
    train(args, data, models, optimizer, grad_vars, start, render_kwargs_train, render_kwargs_test, global_step)

def parse_arguments():
    parser = config_parser()
    args = parser.parse_args()
    return args

def set_random_seed(args):
    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

def load_data(args):
    if args.dataset_type == 'us':
        images, poses, i_test = load_us_data(args.datadir)

        if not isinstance(i_test, list):
            i_test = [i_test]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print("Test {}, train {}".format(len(i_test), len(i_train)))
        return images, poses, i_train, i_val, i_test
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return None

def create_logging_dirs(args):
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    args_file = os.path.join(basedir, expname, 'args.txt')
    with open(args_file, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        config_file = os.path.join(basedir, expname, 'config.txt')
        with open(config_file, 'w') as file:
            file.write(open(args.config, 'r').read())

def create_model(args, data):
    images, poses, i_train, i_val, i_test = data
    render_kwargs_train, render_kwargs_test, start, grad_vars, models = create_nerf(args)

    # Scale settings
    scaling = 0.001
    near = 0
    probe_depth = args.probe_depth * scaling
    probe_width = args.probe_width * scaling
    far = probe_depth
    H, W = images.shape[1], images.shape[2]

    bds_dict = {
        'near': torch.tensor(near).float(),
        'far': torch.tensor(far).float()
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    render_kwargs_train["args"] = args

    model = render_kwargs_train["model"]
    model.summary()

    return render_kwargs_train, render_kwargs_test, start, grad_vars, models

def setup_optimizer(model, lrate, lrate_decay):

    optimizer = torch.optim.Adam(model.parameters(), lr=lrate, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lrate_decay)

    return optimizer, global_step

def train(args, data, models, optimizer, grad_vars, start, render_kwargs_train, render_kwargs_test, global_step):
    images, poses, i_train, i_val, i_test = data
    H, W = images.shape[1], images.shape[2]
    scaling = 0.001
    near = 0
    probe_depth = args.probe_depth * scaling
    probe_width = args.probe_width * scaling
    sy = probe_depth / float(H)
    sx = probe_width / float(W)
    sh = sy
    sw = sx

    N_iters = args.n_iters
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writer
    writer = tf.summary.create_file_writer(
        os.path.join(args.basedir, 'summaries', args.expname))

    for i in range(start, N_iters):
        time0 = time.time()
        # Sample random ray batch
        img_i = np.random.choice(i_train)
        try:
            target = tf.transpose(images[img_i])
        except Exception as e:
            print(f"Error processing image {img_i}: {e}")
            continue

        pose = poses[img_i, :3, :4]
        ssim_weight = args.ssim_lambda
        l2_weight = 1. - ssim_weight

        rays_o, rays_d = get_rays_us_linear(H, W, sw, sh, pose)
        batch_rays = tf.stack([rays_o, rays_d], 0)
        loss = dict()

        #####  Core optimization loop  #####
        with tf.GradientTape() as tape:
            # Make predictions
            rendering_output = render_us(
                H, W, sw, sh, c2w=pose, chunk=args.chunk, rays=batch_rays,
                retraw=True, **render_kwargs_train)

            output_image = rendering_output['intensity_map']
            if args.loss == 'l2':
                l2_intensity_loss = img2mse(output_image, target)
                loss["l2"] = (1., l2_intensity_loss)
            elif args.loss == 'ssim':
                ssim_intensity_loss = 1. - tf.image.ssim_multiscale(
                    tf.expand_dims(tf.expand_dims(output_image, 0), -1),
                    tf.expand_dims(tf.expand_dims(target, 0), -1),
                    max_val=1.0, filter_size=args.ssim_filter_size,
                    filter_sigma=1.5, k1=0.01, k2=0.1
                )
                loss["ssim"] = (ssim_weight, ssim_intensity_loss)
                l2_intensity_loss = img2mse(output_image, target)
                loss["l2"] = (l2_weight, l2_intensity_loss)

            total_loss = sum(w * v for w, v in loss.values())

        gradients = tape.gradient(total_loss, grad_vars)
        optimizer.apply_gradients(zip(gradients, grad_vars))
        dt = time.time() - time0

        # Rest is logging
        if i % args.i_weights == 0:
            for k in models:
                save_weights(models[k], k, i, args)

        if i % args.i_print == 0 or i < 10:
            print(args.expname, i, total_loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))
            with writer.as_default():
                current_learning_rate = optimizer._decayed_lr(tf.float32)
                tf.summary.scalar('misc/learning_rate', current_learning_rate, step=global_step)
                loss_string = "Total loss ="
                for l_key, (weight, value) in loss.items():
                    loss_string += f' + {weight} * {l_key}'
                    tf.summary.scalar(f'train/loss_{l_key}', value, step=global_step)
                    tf.summary.scalar(f'train/penalty_factor_{l_key}', weight, step=global_step)
                    tf.summary.scalar(f'train/total_loss_{l_key}', weight * value, step=global_step)
                tf.summary.scalar('train/total_loss', total_loss, step=global_step)
                print(loss_string)

        if i % args.i_img == 0:
            log_validation_view(args, images, poses, i_val, render_kwargs_test, writer, global_step, ssim_weight, l2_weight)

        global_step.assign_add(1)

def save_weights(net, prefix, i, args):
    path = os.path.join(
        args.basedir, args.expname, '{}_{:06d}.npy'.format(prefix, i))
    np.save(path, net.get_weights())
    print('saved weights at', path)

def log_validation_view(args, images, poses, i_val, render_kwargs_test, writer, global_step, ssim_weight, l2_weight):
    img_i = np.random.choice(i_val)
    target = tf.transpose(images[img_i])
    pose = poses[img_i, :3, :4]
    H, W = images.shape[1], images.shape[2]
    scaling = 0.001
    probe_depth = args.probe_depth * scaling
    probe_width = args.probe_width * scaling
    sy = probe_depth / float(H)
    sx = probe_width / float(W)
    sh = sy
    sw = sx

    rendering_output_test = render_us(H, W, sw, sh, chunk=args.chunk, c2w=pose, **render_kwargs_test)
    output_image_test = rendering_output_test['intensity_map']

    loss_holdout = {}
    if args.loss == 'l2':
        l2_intensity_loss = img2mse(output_image_test, target)
        loss_holdout["l2"] = (1., l2_intensity_loss)
    elif args.loss == 'ssim':
        ssim_intensity_loss = 1. - tf.image.ssim_multiscale(
            tf.expand_dims(tf.expand_dims(output_image_test, 0), -1),
            tf.expand_dims(tf.expand_dims(target, 0), -1),
            max_val=1.0, filter_size=args.ssim_filter_size,
            filter_sigma=1.5, k1=0.01, k2=0.1
        )
        loss_holdout["ssim"] = (ssim_weight, ssim_intensity_loss)
        l2_intensity_loss = img2mse(output_image_test, target)
        loss_holdout["l2"] = (l2_weight, l2_intensity_loss)

    total_loss_holdout = sum(w * v for w, v in loss_holdout.values())

    testimgdir = os.path.join(args.basedir, args.expname, 'tboard_val_imgs')
    os.makedirs(testimgdir, exist_ok=True)
    imageio.imwrite(os.path.join(testimgdir,
                                 '{:06d}.png'.format(global_step.numpy())), to8b(tf.transpose(output_image_test)))

    with writer.as_default():
        tf.summary.image('b_mode/output',
                         tf.expand_dims(tf.expand_dims(to8b(tf.transpose(output_image_test)), 0), -1),
                         step=global_step)
        for l_key, (weight, value) in loss_holdout.items():
            tf.summary.scalar(f'test/loss_{l_key}', value, step=global_step)
            tf.summary.scalar(f'test/penalty_factor_{l_key}', weight, step=global_step)
            tf.summary.scalar(f'test/total_loss_{l_key}', weight * value, step=global_step)
        tf.summary.scalar('test/total_loss', total_loss_holdout, step=global_step)
        tf.summary.image('b_mode/target',
                         tf.expand_dims(tf.expand_dims(to8b(tf.transpose(target)), 0), -1),
                         step=global_step)
        for map_k, map_v in rendering_output_test.items():
            colorbar_image = show_colorbar(tf.transpose(map_v)).getvalue()
            tf.summary.image(f'maps/{map_k}',
                             tf.expand_dims(tf.image.decode_png(colorbar_image, channels=4), 0),
                             step=global_step)

if __name__ == '__main__':
    main()