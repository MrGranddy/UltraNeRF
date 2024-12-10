import os
import torch
import numpy as np
import imageio
import pprint
import pathlib
import argparse

import matplotlib.pyplot as plt

import run_ultranerf  # Ensure this is the PyTorch version
from load_us import load_us_data

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store checkpoints and logs')
    parser.add_argument("--expname", type=str, required=True,
                        help='experiment name')
    parser.add_argument("--model_no", type=str, required=True,
                        help='model checkpoint to use, e.g., model_200000')
    parser.add_argument("--datadir", type=str, default='./data/us_data',
                        help='input data directory')
    parser.add_argument("--output_dir", type=str, default=None,
                        help='directory to save outputs, default is basedir/expname/output_maps')
    parser.add_argument("--downsample", type=int, default=1,
                        help='downsampling factor for rendering')
    parser.add_argument("--save_interval", type=int, default=300,
                        help='interval at which to save parameters')
    return parser

def show_colorbar(image, name=None, cmap='rainbow'):
    figure = plt.figure()
    plt.imshow(image, cmap=cmap)
    plt.tick_params(top=False, bottom=False, left=False, right=False,
                    labelleft=False, labelbottom=False)
    m = plt.cm.ScalarMappable(cmap=cmap)
    m.set_clim(0., 1.)
    plt.colorbar(m)
    if name:
        figure.savefig(name)
    plt.close(figure)

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    basedir = args.basedir
    expname = args.expname
    model_no = args.model_no
    datadir = args.datadir
    output_dir = args.output_dir
    down = args.downsample
    save_it = args.save_interval

    if output_dir is None:
        output_dir = os.path.join(basedir, expname, f'output_maps_{model_no}')

    config = os.path.join(basedir, expname, 'config.txt')
    print('Args:')
    with open(config, 'r') as f:
        print(f.read())

    # Parse the config file
    parser = run_ultranerf.config_parser()
    args_list = ['--config', config]
    args_render = parser.parse_args(args_list)
    args_render.ft_path = os.path.join(basedir, expname, model_no + '.tar')

    print('Loaded args')
    model_name = os.path.basename(args_render.datadir)
    images, poses, i_test = load_us_data(args_render.datadir)
    H, W = images[0].shape
    H = int(H)
    W = int(W)

    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    near = 0.
    far = args_render.probe_depth * 0.001

    # Create NeRF model
    _, render_kwargs_test, start, grad_vars, models = run_ultranerf.create_nerf(args_render)
    render_kwargs_test["args"] = args_render
    bds_dict = {
        'near': torch.tensor(near, dtype=torch.float32, device=device),
        'far': torch.tensor(far, dtype=torch.float32, device=device),
    }
    render_kwargs_test.update(bds_dict)

    print('Render kwargs:')
    pprint.pprint(render_kwargs_test)
    sw = args_render.probe_width * 0.001 / float(W)
    sh = args_render.probe_depth * 0.001 / float(H)

    render_kwargs_fast = {k: render_kwargs_test[k] for k in render_kwargs_test}

    # Set up output directories
    output_dir_params = os.path.join(output_dir, 'params')
    output_dir_output = os.path.join(output_dir, 'output')
    os.makedirs(output_dir_params, exist_ok=True)
    os.makedirs(output_dir_output, exist_ok=True)

    rendering_params_save = None
    for i, c2w in enumerate(poses):
        print(f"Rendering frame {i}")

        c2w = torch.from_numpy(c2w[:3, :4]).float().to(device)
        rendering_params = run_ultranerf.render_us(H, W, sw, sh, c2w=c2w, **render_kwargs_fast)
        output_image = rendering_params['intensity_map'].detach().cpu().numpy().transpose()
        output_image_uint8 = (output_image * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(output_dir_output, f"Generated_{i}.png"),
                        output_image_uint8)

        # Collect rendering parameters
        if rendering_params_save is None:
            rendering_params_save = {key: [] for key in rendering_params.keys()}

        for key in rendering_params:
            value = rendering_params[key].detach().cpu().numpy().transpose()
            rendering_params_save[key].append(value)

        # Save parameters at intervals
        if (i + 1) % save_it == 0:
            for key, value_list in rendering_params_save.items():
                np_to_save = np.array(value_list)
                f_name = os.path.join(output_dir_params, f"{key}.npy")
                if os.path.exists(f_name):
                    np_existing = np.load(f_name)
                    np_to_save = np.concatenate((np_existing, np_to_save), axis=0)
                np.save(f_name, np_to_save)
            rendering_params_save = {key: [] for key in rendering_params.keys()}

    # Save any remaining parameters
    if rendering_params_save:
        for key, value_list in rendering_params_save.items():
            np_to_save = np.array(value_list)
            f_name = os.path.join(output_dir_params, f"{key}.npy")
            if os.path.exists(f_name):
                np_existing = np.load(f_name)
                np_to_save = np.concatenate((np_existing, np_to_save), axis=0)
            np.save(f_name, np_to_save)
