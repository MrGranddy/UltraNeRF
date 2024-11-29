import torch
import torch.nn as nn
import numpy as np
import io
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple

def to8b(x):
    """Clip and convert to 8-bit image
    
    Args:
        x (np.ndarray): Input image array
    
    Returns:
        np.ndarray: 8-bit image
    """
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)

def show_colorbar(image: torch.Tensor, cmap: str = 'rainbow') -> io.BytesIO:
    """
    Generate a colorbar visualization of an image and save it to a BytesIO buffer.

    Args:
        image (torch.Tensor): Input image
        cmap (str, optional): Colormap to use. Defaults to 'rainbow'.

    Returns:
        io.BytesIO: PNG image buffer with colorbar
    """
    figure = plt.figure(figsize=(5, 5))
    plt.imshow(image.numpy(), cmap=cmap)
    plt.colorbar()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    return buf

def img2mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculate Mean Squared Error between two tensors."""
    return torch.mean((x - y)**2)

def mse2psnr(x: torch.Tensor) -> torch.Tensor:
    """Convert Mean Squared Error to Peak Signal-to-Noise Ratio."""
    return -10. * torch.log10(x)


def define_image_grid_3D_np(x_size: int, y_size: int) -> np.ndarray:
    """
    Define a 3D image grid with zero z-coordinate.

    Args:
        x_size (int): Size of x dimension
        y_size (int): Size of y dimension

    Returns:
        np.ndarray: 3D image grid coordinates
    """
    y = np.array(range(x_size))
    x = np.array(range(y_size))
    xv, yv = np.meshgrid(x, y, indexing='ij')
    image_grid_xy = np.vstack((xv.ravel(), yv.ravel()))
    z = np.zeros(image_grid_xy.shape[1])
    return np.vstack((image_grid_xy, z))

def get_rays_us_linear(
    H: int, 
    W: int, 
    sw: float, 
    sh: float, 
    c2w: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rays for uniform sampling in a linear configuration.

    Args:
        H (int): Image height
        W (int): Image width
        sw (float): Scaling factor for width
        sh (float): Scaling factor for height
        c2w (torch.Tensor): Camera-to-world transformation matrix

    Returns:
        Tuple of ray origins and ray directions
    """
    t = c2w[:3, -1]
    R = c2w[:3, :3]
    
    x = torch.linspace(-W/2, W/2, W) * sw
    y = torch.zeros_like(x)
    z = torch.zeros_like(x)

    origin_base = torch.stack([x, y, z], dim=1)
    origin_base_prim = origin_base[..., None, :]
    origin_rotated = R @ origin_base_prim # THIS WAS HADAMARD PRODUCT IN THE ORIGINAL CODE !!!
    rays_o = origin_rotated.sum(dim=-1) + t

    dirs_base = torch.tensor([0., 1., 0.])
    dirs_r = R @ dirs_base.reshape(-1, 1)
    rays_d = dirs_r.expand(rays_o.shape)

    return rays_o, rays_d
