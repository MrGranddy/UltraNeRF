import torch
import torch.nn as nn
import numpy as np
import io
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple

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

def to8b(x: np.ndarray) -> np.ndarray:
    """Convert input to 8-bit unsigned integer array."""
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)

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

class Embedder:
    """Positional encoding embedder for neural networks."""

    def __init__(self, **kwargs):
        """
        Initialize embedder with configuration parameters.

        Args:
            kwargs: Configuration parameters for embedding function
        """
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        """Create embedding functions based on configuration."""
        embed_fns: List[Callable] = []
        d = self.kwargs['input_dims']
        out_dim = 0

        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        B = self.kwargs['B']

        freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs) if self.kwargs['log_sampling'] \
            else torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                if B is not None:
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq, B=B: p_fn(x @ torch.t(B) * freq))
                    out_dim += d + B.shape[1]
                else:
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                    out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Embed input tensor using created embedding functions.

        Args:
            inputs (torch.Tensor): Input tensor to embed

        Returns:
            torch.Tensor: Embedded output tensor
        """
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(
    multires: int, 
    i: int = 0, 
    b: int = 0,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], int]:
    """
    Get positional encoding embedder.

    Args:
        multires (int): Number of frequency bands
        i (int, optional): Special flag for identity embedding. Defaults to 0.
        b (int, optional): Number of random frequencies. Defaults to 0.

    Returns:
        Tuple containing embedding function and output dimension
    """
    if i == -1:
        return torch.nn.Identity(), 3

    B = torch.randn(b, 3) if b != 0 else None

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
        'B': B
    }
    embedder_obj = Embedder(**embed_kwargs)
    def embed(x: torch.Tensor, eo: Embedder = embedder_obj) -> torch.Tensor:
        return eo.embed(x)
    
    return embed, embedder_obj.out_dim

class NeRF(nn.Module):

    def __init__(
        self,
        D: int = 8, 
        W: int = 256, 
        input_ch: int = 3, 
        input_ch_views: int = 3, 
        output_ch: int = 6, 
        skips: List[int] = [4], 
        use_viewdirs: bool = False
    ):
        """
        Initialize neural radiance field (NeRF) model.

        Args:
            D (int, optional): Number of layers. Defaults to 8.
            W (int, optional): Layer width. Defaults to 256.
            input_ch (int, optional): Input coordinate channels. Defaults to 3.
            input_ch_views (int, optional): Input view direction channels. Defaults to 3.
            output_ch (int, optional): Output channel dimension. Defaults to 6.
            skips (List[int], optional): Skip connection layers. Defaults to [4].
            use_viewdirs (bool, optional): Whether to use view directions. Defaults to False.
        """

        super(NeRF, self).__init__()
        relu = nn.ReLU()

        self.use_viewdirs = use_viewdirs
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views

        self.skips = skips

        self.layers = nn.ModuleList([])
        inner_in_chns = [input_ch] + [W if i - 1 in skips else W + input_ch for i in range(1, D)]
        for i in range(D):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(inner_in_chns[i], W),
                    relu
                )
            )

        inner_out_ch = W if D - 1 not in skips else W + input_ch

        if use_viewdirs:
            self.alpha_layer = nn.Linear(inner_out_ch, 1)
            self.bottleneck_layer = nn.Linear(inner_out_ch, 256)
            self.output_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(256 + input_ch_views, W//2),
                    relu
                ),
            ])
            for _ in range(3):
                self.output_layers.append(
                    nn.Sequential(
                        nn.Linear(W//2, W//2),
                        relu
                    )
                )
            self.output_layers.append(
                nn.Linear(W//2, output_ch - 1)
            )
        else:
            self.output_layers = nn.ModuleList([
                nn.Linear(inner_out_ch, output_ch)
            ])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of NeRF model.

        Args:
            inputs (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        inputs_pts, inputs_views = torch.split(inputs, [self.input_ch, self.input_ch_views], -1)

        outputs = inputs_pts
        for i, layer in enumerate(self.layers):
            outputs = layer(outputs)
            if i in self.skips:
                outputs = torch.cat([inputs_pts, outputs], -1)

        if self.use_viewdirs:
            alpha_out = self.alpha_layer(outputs)
            bottleneck = self.bottleneck_layer(outputs)
            inputs_viewdirs = torch.cat([bottleneck, inputs_views], -1)

            outputs = inputs_viewdirs
            for layer in self.output_layers:
                outputs = layer(outputs)
            outputs = torch.cat([outputs, alpha_out], -1)

        else:
            for layer in self.output_layers:
                outputs = layer(outputs)

        return outputs


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
