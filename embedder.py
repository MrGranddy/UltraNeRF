from typing import Callable, List, Tuple

import torch

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
