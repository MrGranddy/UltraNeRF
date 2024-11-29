from typing import List

import torch
import torch.nn as nn

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

