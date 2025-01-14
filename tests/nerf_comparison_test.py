import os
from typing import List

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


tf.enable_eager_execution()  # Enable eager execution for TensorFlow


def init_nerf_model(
    D=8, W=256, input_ch=3, input_ch_views=3, output_ch=6, skips=[4], use_viewdirs=False
):

    relu = tf.keras.layers.ReLU()

    def dense(W, act=relu):
        return tf.keras.layers.Dense(W, activation=act)

    #                                                     bias_initializer=tf.keras.initializers.RandomNormal(mean=-0.0, stddev=1.),
    #                                                     kernel_initializer=tf.keras.initializers.RandomNormal(mean=-0.0, stddev=1.))

    print(
        "MODEL",
        input_ch,
        input_ch_views,
        type(input_ch),
        type(input_ch_views),
        use_viewdirs,
    )
    input_ch = int(input_ch)
    input_ch_views = int(input_ch_views)

    inputs = tf.keras.Input(shape=(input_ch + input_ch_views))
    inputs_pts, inputs_views = tf.split(inputs, [input_ch, input_ch_views], -1)
    inputs_pts.set_shape([None, input_ch])
    inputs_views.set_shape([None, input_ch_views])

    print(inputs.shape, inputs_pts.shape, inputs_views.shape)
    outputs = inputs_pts
    print("input {}".format(inputs_pts.shape))
    for i in range(D):
        outputs = dense(W)(outputs)
        print("{} layer, {} shape".format(i, outputs.shape))
        if i in skips:
            outputs = tf.concat([inputs_pts, outputs], -1)

    if use_viewdirs:
        alpha_out = dense(1, act=None)(outputs)
        bottleneck = dense(256, act=None)(outputs)
        inputs_viewdirs = tf.concat([bottleneck, inputs_views], -1)
        outputs = inputs_viewdirs
        for i in range(4):
            outputs = dense(W // 2)(outputs)
        outputs = dense(output_ch - 1, act=None)(outputs)
        outputs = tf.concat([outputs, alpha_out], -1)

    else:
        outputs = dense(output_ch, act=None)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


class NeRF(nn.Module):

    def __init__(
        self,
        D: int = 8,
        W: int = 256,
        input_ch: int = 3,
        input_ch_views: int = 3,
        output_ch: int = 6,
        skips: List[int] = [4],
        use_viewdirs: bool = False,
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
        inner_in_chns = [input_ch] + [
            W if i - 1 in skips else W + input_ch for i in range(1, D)
        ]
        for i in range(D):
            self.layers.append(nn.Sequential(nn.Linear(inner_in_chns[i], W), relu))

        inner_out_ch = W if D - 1 not in skips else W + input_ch

        if use_viewdirs:
            self.alpha_layer = nn.Linear(inner_out_ch, 1)
            self.bottleneck_layer = nn.Linear(inner_out_ch, 256)
            self.output_layers = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(256 + input_ch_views, W // 2), relu),
                ]
            )
            for _ in range(3):
                self.output_layers.append(
                    nn.Sequential(nn.Linear(W // 2, W // 2), relu)
                )
            self.output_layers.append(nn.Linear(W // 2, output_ch - 1))
        else:
            self.output_layers = nn.ModuleList([nn.Linear(inner_out_ch, output_ch)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of NeRF model.

        Args:
            inputs (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        inputs_pts, inputs_views = torch.split(
            inputs, [self.input_ch, self.input_ch_views], -1
        )

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


def compare_models():
    # Test scenarios
    scenarios = [
        {
            "name": "Basic 3D Coordinates (No View Directions)",
            "input_ch": 3,
            "input_ch_views": 0,
            "use_viewdirs": False,
            "output_ch": 4,
        },
        {
            "name": "Full NeRF with View Directions",
            "input_ch": 3,
            "input_ch_views": 3,
            "use_viewdirs": True,
            "output_ch": 6,
        },
    ]

    for scenario in scenarios:
        print(f"\nTesting Scenario: {scenario['name']}")

        # TensorFlow Model
        tf_model = init_nerf_model(
            D=scenario.get("D", 8),
            W=scenario.get("W", 256),
            input_ch=scenario["input_ch"],
            input_ch_views=scenario["input_ch_views"],
            output_ch=scenario["output_ch"],
            use_viewdirs=scenario["use_viewdirs"],
        )

        # PyTorch Model
        torch_model = NeRF(
            D=scenario.get("D", 8),
            W=scenario.get("W", 256),
            input_ch=scenario["input_ch"],
            input_ch_views=scenario["input_ch_views"],
            output_ch=scenario["output_ch"],
            use_viewdirs=scenario["use_viewdirs"],
        )

        # Create identical input
        total_input_ch = scenario["input_ch"] + scenario["input_ch_views"]
        input_np = np.random.randn(1, total_input_ch).astype(np.float32)

        # Convert input to TensorFlow and PyTorch tensors
        tf_input = tf.constant(input_np)
        torch_input = torch.tensor(input_np)

        # Set weights identically

        print(tf_model)
        print(torch_model)

        # torch.Tensor(tf_model.layers[2].get_weights().numpy()).float()
        for i in range(5):
            torch_model.layers[i][0].weight = nn.Parameter(torch.tensor(tf_model.layers[i + 2].get_weights()[0].T))  # type: ignore
            torch_model.layers[i][0].bias = nn.Parameter(torch.tensor(tf_model.layers[i + 2].get_weights()[1]))  # type: ignore

        for i in range(3):
            torch_model.layers[i + 5][0].weight = nn.Parameter(torch.tensor(tf_model.layers[i + 8].get_weights()[0].T))  # type: ignore
            torch_model.layers[i + 5][0].bias = nn.Parameter(torch.tensor(tf_model.layers[i + 8].get_weights()[1]))  # type: ignore

        if scenario["use_viewdirs"]:
            torch_model.bottleneck_layer.weight = nn.Parameter(
                torch.tensor(tf_model.layers[11].get_weights()[0].T)
            )
            torch_model.bottleneck_layer.bias = nn.Parameter(
                torch.tensor(tf_model.layers[11].get_weights()[1])
            )

            for i in range(4):
                torch_model.output_layers[i][0].weight = nn.Parameter(torch.tensor(tf_model.layers[13 + i].get_weights()[0].T))  # type: ignore
                torch_model.output_layers[i][0].bias = nn.Parameter(torch.tensor(tf_model.layers[13 + i].get_weights()[1]))  # type: ignore
            torch_model.output_layers[4].weight = nn.Parameter(
                torch.tensor(tf_model.layers[17].get_weights()[0].T)
            )
            torch_model.output_layers[4].bias = nn.Parameter(
                torch.tensor(tf_model.layers[17].get_weights()[1])
            )

            torch_model.alpha_layer.weight = nn.Parameter(
                torch.tensor(tf_model.layers[18].get_weights()[0].T)
            )
            torch_model.alpha_layer.bias = nn.Parameter(
                torch.tensor(tf_model.layers[18].get_weights()[1])
            )

        else:
            torch_model.output_layers[0].weight = nn.Parameter(
                torch.tensor(tf_model.layers[11].get_weights()[0].T)
            )
            torch_model.output_layers[0].bias = nn.Parameter(
                torch.tensor(tf_model.layers[11].get_weights()[1])
            )

        # Compute outputs

        # Run tf model to get output (currently symbolic)
        tf_output = tf_model(tf_input).numpy()
        torch_output = torch_model(torch_input).detach().numpy()

        # Compare outputs
        np.testing.assert_allclose(
            tf_output,
            torch_output,
            rtol=1e-5,
            atol=1e-8,
            err_msg=f"Outputs differ for scenario: {scenario['name']}",
        )
        print(tf_output, torch_output)
        print("✓ Outputs match!")

    print("\nAll scenarios passed successfully!")


if __name__ == "__main__":
    compare_models()
