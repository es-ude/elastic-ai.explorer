import copy
import math
from typing import Iterable

import nni
from nni.mutable import MutableExpression, ensure_frozen, Mutable
from nni.nas.nn.pytorch import Repeat, MutableConv2d, MutableLinear, ModelSpace
from numpy import block
from torch import nn
import torch

class GaussianDropout(nn.Module):
    def __init__(self, p=0.5):
        super(GaussianDropout, self).__init__()
        if p < 0 or p >= 1:
            raise Exception("p value should accomplish 0 <= p < 1")
        self.p = p
        
    def forward(self, x):
        if self.training:
            stddev = (self.p / (1.0 - self.p))**0.5
            epsilon = torch.randn_like(x) * stddev
            return x * epsilon
        else:
            return x


class LambdaLayer(nn.Module):
    def __init__(self, func):
        super(LambdaLayer, self).__init__()
        self.func = func
    def forward(self, x):
        return self.func(x)


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)

        if not self.batch_first:
            y = y.transpose(0, 1).contiguous()  # transpose to (timesteps, samples, output_size)

        return y


@torch.no_grad
def step_with_update_clip_(optimizer, value=1.0):

    # extract parameters from optimizer
    params: list[torch.Tensor] = [p for g in optimizer.param_groups for p in g['params'] if p.grad is not None]
    params_before = [p.clone() for p in params] # store parameters before step
    optimizer.step() # update parameters

    # update is difference in parameters before and after step, apply clipping to it
    clipped_update = [torch.clip(p - p_before, -value, value) for p, p_before in zip(params, params_before)]

    # revert parameters and add clipped update instead
    for p, p_before, u in zip(params, params_before, clipped_update):
        p.set_(p_before + u) # type: ignore


class BlockFactory:
    def createBlock(
        self, operation, input_width, output_width, block, depth, max_depth
    ):
        match operation:
            case "conv2d":
                return Conv2dBlock(input_width, output_width)
            case "linear":
                return LinearBlock(input_width, output_width, block, depth, max_depth)
            case _:
                raise ValueError(f"Operation {operation} is not supported")


class LinearBlock(ModelSpace):
    def __init__(self, input_width, output_width, block: dict, depth, max_depth):
        self.depth = depth
        super().__init__()
        layers = []
        flattened_input = (
            math.prod(input_width) if isinstance(input_width, Iterable) else input_width
        )
        print(flattened_input)
        self.h_l_widths = [flattened_input] + [
            nni.choice(f"layer_width_{i}", block["linear"]["width"])
            for i in range(max_depth)
        ]
        for width in self.h_l_widths[1:]:
            if isinstance(width, Mutable):
                self.add_mutable(width)

        repeat = Repeat(
            lambda index: (
                LinearFlattened(self.h_l_widths[index], self.h_l_widths[index + 1])
                if index == 0
                else nn.Sequential(
                    MutableLinear(self.h_l_widths[index], self.h_l_widths[index + 1]),
                    #    LayerChoice(activation, label=f"activation_block_{block_id}"),
                )
            ),
            self.depth,
        )

        layers.append(repeat)
        last_layer_width = self.h_l_widths[ensure_frozen(self.depth)]
        if output_width is not None:
            layers.append(
                nn.Sequential(
                    MutableLinear(last_layer_width, output_width),
                    #     LayerChoice(activation, label=f"activation_block_{block_id}"),
                )
            )
        self.layers = nn.Sequential(*layers)
        self.output_shape = last_layer_width

    # return nn.Sequential(*layers), last_layer_width
    def forward(self, x):
        return self.layers(x)


class Conv2dBlock(nn.Module):

    def __init__(self, input_width, output_width):
        super().__init__()
        layers = []

        layers.append(
            Repeat(
                lambda index: nn.Sequential(
                    MutableConv2d(
                        self.out_channels[index],
                        self.out_channels[index + 1],
                        self.kernel_size,
                        self.stride,
                    )
                ),
                self.depth,
            )
        )
        x_shape = input_width
        self.output_shapes = []
        frozen_depth = ensure_frozen(self.depth)
        for i in range(frozen_depth):
            x_shape = self.calculate_conv_output_shape(
                x_shape,
                self.out_channels[i + 1],
                kernel_size=self.kernel_size,
                stride=self.stride,
            )
            self.output_shapes.append(x_shape)

        x_shape = self.output_shapes[frozen_depth - 1]
        input_width = x_shape[0] * x_shape[1] * x_shape[2]
        print(input_width)
        if output_width is not None:
            layers.append(LinearFlattened(input_width, 10))
        self.layers = nn.Sequential(*layers)
        self.output_shape = x_shape

    # return nn.Sequential(*layers), x_shape

    def forward(self, x):
        return self.layers(x)

    def calculate_conv_output_shape(
        self,
        shape,
        out_channels: int | MutableExpression[int],
        kernel_size: int | tuple[int, int] | MutableExpression[int],
        stride: int | tuple[int, int] | MutableExpression[int] = (1, 1),
        dilation: int | tuple[int, int] = (1, 1),
        padding: int | tuple[int, int] = (0, 0),
    ):
        kernel_size, stride, dilation, padding = self._convert_to_tuples(
            [kernel_size, stride, dilation, padding]
        )
        new_shape = copy.deepcopy(shape)
        new_shape[-3] = out_channels
        new_shape[-2] = (
            shape[-2] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
        ) // stride[0] + 1
        new_shape[-1] = (
            shape[-1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
        ) // stride[1] + 1

        return new_shape

    def _convert_to_tuples(self, values) -> list[tuple[int, int]]:
        return [x if isinstance(x, tuple) else (x, x) for x in values]


class LinearFlattened(nn.Sequential):

    def __init__(
        self,
        in_feat,
        out_feat,
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.lin = MutableLinear(in_feat, out_feat)

    def forward(self, x):
        x = self.flatten(x)
        return self.lin(x)
