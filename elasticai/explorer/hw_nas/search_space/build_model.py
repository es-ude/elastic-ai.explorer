from collections import OrderedDict
from typing import Sequence


from torch import nn

from elasticai.explorer.hw_nas.search_space.layer_builder import LAYER_REGISTRY
from elasticai.explorer.hw_nas.search_space.registry import ADAPTER_REGISTRY


class ShapeValueError(ValueError):
    pass


def insert_needed_adapters(input_shape, op, prev_operation, layers):
    adapter_cls = ADAPTER_REGISTRY.get((prev_operation, op))
    if adapter_cls is None:
        adapter_cls = ADAPTER_REGISTRY.get(("*", op))
    if adapter_cls is not None:
        adapter = adapter_cls()
        layers.append(adapter)
        next_input_shape = adapter_cls.infer_output_shape(input_shape)
        return layers, next_input_shape
    return layers, input_shape


def is_last_layer(block_index, layer_index, sample):
    block_id, layers = next(reversed(sample.items()))
    layer_id, layer = next(reversed(layers.items()))
    return block_index == block_id and layer_id == layer_index


def is_negative(value):
    if isinstance(value, Sequence):
        for val in value:
            if val <= 0:
                return True
    else:
        if value <= 0:
            return True
    return False


def construct_model(sample: OrderedDict, in_dim, out_dim):
    layers = []
    next_in_shape = in_dim
    prev_op = None
    for i, block in sample.items():
        for layer_index, layer_params in block.items():
            layers, next_in_shape = insert_needed_adapters(
                next_in_shape, layer_params["operation"], prev_op, layers
            )
            layer = LAYER_REGISTRY[layer_params["operation"]]()
            if is_last_layer(i, layer_index, sample):
                build_layer, next_in_shape = layer.build(
                    input_shape=next_in_shape,
                    search_parameters=layer_params["params"],
                    output_shape=out_dim,
                )
            else:
                build_layer, next_in_shape = layer.build(
                    input_shape=next_in_shape, search_parameters=layer_params["params"]
                )
            layers.append(build_layer)
            prev_op = layer_params["operation"]
            if is_negative(next_in_shape):
                raise ShapeValueError("Shape must not be negative")

    return nn.Sequential(*layers)
