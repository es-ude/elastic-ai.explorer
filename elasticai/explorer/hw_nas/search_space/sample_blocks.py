from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from typing import Any, Sequence
import optuna
import yaml
from optuna.samplers import RandomSampler
from torch import nn as nn
from yaml.error import YAMLError

from elasticai.explorer.hw_nas.search_space.layer_builder import LAYER_REGISTRY
from elasticai.explorer.hw_nas.search_space.registry import ADAPTER_REGISTRY

from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

settings_path = Path(__file__).resolve().parents[4] / "settings.py"
spec = spec_from_file_location("settings", settings_path)
settings = module_from_spec(spec)
spec.loader.exec_module(settings)

ROOT_DIR = settings.ROOT_DIR


class RepeatType(Enum):
    REPEAT_OP = "repeat_op"
    REPEAT_PARAMS = "repeat_params"
    VARY_ALL = "vary_all"
    REPEAT_BLOCK = "repeat_block"
    MIRROR_BLOCK = "mirror_block"
    NONE = "none"

    def is_depth_needed(self) -> bool:
        if (
            self == RepeatType.REPEAT_OP
            or RepeatType.REPEAT_PARAMS
            or RepeatType.VARY_ALL
        ):
            return True
        return False

    def is_ref_block_needed(self) -> bool:
        if self == RepeatType.REPEAT_BLOCK or RepeatType.MIRROR_BLOCK:
            return True
        return False


class ShapeValueError(ValueError):
    def __init__(self, msg):
        super(ShapeValueError, self).__init__(msg)


def parse_search_param(
    trial, name: str, params: dict, key: str, default_value: Any = None
) -> Any:
    if key in params:
        param = params[key]
    else:
        if default_value is not None:
            return default_value
        else:
            raise YAMLError(
                "Parameter '{}' is not optional and missing in configuration.".format(
                    name
                )
            )
    if isinstance(param, list):
        return trial.suggest_categorical(name, param)
    elif isinstance(param, dict) and "start" in param and "end" in param:
        if isinstance(param["start"], int):
            return trial.suggest_int(name, param["start"], param["end"])
    else:
        return param
    raise ValueError(f"Invalid search space parameter '{name}'")


class Sampler:
    def __init__(self, trial):
        self.block_factories = []
        self.trial = trial
        self.default_op_params = {}

    def get_factory(self, repeat_type, block_identifier, block_params):
        if repeat_type == RepeatType.REPEAT_OP:
            return RepeatOpVaryParamsFactory(
                self.trial, block_identifier, block_params, self.default_op_params
            )
        elif repeat_type == RepeatType.REPEAT_PARAMS:
            return RepeatOpRepeatParamsFactory(
                self.trial, block_identifier, block_params, self.default_op_params
            )
        elif repeat_type == RepeatType.VARY_ALL:
            return VaryAllFactory(
                self.trial, block_identifier, block_params, self.default_op_params
            )
        else:
            raise TypeError("Unknown repeat type")

    def construct_sample(self, search_space: dict):
        model = OrderedDict()
        if "default_op_params" in search_space:
            self.default_op_params = search_space["default_op_params"]

        for block in search_space["sequence"]:
            block_identifier = block["block"]
            print(block_identifier)
            repeat = block.get("type_repeat", {})
            repeat_type = RepeatType(repeat.get("type", "repeat_op"))

            if repeat_type == RepeatType.REPEAT_BLOCK:
                print("model with literal", model["1"])
                ref_block = repeat["ref_block"]
                model[block_identifier] = model[f"{ref_block}"]
                continue

            if repeat_type == RepeatType.MIRROR_BLOCK:
                continue

            layer_factory = self.get_factory(
                repeat_type, block_identifier, block_params=block
            )
            instanced_block = Block(self.trial, block_identifier, repeat, layer_factory)
            model[block_identifier] = instanced_block.sample()

        return model


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
    print(value)
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
        print(f"block{i} here{block}")
        for layer_index, layer_params in block.items():
            print(f"layer_index {layer_index} here{layer_params}")
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


class Block:
    def __init__(self, trial, block_identifier: str, depth: dict, block_factory):
        print(depth)
        self.depth = parse_search_param(
            trial, f"depth_b{block_identifier}", depth, "depth", default_value=1
        )
        self.all_searched_params = OrderedDict()
        self.block_identifier = block_identifier
        self.layer_sampler = block_factory

    def sample(self):
        for layer in range(self.depth):
            operation = self.layer_sampler.sampleOperation()
            parameters = self.layer_sampler.sampleParams(operation)
            self.all_searched_params[f"{self.block_identifier}_{layer}"] = {
                "operation": operation,
                "params": parameters,
            }
        return self.all_searched_params


class Operation:
    name: str


class OpCandidates(ABC):
    def sample(self) -> str:
        pass


class RepeatOp(OpCandidates):
    def __init__(self, trial, block_identifier, op_candidates: dict):
        print(op_candidates)
        self.op = parse_search_param(
            trial,
            f"operation_b{block_identifier}",
            op_candidates,
            "op_candidates",
            default_value=None,
        )

    def sample(self):
        return self.op


class VaryOp(OpCandidates):
    def __init__(self, trial, block_identifier, op_candidates: dict):
        self.trial = trial
        self.block_identifier = block_identifier
        self.op_candidates = op_candidates
        self.layer_num = 0

    def sample(self):
        op = parse_search_param(
            self.trial,
            f"operation_b{self.block_identifier}_l{self.layer_num}",
            self.op_candidates,
            "op_candidates",
            default_value=None,
        )
        self.layer_num += 1
        return op


class ParamCandidates(ABC):
    def __init__(
        self, trial, block_identifier, search_parameters: dict, default_op_params: dict
    ):
        self.trial = trial
        self.block_identifier = block_identifier
        self.search_parameters = search_parameters
        self.default_op_params = default_op_params

    def find_params_for_op(self, op):
        if op in self.search_parameters:
            return self.search_parameters[op]
        elif op in self.default_op_params:
            return self.default_op_params[op]
        else:
            return {}

    @abstractmethod
    def sample(self, op: str) -> dict:
        pass


class RepeatParams(ParamCandidates):
    def __init__(
        self, trial, block_identifier, search_parameters: dict, default_op_params: dict
    ):
        super().__init__(trial, block_identifier, search_parameters, default_op_params)
        self.op_search_parameters = {}

    def sample(self, op: str):

        if op not in self.op_search_parameters:
            op_dict = {}
            op_parameters = self.find_params_for_op(op)
            for param in op_parameters:
                sampled_value = parse_search_param(
                    self.trial,
                    f"{param}_b{self.block_identifier}",
                    op_parameters,
                    param,
                    default_value=None,
                )
                op_dict[param] = sampled_value
            self.op_search_parameters[op] = op_dict

        return self.op_search_parameters[op]


class VaryParams(ParamCandidates):
    def __init__(
        self, trial, block_identifier, search_parameters: dict, default_op_params: dict
    ):
        super().__init__(trial, block_identifier, search_parameters, default_op_params)
        self.layer_num = 0

    def sample(self, op: str):
        op_parameters = self.find_params_for_op(op)
        op_dict = {}
        for param in op_parameters:
            name = f"{param}_b{self.block_identifier}_l{self.layer_num}"
            sampled_value = parse_search_param(
                self.trial, name, op_parameters, param, default_value=None
            )
            op_dict[param] = sampled_value
        self.layer_num += 1
        return op_dict


class BlockFactory(ABC):
    def __init__(self, op_sampler: OpCandidates, param_sampler: ParamCandidates):
        self.op_sampler = op_sampler
        self.param_sampler = param_sampler

    def sampleOperation(self) -> str:
        return self.op_sampler.sample()

    def sampleParams(self, op: str) -> dict:
        return self.param_sampler.sample(op)


class RepeatOpVaryParamsFactory(BlockFactory):
    def __init__(
        self, trial, block_identifier, block_parameters: dict, default_op_params: dict
    ):
        super().__init__(
            RepeatOp(trial, block_identifier, block_parameters),
            VaryParams(trial, block_identifier, block_parameters, default_op_params),
        )


class RepeatOpRepeatParamsFactory(BlockFactory):
    def __init__(
        self, trial, block_identifier, block_parameters: dict, default_op_params: dict
    ):
        super().__init__(
            RepeatOp(trial, block_identifier, block_parameters),
            RepeatParams(trial, block_identifier, block_parameters, default_op_params),
        )


class VaryAllFactory(BlockFactory):
    def __init__(
        self, trial, block_identifier, block_parameters: dict, default_op_params: dict
    ):
        super().__init__(
            VaryOp(trial, block_identifier, block_parameters),
            VaryParams(trial, block_identifier, block_parameters, default_op_params),
        )


def objective(trial, search_space):
    search_space_sampler = Sampler(trial)
    sample = search_space_sampler.construct_sample(search_space)
    print(sample)
    model = construct_model(sample, search_space["input"], search_space["output"])
    print(model)
    # first=trial.suggest_categorical("test", [1, 3, 4, 5, 6,7])
    # second=trial.suggest_categorical("test", [1, 3, 4, 5, 6,7])
    # print(first, second)
    return 1


if __name__ == "__main__":
    search_space = yaml.safe_load(open(ROOT_DIR / "search_space.yaml"))

    sampler = RandomSampler(seed=1)
    study = optuna.create_study(
        direction="maximize",
        sampler=RandomSampler(),
        study_name="study_name5",
        load_if_exists=True,
    )
    study.optimize(lambda trial: objective(trial, search_space), n_trials=5)
    print(study.best_trial)
