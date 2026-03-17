from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from yaml.error import YAMLError

from elasticai.explorer.hw_nas.search_space.quantization import (
    FullPrecisionScheme,
    QuantizationScheme,
)
from elasticai.explorer.hw_nas.search_space.quantization_builder import QUANT_REGISTRY
from elasticai.explorer.hw_nas.search_space.registry import COMPOSITE_REGISTRY
from settings import ROOT_DIR


class RepeatType(Enum):
    REPEAT_OP = "repeat_op"
    REPEAT_PARAMS = "repeat_params"
    VARY_ALL = "vary_all"
    REPEAT_BLOCK = "repeat_block"
    MIRROR_BLOCK = "mirror_block"
    NONE = "none"


"""These parameters have to be checked for in the last layer of the model while sampling the search space 
because they are ignored during model creation where the output shape is taken instead. Not doing this would lead to 
multiple different search space samples leading to the same architecture, increasing the search costs."""
FORCED_PARAMS = {
    "linear": ["width"],
    "lstm": ["hidden_size"],
    "conv1d": ["out_channels"],
}


def is_composite_op(op: str) -> bool:
    return op in COMPOSITE_REGISTRY


def parse_search_param(
    trial,
    name: str,
    params: dict,
    key: str,
    default_value: any = None,
) -> any:
    if key not in params:
        if default_value is not None:
            return default_value
        raise YAMLError(f"Missing required parameter '{name}'")

    param = params[key]
    if isinstance(param, list):
        return trial.suggest_categorical(name, param)

    if isinstance(param, dict) and "start" in param and "end" in param:
        if isinstance(param["start"], int):
            return trial.suggest_int(
                name, param["start"], param["end"], step=param.get("step", 1)
            )
        elif isinstance(param["start"], float):
            step = param.get("step", None)
            if step is not None:
                return trial.suggest_float(
                    name, param["start"], param["end"], step=step
                )
            else:
                return trial.suggest_float(name, param["start"], param["end"], log=True)

    return param


class Sampler:
    def __init__(self, trial, scope=""):
        self.trial = trial
        self.scope = scope
        self.default_op_params = {}
        self.op_cache = {}
        self.param_cache = {}
        self.block_cache = {}
        self.composite_cache = {}

    def scoped(self, name: str) -> str:
        return f"{self.scope}/{name}" if self.scope else name

    def child(self, name: str) -> "Sampler":
        child = Sampler(self.trial, self.scoped(name))
        child.default_op_params = self.default_op_params
        child.composite_cache = self.composite_cache
        child.block_cache = self.block_cache
        return child

    def get_factory(self, repeat_type, block_identifier, block_params):
        if repeat_type == RepeatType.REPEAT_OP:
            return RepeatOpVaryParamsFactory(self, block_identifier, block_params)
        elif repeat_type == RepeatType.REPEAT_PARAMS:
            return RepeatOpRepeatParamsFactory(self, block_identifier, block_params)
        elif repeat_type == RepeatType.VARY_ALL:
            return VaryAllFactory(self, block_identifier, block_params)
        else:
            raise TypeError(f"Unsupported repeat type {repeat_type}")

    def construct_sample(self, search_space: dict):

        quant_scheme = get_quantization_scheme(search_space, self.trial)

        model = OrderedDict()

        if "default_op_params" in search_space:
            self.default_op_params = search_space["default_op_params"]

        if "composites" in search_space:
            COMPOSITE_REGISTRY.update(search_space["composites"])

        sequence = search_space["sequence"]
        total_blocks = len(sequence)
        for block_idx, block in enumerate(sequence):
            block_id = block["block"]
            repeat_cfg = block.get("type_repeat", {})
            repeat_type = RepeatType(repeat_cfg.get("type", "repeat_op"))
            if repeat_type == RepeatType.REPEAT_BLOCK:
                ref = repeat_cfg.get("ref_block")
                model[block_id] = model[ref]
                continue
            block_sampler = self.child(f"block_{block_id}")
            factory = block_sampler.get_factory(repeat_type, block_id, block)
            is_last_block = block_idx == total_blocks - 1
            block_instance = Block(
                block_sampler,
                block_id,
                repeat_cfg,
                factory,
            )

            model[block_id] = block_instance.sample(
                is_last_block=is_last_block, last_model_layer=is_last_block
            )

        return model, quant_scheme


class LayerContext:
    def __init__(
        self, block_id, layer_idx, depth, last_block=False, last_model_layer=False
    ):
        self.block_id = block_id
        self.layer_idx = layer_idx
        self.depth = depth
        self.last_block = last_block
        self.last_model_layer = last_model_layer

    @property
    def is_last_layer(self):
        return self.layer_idx == self.depth - 1

    @property
    def is_last_block(self):
        return self.last_block

    @property
    def is_last_model_layer(self):
        return self.last_model_layer


class Block:
    def __init__(self, sampler: Sampler, block_id: str, depth_cfg: dict, factory):
        self.sampler = sampler
        self.block_id = block_id
        self.factory = factory
        self.depth = parse_search_param(
            sampler.trial,
            sampler.scoped(f"depth"),
            depth_cfg,
            "depth",
            default_value=1,
        )

        self.repeat_type = factory.param_sampler.__class__
        self.repeat_op = isinstance(factory.op_sampler, RepeatOp)
        self.repeat_params = isinstance(factory.param_sampler, RepeatParams)
        self.repeat_block = isinstance(factory.op_sampler, RepeatOp) and isinstance(
            factory.param_sampler, RepeatParams
        )
        self.sampled_layers = OrderedDict()

    def sample(self, is_last_block: bool = False, last_model_layer: bool = False):
        for layer_idx in range(self.depth):
            last_layer_flag = last_model_layer and (layer_idx == self.depth - 1)

            ctx = LayerContext(
                self.block_id,
                layer_idx,
                self.depth,
                last_block=is_last_block,
                last_model_layer=last_layer_flag,
            )

            layer_scope = f"l{layer_idx}"
            layer_sampler = self.sampler.child(layer_scope)

            op = self.factory.sampleOperation(layer_sampler)
            if op == "identity":
                continue
            if is_composite_op(op):
                composite_space = COMPOSITE_REGISTRY[op]
                cache_key = (self.sampler.scope, op)

                if self.repeat_block:
                    if cache_key not in self.sampler.block_cache:
                        composite_sampler = layer_sampler.child(op)
                        self.sampler.block_cache[cache_key] = (
                            composite_sampler.construct_sample(
                                composite_space,
                            )
                        )
                    composite_sample = self.sampler.block_cache[cache_key]

                elif self.repeat_params:
                    if cache_key not in self.sampler.composite_cache:
                        composite_sampler = layer_sampler.child(op)
                        self.sampler.composite_cache[cache_key] = (
                            composite_sampler.construct_sample(composite_space)
                        )
                    composite_sample = self.sampler.composite_cache[cache_key]
                else:
                    composite_sampler = layer_sampler.child(op)
                    composite_sample = composite_sampler.construct_sample(
                        composite_space
                    )

                for cb, layers in composite_sample.items():
                    for k, v in layers.items():
                        self.sampled_layers[f"{layer_scope}/{cb}/{k}"] = v
            else:

                params = self.factory.sampleParams(layer_sampler, op, ctx)
                self.sampled_layers[layer_scope] = {
                    "operation": op,
                    "params": params,
                }
        return self.sampled_layers


class OpCandidates(ABC):
    @abstractmethod
    def sample(self, sampler: Sampler) -> str:
        pass


class RepeatOp(OpCandidates):
    def __init__(self, block_identifier, op_cfg):
        self.block_identifier = block_identifier
        self.op_cfg = op_cfg
        self.cached = None

    def sample(self, sampler: Sampler):
        if self.cached is None:
            self.cached = parse_search_param(
                sampler.trial,
                sampler.scoped("operation"),
                self.op_cfg,
                "op_candidates",
            )
        return self.cached


class VaryOp(OpCandidates):
    def __init__(self, block_identifier, op_cfg):
        self.block_identifier = block_identifier
        self.op_cfg = op_cfg

    def sample(self, sampler: Sampler):
        name = sampler.scoped(f"operation")
        op = parse_search_param(
            sampler.trial,
            name,
            self.op_cfg,
            "op_candidates",
        )
        return op


class ParamCandidates(ABC):
    def __init__(self, block_identifier, search_params, default_params):
        self.block_identifier = block_identifier
        self.search_params = search_params
        self.default_params = default_params

    def find_params(self, op):
        if isinstance(self.search_params.get(op), dict):
            return self.search_params[op]

        if isinstance(self.default_params.get(op), dict):
            return self.default_params[op]

        return {}

    @abstractmethod
    def sample(self, sampler: Sampler, op: str, layer_ctx: LayerContext) -> dict:
        pass


class RepeatParams(ParamCandidates):
    def __init__(self, block_identifier, search_params, default_params):
        super().__init__(block_identifier, search_params, default_params)
        self.cache = {}

    def sample(self, sampler: Sampler, op: str, layer_ctx: LayerContext) -> dict:
        if op not in self.cache:
            params = {}
            for p, cfg in self.find_params(op).items():
                if (
                    layer_ctx.is_last_layer
                    and layer_ctx.is_last_block
                    and op in FORCED_PARAMS
                ):
                    if p in FORCED_PARAMS[op]:
                        continue
                params[p] = parse_search_param(
                    sampler.trial,
                    sampler.scoped(f"{op}/{p}"),
                    self.find_params(op),
                    p,
                )
            self.cache[op] = params
        return self.cache[op]


class VaryParams(ParamCandidates):
    def sample(self, sampler: Sampler, op: str, layer_ctx: LayerContext) -> dict:
        params = {}
        for p in self.find_params(op):
            if (
                layer_ctx.is_last_layer
                and layer_ctx.is_last_block
                and op in FORCED_PARAMS
            ):
                if p in FORCED_PARAMS[op]:
                    continue
            params[p] = parse_search_param(
                sampler.trial,
                sampler.scoped(f"{op}/{p}"),
                self.find_params(op),
                p,
            )
        return params


class BlockFactory(ABC):
    def __init__(self, op_sampler: OpCandidates, param_sampler: ParamCandidates):
        self.op_sampler = op_sampler
        self.param_sampler = param_sampler

    def sampleOperation(self, sampler: Sampler) -> str:
        return self.op_sampler.sample(sampler)

    def sampleParams(self, sampler: Sampler, op: str, block_ctx) -> dict:
        return self.param_sampler.sample(sampler, op, block_ctx)


class RepeatOpVaryParamsFactory(BlockFactory):
    def __init__(self, sampler, block_id, block_cfg):
        super().__init__(
            RepeatOp(block_id, block_cfg),
            VaryParams(block_id, block_cfg, sampler.default_op_params),
        )


class RepeatOpRepeatParamsFactory(BlockFactory):
    def __init__(self, sampler, block_id, block_cfg):
        super().__init__(
            RepeatOp(block_id, block_cfg),
            RepeatParams(block_id, block_cfg, sampler.default_op_params),
        )


class VaryAllFactory(BlockFactory):
    def __init__(self, sampler, block_id, block_cfg):
        super().__init__(
            VaryOp(block_id, block_cfg),
            VaryParams(block_id, block_cfg, sampler.default_op_params),
        )


def get_quantization_scheme(search_space: dict, trial) -> QuantizationScheme:
    quant_scheme = QuantizationScheme
    if "quantization" in search_space:
        quant_cfg = search_space["quantization"]
        quant_name = parse_search_param(
            trial,
            "quantization",
            quant_cfg,
            "quant_candidates",
        )
        quant_params = quant_cfg.get(quant_name, {})
        quant_builder_cls = QUANT_REGISTRY[quant_name]
        quant_builder = quant_builder_cls(trial, quant_params)
        quant_scheme = quant_builder.build()
    else:
        quant_scheme = FullPrecisionScheme()
    return quant_scheme
