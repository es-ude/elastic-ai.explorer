from typing import OrderedDict, Any

import pytest
from optuna.trial import FixedTrial

from elasticai.explorer.hw_nas.search_space.sample_blocks import (
    Sampler,
    parse_search_param,
    VaryParams,
)


def get_sample(params, sp):
    trial = FixedTrial(params)
    search_space_sampler = Sampler(trial)
    sample = search_space_sampler.construct_sample(sp)
    return sample


@pytest.mark.parametrize(
    " name,  params,  key, expected",
    [
        ("b1/l0/width", {"width": [1, 2, 3]}, "width", 3),
        ("b1/l0/width", {"width": {"start": 1, "end": 4}}, "width", 2),
        ("", {"width": 2}, "width", 2),
        ("", {"op": "Hello"}, "op", "Hello"),
    ],
)
def test_parse_params(name, params, key, expected):
    trial = FixedTrial({name: expected})
    actual = parse_search_param(trial, name, params, key)
    assert actual == expected


def setup_search_space(block_id, op_candidates, repeat_type):
    b1 = {
        "block": block_id,
        "op_candidates": op_candidates,
        "type_repeat": {"type": repeat_type, "depth": [1, 2, 5]},
        "linear": {"width": [32, 64], "activation": "relu"},
    }
    sp = {"sequence": [b1]}
    return sp


@pytest.mark.parametrize("depth", [1, 2, 5])
def test_repeat_op(depth):

    search_space_with_op_repeat = setup_search_space(
        block_id="1", op_candidates=["linear"], repeat_type="repeat_op"
    )

    params = {"block_1/depth": depth, "block_1/l0/operation": "linear"}
    for i in range(depth - 1):
        params[f"block_1/l{i}/linear/width"] = 32
    sample = get_sample(params, search_space_with_op_repeat)
    assert (len(sample) == 1) & (len(sample["1"]) == depth)
    for i in range(depth - 1):
        assert sample["1"][f"l{i}"]["params"] == {"width": 32, "activation": "relu"}
    assert sample["1"][f"l{depth-1}"]["params"] == {"activation": "relu"}
    assert sample["1"]["l0"]["operation"] == "linear"


def build_block_params(depth, block_id, operation, attributes, is_last_block):
    params = {}

    params[f"{block_id}/depth"] = depth
    for i in range(depth):
        params[f"block_{i}/l{i}/operation"] = operation
        for key, value in attributes.items():
            params[f"block_{block_id}/l{i}/{operation}/{key}"] = value
    return params


@pytest.mark.parametrize("depth", [1, 2, 5])
def test_vary_all_op(depth):
    search_space_with_vary_all = setup_search_space(
        block_id="1", op_candidates=["linear", "conv2d"], repeat_type="vary_all"
    )
    params: dict[str, Any] = {"block_1/depth": depth}
    operation = "linear"
    for i in range(depth - 1):
        params[f"block_1/l{i}/operation"] = operation
        params[f"block_1/l{i}/linear/width"] = 32
    params[f"block_1/l{depth-1}/operation"] = operation

    sample = get_sample(params, search_space_with_vary_all)
    assert (len(sample) == 1) and (len(sample["1"]) == depth)
    for i in range(depth - 1):
        assert sample["1"][f"l{i}"]["params"] == {"width": 32, "activation": "relu"}
    assert sample["1"][f"l{depth-1}"]["params"] == {"activation": "relu"}
    assert sample["1"]["l0"]["operation"] == "linear"


@pytest.mark.parametrize("depth", [1, 2, 5])
def test_repeat_block(depth):
    sp = setup_search_space(
        block_id="1", op_candidates=["linear", "conv2d"], repeat_type="repeat_op"
    )
    b2 = {"block": "2", "type_repeat": {"type": "repeat_block", "ref_block": "1"}}
    sp["sequence"].append(b2)

    params = {"block_1/depth": depth, "block_1/l0/operation": "linear"}
    for i in range(depth):
        params[f"block_1/l{i}/linear/width"] = 32
    sample = get_sample(params, sp)
    assert len(sample) == 2
    assert sample["1"] == sample["2"]


@pytest.mark.parametrize("depth", [1, 2, 5])
def test_repeat_params(depth):
    search_space_with_repeat_params = setup_search_space(
        block_id="1", op_candidates=["linear", "conv2d"], repeat_type="repeat_params"
    )
    params = {
        "block_1/depth": depth,
        "block_1/l0/operation": "linear",
        "block_1/l0/linear/width": 32,
    }
    sample = get_sample(params, search_space_with_repeat_params)
    print(sample)
    assert (len(sample) == 1) and (len(sample["1"]) == depth)
    assert sample["1"][f"l0"]["operation"] == "linear"
    if depth == 1:
        assert sample["1"][f"l0"]["params"] == {"activation": "relu"}
    else:
        for i in range(depth):
            assert sample["1"][f"l{i}"]["params"] == {"width": 32, "activation": "relu"}
    assert sample["1"]["l0"]["operation"] == "linear"


CONV_ACT_POOL_COMPOSITE = {
    "sequence": [
        {
            "block": "conv_act_pool_1",
            "op_candidates": "conv1d",
            "type_repeat": {"type": "repeat_op", "depth": 2},
        },
        {"block": "conv_act_pool_2", "op_candidates": "batch_norm"},
        {"block": "conv_act_pool_3", "op_candidates": "activation"},
        {
            "block": "conv_act_pool_4",
            "op_candidates": "maxpool",
            "maxpool": {"kernel_size": 2},
        },
    ]
}

DEFAULT_OP_PARAMS = {
    "conv1d": {"kernel_size": [2, 3], "stride": 1, "out_channels": 16, "padding": 0},
    "batch_norm": {},
    "activation": {"op": ["relu", "sigmoid"]},
}


# Usecase: composite was used in block. Block repeats the params -> sample composite op once and reuse
@pytest.mark.parametrize("depth", [3, 5])
def test_composite_repeat_params_caches_composite(depth):
    sp = {
        "sequence": [
            {
                "block": "1",
                "op_candidates": ["conv-act-pool", "conv1d"],
                "type_repeat": {"type": "repeat_params", "depth": [3, 5]},
            }
        ],
        "default_op_params": DEFAULT_OP_PARAMS,
        "composites": {"conv-act-pool": CONV_ACT_POOL_COMPOSITE},
    }
    params = {
        "block_1/l0/operation": "conv-act-pool",
        "block_1/depth": depth,
        "block_1/l0/conv-act-pool/block_conv_act_pool_1/l0/conv1d/kernel_size": 3,
        "block_1/l0/conv-act-pool/block_conv_act_pool_1/l1/conv1d/kernel_size": 2,
        "block_1/l0/conv-act-pool/block_conv_act_pool_3/l0/activation/op": "sigmoid",
    }

    result = get_sample(params, sp)["1"]

    # all layers must be identical — composite was sampled once and reused
    first_layer = {
        "conv_act_pool_1/l0": result["l0/conv_act_pool_1/l0"],
        "conv_act_pool_1/l1": result["l0/conv_act_pool_1/l1"],
        "conv_act_pool_2/l0": result["l0/conv_act_pool_2/l0"],
        "conv_act_pool_3/l0": result["l0/conv_act_pool_3/l0"],
        "conv_act_pool_4/l0": result["l0/conv_act_pool_4/l0"],
    }
    for i in range(1, depth):
        layer = {
            "conv_act_pool_1/l0": result[f"l{i}/conv_act_pool_1/l0"],
            "conv_act_pool_1/l1": result[f"l{i}/conv_act_pool_1/l1"],
            "conv_act_pool_2/l0": result[f"l{i}/conv_act_pool_2/l0"],
            "conv_act_pool_3/l0": result[f"l{i}/conv_act_pool_3/l0"],
            "conv_act_pool_4/l0": result[f"l{i}/conv_act_pool_4/l0"],
        }
        assert layer == first_layer, f"l{i} differs from l0 -> composite was not cached"


def test_find_params_falls_back_to_empty_dict():
    candidate = VaryParams(
        block_identifier="1",
        search_params={},  # op not defined in block
        default_params={},  # op not defined in defaults
    )
    assert candidate.find_params("unknown_op") == {}


def test_find_params_prefers_block_over_defaults():
    candidate = VaryParams(
        block_identifier="1",
        search_params={"linear": {"width": 64}},
        default_params={"linear": {"width": 32}},
    )
    assert candidate.find_params("linear") == {"width": 64}


def test_find_params_falls_back_to_default_params():
    candidate = VaryParams(
        block_identifier="1",
        search_params={},
        default_params={"linear": {"width": 32}},
    )
    assert candidate.find_params("linear") == {"width": 32}


# Usecase: Block 1 uses composite operation, uses vary_all repeat type:
# Should sample each conv_act_pool sequence independently from each other.
@pytest.mark.parametrize("depth", [3, 5])  # still think about this one
def test_composite_vary_all_samples_independently(depth):
    sp = {
        "sequence": [
            {
                "block": "1",
                "op_candidates": ["conv-act-pool", "conv1d"],
                "type_repeat": {"type": "vary_all", "depth": [3, 5]},
            }
        ],
        "default_op_params": DEFAULT_OP_PARAMS,
        "composites": {"conv-act-pool": CONV_ACT_POOL_COMPOSITE},
    }

    params = {
        "block_1/depth": depth,
        **{f"block_1/l{i}/operation": "conv-act-pool" for i in range(depth)},
        **{
            f"block_1/l{i}/conv-act-pool/block_conv_act_pool_1/l0/conv1d/kernel_size": 3
            for i in range(depth)
        },
        **{
            f"block_1/l{i}/conv-act-pool/block_conv_act_pool_1/l1/conv1d/kernel_size": 2
            for i in range(depth)
        },
        **{
            f"block_1/l{i}/conv-act-pool/block_conv_act_pool_3/l0/activation/op": (
                "sigmoid"
            )
            for i in range(depth)
        },
    }

    result = get_sample(params, sp)["1"]
    print(result)
    # every layer's sub-blocks must be present regardless of caching behaviour
    for i in range(depth):
        assert f"l{i}/conv_act_pool_1/l0" in result
        assert f"l{i}/conv_act_pool_1/l1" in result
        assert f"l{i}/conv_act_pool_2/l0" in result
        assert f"l{i}/conv_act_pool_3/l0" in result
        assert f"l{i}/conv_act_pool_4/l0" in result


def base_composite():
    """
    Composite with two blocks.
    Inner blocks have depth 2 to test internal repetition.
    """
    return {
        "sequence": [
            {
                "block": "inner_a",
                "depth": 2,
                "op_candidates": ["linear", "conv1d"],
                "linear": {"width": [16, 32]},
                "conv1d": {"out_channels": [8, 16]},
            },
            {
                "block": "inner_b",
                "depth": 1,
                "op_candidates": ["linear"],
                "linear": {"width": [64, 128]},
            },
        ]
    }
