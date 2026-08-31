from optuna.trial import FixedTrial

from elasticai.explorer import get_path_to_project
from elasticai.explorer.hw_nas.hw_nas import sample_and_create_model
from elasticai.explorer.hw_nas.search_space.utils import yaml_to_dict


def test_depthwise_separable_composite():
    path2yaml = get_path_to_project("tests/integration_tests/samples/depthwise_separable_sp.yaml")
    search_space_cfg = yaml_to_dict(path2yaml)
    trial = FixedTrial(
        {
            "block_1/l0/depthwise-separable/block_dw/l0/conv2d/kernel_size": 3,
            "block_1/l0/depthwise-separable/block_dw/l0/conv2d/stride": 2,
        }
    )
    model = sample_and_create_model(trial, search_space_cfg)
    layer_iter = model.named_children()
    dw = next(layer_iter)[1]
    pw = next(layer_iter)[1]
    assert dw.out_channels == 3 and dw.groups == 3
    assert pw.out_channels == 10 and pw.groups == 1 and pw.kernel_size == (1, 1)
