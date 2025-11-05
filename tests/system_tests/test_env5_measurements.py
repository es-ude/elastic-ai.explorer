from pathlib import Path
from typing import Any, Callable

import torch
from elasticai.explorer.config import DeploymentConfig, HWNASConfig
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.generator.deployment.compiler import ENv5Compiler
from elasticai.explorer.generator.deployment.device_communication import ENv5Host
from elasticai.explorer.generator.deployment.hw_manager import ENv5HWManager, Metric
from elasticai.explorer.generator.generator import Generator
from elasticai.explorer.generator.model_compiler.model_compiler import (
    CreatorModelCompiler,
)
from elasticai.explorer.hw_nas.search_space.quantization import FixedPointInt8Scheme
from elasticai.explorer.knowledge_repository import KnowledgeRepository
from elasticai.explorer.training.data import BaseDataset, DatasetSpecification

from settings import ROOT_DIR
from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)
from elasticai.runtime.env5.usb import get_env5_port
BATCH_SIZE = 64
INPUT_DIM = 6

class DatasetExample(BaseDataset):

    def __init__(
        self,
        root: str | Path,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(root, transform, target_transform, *args, **kwargs)

        self.test_data = torch.randn(BATCH_SIZE * 10, INPUT_DIM) * 5
        self.targets = torch.empty(BATCH_SIZE * 10, dtype=torch.long).random_(4)

    def __getitem__(self, idx) -> Any:
        data, target = self.test_data[idx], self.targets[idx]
        if self.transform is not None:
            data = self.transform(self.test_data[idx])

        if self.target_transform is not None:
            target = self.target_transform(self.targets[idx])
        return data, target

    def __len__(self) -> int:
        return len(self.test_data)


def create_example_dataset_spec(quantization_scheme):

    fxp_params = FxpParams(
        total_bits=quantization_scheme.total_bits,
        frac_bits=quantization_scheme.frac_bits,
        signed=quantization_scheme.signed,
    )
    fxp_conf = FxpArithmetic(fxp_params)
    return DatasetSpecification(
        dataset_type=DatasetExample,
        dataset_location=Path(""),
        deployable_dataset_path=None,
        transform=lambda x: fxp_conf.as_rational(fxp_conf.cut_as_integer(x)),
        target_transform=None,
    )


class TestMeasurement:
    def setup_class(self):
        self.deploy_cfg = DeploymentConfig(
            config_path=ROOT_DIR
            / Path("tests/system_tests/test_configs/deployment_config_env5.yaml")
        )
        

    def test_measurement(self):
        env5_host = ENv5Host(self.deploy_cfg)
        env5_compiler = ENv5Compiler(self.deploy_cfg)
        env5_hw_manager = ENv5HWManager(env5_host, env5_compiler)
        quantization_scheme = FixedPointInt8Scheme()
        data_spec = create_example_dataset_spec(quantization_scheme)
        env5_hw_manager.prepare_dataset(data_spec, quantization_scheme)
        env5_host.put_file(
            local_path=Path(
                "tests/system_tests/test_configs/deployment_config_env5.yaml"
            ),
            remote_path=None,
        )

        accuracy = env5_hw_manager._run_accuracy_test()
