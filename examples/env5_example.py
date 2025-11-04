import json
import logging.config
from pathlib import Path
from typing import Any, Callable

import torch


from elasticai.explorer.config import DeploymentConfig, HWNASConfig
from elasticai.explorer.generator.generator import Generator
from elasticai.explorer.generator.model_builder.model_builder import CreatorModelBuilder
from elasticai.explorer.hw_nas.search_space.quantization import (
    FixedPointInt8Scheme,
)
from elasticai.explorer.training.data import (
    BaseDataset,
    DatasetSpecification,
)
from elasticai.explorer.utils.data_to_csv import build_search_space_measurements_file
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import (
    KnowledgeRepository,
)
from elasticai.explorer.generator.deployment.compiler import ENv5Compiler
from elasticai.explorer.generator.deployment.device_communication import ENv5Host
from elasticai.explorer.generator.deployment.hw_manager import ENv5HWManager, Metric
from elasticai.explorer.generator.model_compiler.model_compiler import (
    CreatorModelCompiler,
)

from elasticai.explorer.training.trainer import MLPTrainer

logging.config.fileConfig("logging.conf", disable_existing_loggers=False)

logger = logging.getLogger("explorer.main")
from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)

INPUT_DIM = 6
SEQ_LENGTH = 1
BATCH_SIZE = 64
OUTPUT_SIZE = 4


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


def setup_knowledge_repository_env5() -> KnowledgeRepository:
    knowledge_repository = KnowledgeRepository()
    knowledge_repository.register_hw_platform(
        Generator(
            "env5",
            "Env5 with RP2040 and xc7s50ftgb196-2 FPGA",
            CreatorModelCompiler,
            ENv5HWManager,
            ENv5Host,
            ENv5Compiler,
        )
    )
    list()
    return knowledge_repository


def find_generate_measure_for_pico(
    explorer: Explorer,
    deploy_cfg: DeploymentConfig,
    hwnas_cfg: HWNASConfig,
    search_space: Path,
    retrain_epochs: int = 4,
):
    explorer.choose_target_hw(deploy_cfg)
    explorer.generate_search_space(search_space)

    quantization_scheme = FixedPointInt8Scheme()
    fxp_params = FxpParams(
        total_bits=quantization_scheme.total_bits,
        frac_bits=quantization_scheme.frac_bits,
        signed=quantization_scheme.signed,
    )
    fxp_conf = FxpArithmetic(fxp_params)
    dataset_spec = DatasetSpecification(
        dataset_type=DatasetExample,
        dataset_location=Path(""),
        deployable_dataset_path=None,
        transform=lambda x: fxp_conf.as_rational(fxp_conf.cut_as_integer(x)),
        target_transform=None,
    )

    top_models = explorer.search(
        hwnas_cfg, dataset_spec, MLPTrainer, CreatorModelBuilder()
    )

    latency_measurements = []
    accuracy_measurements_on_device = []
    accuracy_after_retrain = []
    retrain_device = "cpu"
    for i, model in enumerate(top_models):
        mlp_trainer = MLPTrainer(
            device=retrain_device,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            dataset_spec=dataset_spec,
        )
        mlp_trainer.train(model, epochs=retrain_epochs)
        accuracy_after_retrain_value, _ = mlp_trainer.test(model)
        model_name = "ts_model_" + str(i) + ".tflite"
        explorer.generate_for_hw_platform(model, model_name, dataset_spec)

        metric_to_source = {
            Metric.ACCURACY: Path("code/pico_crosscompiler/measure_accuracy"),
            Metric.LATENCY: Path("code/pico_crosscompiler/measure_latency"),
        }
        explorer.hw_setup_on_target(metric_to_source, dataset_spec)

        try:
            latency = explorer.run_measurement(Metric.LATENCY, model, model_name)
        except Exception as e:
            latency = json.loads('{ "Latency": { "value": -2, "unit": "microseconds"}}')
            print(f"An error occurred when measuring Latency on Pico: {e}")
        try:
            accuracy_on_device = explorer.run_measurement(
                Metric.ACCURACY, model, model_name
            )
        except Exception as e:
            accuracy_on_device = json.loads(
                '{"Accuracy": { "value":  -2, "unit": "percent"}}'
            )
            print(f"An error occurred when measuring accuracy on Pico: {e}")

        accuracy_after_retrain_dict = json.loads(
            '{"Accuracy after retrain": { "value":'
            + str(accuracy_after_retrain_value)
            + ' , "unit": "percent"}}'
        )
        latency_measurements.append(latency)
        accuracy_measurements_on_device.append(accuracy_on_device)
        accuracy_after_retrain.append(accuracy_after_retrain_dict)

    latencies = [latency["Latency"]["value"] for latency in latency_measurements]
    accuracies_on_device = [
        accuracy["Accuracy"]["value"] for accuracy in accuracy_measurements_on_device
    ]
    accuracy_after_retrain = [
        accuracy["Accuracy after retrain"]["value"]
        for accuracy in accuracy_after_retrain
    ]

    df = build_search_space_measurements_file(
        latencies,
        accuracy_after_retrain,
        accuracies_on_device,
        explorer.metric_dir / "metrics.json",
        explorer.model_dir / "models.json",
        explorer.experiment_dir / "experiment_data.csv",
    )
    logger.info("Models:\n %s", df)


if __name__ == "__main__":
    hwnas_cfg = HWNASConfig(config_path=Path("configs/env5/hwnas_config.yaml"))
    deploy_cfg = DeploymentConfig(
        config_path=Path("configs/env5/deployment_config.yaml")
    )

    knowledge_repo = setup_knowledge_repository_env5()
    explorer = Explorer(knowledge_repo)
    search_space = Path("configs/env5/search_space.yaml")
    retrain_epochs = 3
    find_generate_measure_for_pico(
        explorer, deploy_cfg, hwnas_cfg, search_space, retrain_epochs
    )
