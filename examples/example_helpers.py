from pathlib import Path
from typing import List
import torch
from torch import nn
from torchvision.transforms import transforms
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.hw_nas.estimators import (
    TrainMetricsEstimator,
    FLOPsEstimator,
)
from elasticai.explorer.hw_nas.optimization_criteria import OptimizationCriteria
from math import log10
from elasticai.explorer.knowledge_repository import HWPlatform, KnowledgeRepository
from elasticai.explorer.platforms.deployment.compiler import PicoCompiler, RPICompiler
from elasticai.explorer.platforms.deployment.device_communication import (
    PicoHost,
    RPiHost,
)
from elasticai.explorer.platforms.deployment.hw_manager import (
    Metric,
    PicoHWManager,
    RPiHWManager,
)
from elasticai.explorer.platforms.generator.generator import PicoGenerator, RPiGenerator
from elasticai.explorer.training.data import DatasetSpecification, MNISTWrapper
from elasticai.explorer.training.trainer import SupervisedTrainer, accuracy_fn
from torch import optim

from elasticai.explorer.utils.data_to_csv import build_search_space_measurements_file


def setup_knowledge_repository() -> KnowledgeRepository:
    knowledge_repository = KnowledgeRepository()
    knowledge_repository.register_hw_platform(
        HWPlatform(
            "pico",
            "Pico with RP2040 MCU and 2MB control memory",
            PicoGenerator,
            PicoHWManager,
            PicoHost,
            PicoCompiler,
        )
    )
    knowledge_repository.register_hw_platform(
        HWPlatform(
            "rpi5",
            "Raspberry PI 5 with A76 processor and 8GB RAM",
            RPiGenerator,
            RPiHWManager,
            RPiHost,
            RPICompiler,
        )
    )

    knowledge_repository.register_hw_platform(
        HWPlatform(
            "rpi4",
            "Raspberry PI 4 with A72 processor and 4GB RAM",
            RPiGenerator,
            RPiHWManager,
            RPiHost,
            RPICompiler,
        )
    )

    return knowledge_repository


def setup_example_optimization_criteria(dataset_spec, device) -> OptimizationCriteria:
    criteria = OptimizationCriteria()
    data_sample = torch.randn((1, 1, 28, 28), dtype=torch.float32, device=device)
    flops_estimator = FLOPsEstimator(data_sample)
    accuracy_estimator = TrainMetricsEstimator(
        SupervisedTrainer(
            device,
            dataset_spec,
            batch_size=64,
        ),
        metric_name="accuracy",
        n_estimation_epochs=3,
    )
    criteria.register_objective(estimator=accuracy_estimator)

    criteria.register_objective(estimator=flops_estimator, transform=log10, weight=-2.0)

    return criteria


def setup_mnist(path_to_test_data: Path):
    transf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset_spec = DatasetSpecification(
        dataset=MNISTWrapper(root=path_to_test_data, transform=transf),
        deployable_dataset_path=path_to_test_data,
    )
    return dataset_spec


def measure_on_device(
    explorer: Explorer,
    top_models: List,
    metric_to_source: dict[Metric, Path],
    retrain_epochs: int,
    device: str,
    dataset_spec: DatasetSpecification,
    model_suffix: str = ".pt",
):

    metrics_to_measurements = {"accuracy after retrain in %": []}
    for metric, _ in metric_to_source.items():
        metrics_to_measurements.update({metric.value + " on device": []})

    for i, model in enumerate(top_models):

        trainer = SupervisedTrainer(
            device=device,
            dataset_spec=dataset_spec,
            loss_fn=nn.CrossEntropyLoss(),
            batch_size=64,
            extra_metrics={"accuracy": accuracy_fn},
        )
        trainer.configure_optimizer(optimizer=optim.Adam(model.parameters(), lr=1e-3))
        trainer.train(model, epochs=retrain_epochs)
        test_metrics, _ = trainer.test(model)
        metrics_to_measurements["accuracy after retrain in %"].append(
            test_metrics.get("accuracy")
        )
        model_name = "model_" + str(i) + model_suffix
        explorer.generate_for_hw_platform(model, model_name, dataset_spec)

        for metric in metric_to_source.keys():
            measure = explorer.run_measurement(metric, model_name)
            metrics_to_measurements[metric.value + " on device"].append(
                measure[metric.value]["value"]
            )

    return build_search_space_measurements_file(
        metrics_to_measurements,
        explorer.metric_dir / "metrics.json",
        explorer.model_dir / "models.json",
        explorer.experiment_dir / "experiment_data.csv",
    )
