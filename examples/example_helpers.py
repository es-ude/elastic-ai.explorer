from pathlib import Path
from typing import Any, List
import torch
from torch import nn
from torchvision.transforms import transforms
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.generator.deployment.compiler import (
    ENv5Compiler,
    PicoCompiler,
    RPICompiler,
)
from elasticai.explorer.generator.deployment.device_communication import (
    ENv5Host,
    PicoHost,
    RPiHost,
)
from elasticai.explorer.generator.deployment.hw_manager import (
    ENv5HWManager,
    PicoHWManager,
    RPiHWManager,
)
from elasticai.explorer.generator.generator import Generator
from elasticai.explorer.generator.model_builder.model_builder import CreatorModelBuilder
from elasticai.explorer.generator.model_compiler.model_compiler import (
    CreatorModelCompiler,
    TFliteModelCompiler,
    TorchscriptModelCompiler,
)
from elasticai.explorer.hw_nas.estimators import (
    TrainMetricsEstimator,
    FLOPsEstimator,
)
from elasticai.explorer.hw_nas.optimization_criteria import OptimizationCriteria
from math import log10
from elasticai.explorer.hw_nas.search_space.quantization import QuantizationScheme

from elasticai.explorer.knowledge_repository import KnowledgeRepository


from elasticai.explorer.training.data import DatasetSpecification, MNISTWrapper
from elasticai.explorer.training.trainer import SupervisedTrainer, accuracy_fn
from torch import optim

from elasticai.explorer.utils.data_to_csv import build_search_space_measurements_file


def setup_knowledge_repository() -> KnowledgeRepository:
    knowledge_repository = KnowledgeRepository()
    knowledge_repository.register_hw_platform(
        Generator(
            "pico",
            "Pico with RP2040 MCU and 2MB control memory",
            TFliteModelCompiler,
            PicoHWManager,
            PicoHost,
            PicoCompiler,
        )
    )
    knowledge_repository.register_hw_platform(
        Generator(
            "rpi5",
            "Raspberry PI 5 with A76 processor and 8GB RAM",
            TorchscriptModelCompiler,
            RPiHWManager,
            RPiHost,
            RPICompiler,
        )
    )

    knowledge_repository.register_hw_platform(
        Generator(
            "rpi4",
            "Raspberry PI 4 with A72 processor and 4GB RAM",
            TorchscriptModelCompiler,
            RPiHWManager,
            RPiHost,
            RPICompiler,
        )
    )
    knowledge_repository.register_hw_platform(
        Generator(
            "env5_s50",
            "Env5 with RP2040 and xc7s50ftgb196-2 FPGA",
            CreatorModelCompiler,
            ENv5HWManager,
            ENv5Host,
            ENv5Compiler,
            CreatorModelBuilder,
        )
    )
    knowledge_repository.register_hw_platform(
        Generator(
            "env5_s15",
            "Env5 with RP2040 and xc7s15ftgb196-2 FPGA",
            CreatorModelCompiler,
            ENv5HWManager,
            ENv5Host,
            ENv5Compiler,
            CreatorModelBuilder,
        )
    )

    return knowledge_repository


def setup_example_optimization_criteria(
    dataset_spec, device, sample_shape=(1, 1, 28, 28)
) -> OptimizationCriteria:
    criteria = OptimizationCriteria()
    data_sample = torch.randn(sample_shape, dtype=torch.float32, device=device)
    flops_estimator = FLOPsEstimator(data_sample)
    accuracy_estimator = TrainMetricsEstimator(
        SupervisedTrainer(
            device,
            dataset_spec,
            batch_size=64,
            extra_metrics={"accuracy": accuracy_fn},
        ),
        metric_name="accuracy",
        n_estimation_epochs=3,
    )
    criteria.register_objective(estimator=accuracy_estimator, weight=100)

    criteria.register_objective(estimator=flops_estimator, transform=log10, weight=-2.0)

    return criteria


def setup_mnist(path_to_test_data: Path):
    transf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset_spec = DatasetSpecification(
        dataset_type=MNISTWrapper,
        dataset_location=path_to_test_data,
        deployable_dataset_path=path_to_test_data,
        transform=transf,
    )
    return dataset_spec


def measure_on_device(
    explorer: Explorer,
    top_models: List,
    metric_to_source: Any,
    retrain_epochs: int,
    retrain_device: str,
    dataset_spec: DatasetSpecification,
    model_suffix: str = ".pt",
    top_quantization_schemes: list[QuantizationScheme] = [],
):

    metrics_to_measurements = {"accuracy after retrain in %": []}
    for metric, _ in metric_to_source.items():
        metrics_to_measurements.update({metric.value + " on device": []})

    previous_quant_scheme = None
    for i, (model, quant_scheme) in enumerate(
        zip(top_models, top_quantization_schemes)
    ):
        if i == 0 or previous_quant_scheme != quant_scheme:
            explorer.hw_setup_on_target(metric_to_source, dataset_spec, quant_scheme)
        previous_quant_scheme = quant_scheme
        trainer = SupervisedTrainer(
            device=retrain_device,
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
        explorer.generate_for_hw_platform(model, model_name, dataset_spec, quant_scheme)

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
