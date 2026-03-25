from pathlib import Path
from typing import Any, List
import torch
from torch import nn
from torchvision.transforms import transforms
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.generator.deployment.compiler import (
    PicoCompiler,
    RPICompiler,
)
from elasticai.explorer.generator.deployment.device_communication import (
    PicoHost,
    RPiHost,
)
from elasticai.explorer.generator.deployment.hw_manager import (
    PicoHWManager,
    RPiHWManager,
)
from elasticai.explorer.generator.generator import Generator
from elasticai.explorer.generator.model_builder.model_builder import PicoModelBuilder
from elasticai.explorer.generator.model_translator.model_translator import (
    TFliteModelTranslator,
    TorchscriptModelTranslator,
)
from elasticai.explorer.hw_nas.estimators import (
    TrainMetricsEstimator,
    FLOPsEstimator,
)
from elasticai.explorer.hw_nas.optimization_criteria import OptimizationCriteria
from math import log10
from elasticai.explorer.hw_nas.search_space.quantization import QuantizationScheme

from elasticai.explorer.generator_registry import GeneratorRegistry


from elasticai.explorer.training.data import DatasetSpecification, MNISTWrapper
from elasticai.explorer.training.trainer import SupervisedTrainer, accuracy_fn
from torch import optim

from elasticai.explorer.utils.data_to_csv import build_search_space_measurements_file
from torch.utils.data import DataLoader


def setup_generator_registry() -> GeneratorRegistry:
    generator_registry = GeneratorRegistry()
    generator_registry.register_generator(
        Generator(
            "pico",
            "Pico with RP2040 MCU and 2MB control memory",
            TFliteModelTranslator,
            PicoHWManager,
            PicoHost,
            PicoCompiler,
            PicoModelBuilder,
        )
    )
    generator_registry.register_generator(
        Generator(
            "rpi5",
            "Raspberry PI 5 with A76 processor and 8GB RAM",
            TorchscriptModelTranslator,
            RPiHWManager,
            RPiHost,
            RPICompiler,
        )
    )

    generator_registry.register_generator(
        Generator(
            "rpi4",
            "Raspberry PI 4 with A72 processor and 4GB RAM",
            TorchscriptModelTranslator,
            RPiHWManager,
            RPiHost,
            RPICompiler,
        )
    )

    return generator_registry


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
        dataset=MNISTWrapper(root=path_to_test_data, transform=transf),
        deployable_dataset_path=path_to_test_data,
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

        sample_sample, _ = next(iter(dataset_spec.dataset))

        explorer.generate_for_hw_platform(
            model, model_name, sample_sample.unsqueeze(1), quant_scheme
        )

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
