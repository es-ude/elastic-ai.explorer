import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

from typing import Any, Callable, Dict

from elasticai.explorer.generator.deployment.compiler import Compiler
from elasticai.explorer.generator.deployment.device_communication import (
    Host,
)
from elasticai.explorer.hw_nas.search_space.quantization import (
    QuantizationScheme,
)
from elasticai.explorer.training.data import DatasetSpecification
from torch.utils.data import DataLoader

MetricFunction = Callable[[Host, "HWManager"], dict[str, dict]]


class Metric(Enum):
    LATENCY = "Latency"
    ACCURACY = "Accuracy"


class HWManager(ABC):
    def __init__(self, target: Host, compiler: Compiler):
        self.target = target
        self.compiler = compiler
        self.path_to_executable = None
        self.dataset_spec: None | DatasetSpecification = None
        self.quantization_scheme: None | QuantizationScheme = None
        self.test_loader: None | DataLoader = None
        self._metric_to_source: dict[Metric, Path | MetricFunction] = {}
        self.logger = logging.getLogger(
            "explorer.generator.deployment.hw_manager.HWManager"
        )

    @staticmethod
    def _create_relative_path(build_context: Path, source: Any) -> Path:
        # If the source contains the docker path, then make it relative to the docker context.
        if isinstance(source, Path) and source.is_relative_to(build_context):
            source = Path("/" + str(source.relative_to(build_context)))

        # Else it assumes the path already was relative to docker context.
        elif isinstance(source, Path):
            source = Path("/" + str(source))

        return source

    def _register_metric_to_source(self, metric: Metric, source: Path | MetricFunction):
        self._metric_to_source.update({metric: source})

    def _get_metric_source(self, metric: Metric):
        return self._metric_to_source.get(metric)

    def prepare_measurement(self, source: Path | MetricFunction, metric: Metric):
        self._register_metric_to_source(metric, source)

    def _invoke_metric_source(self, metric: Metric, path_to_model: Path) -> dict:
        source = self._get_metric_source(metric)
        if not source:
            raise Exception(f"No source code registered for Metric: {metric}")

        if callable(source):
            result = source(self.target, self)
            return result
        return {}

    def prepare_dataset(
        self,
        dataset_spec: DatasetSpecification,
        quantization_scheme: QuantizationScheme | None,
    ):
        self.dataset_spec = dataset_spec
        self.quantization_scheme = quantization_scheme

    @abstractmethod
    def prepare_model(self, path_to_model: Path):
        pass

    def measure_metric(self, metric: Metric, path_to_model: Path) -> Dict:
        return self._invoke_metric_source(metric, path_to_model)


class CommandBuilder:
    def __init__(self, name_of_exec: str):
        self.command: list[str] = ["./{}".format(name_of_exec)]

    def add_argument(self, arg):
        self.command.append(arg)

    def build(self) -> str:
        return " ".join(self.command)
