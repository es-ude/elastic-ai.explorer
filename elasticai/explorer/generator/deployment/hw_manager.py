import json
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import shutil
import tarfile
from typing import Callable, Dict

from elasticai.explorer.generator.deployment.compiler import Compiler
from elasticai.explorer.generator.deployment.device_communication import (
    Host,
    SSHHost,
    PicoHost,
    RPiHost,
    SerialHost,
)
from elasticai.explorer.generator.model_compiler import tflite_to_resolver
from elasticai.explorer.hw_nas.search_space.quantization import (
    QuantizationScheme,
)
from elasticai.explorer.training.data import DatasetSpecification
from torch.utils.data import DataLoader

MetricFunction = Callable[[Host, "HWManager"], dict[str, dict]]


class Metric(Enum):
    LATENCY = "Latency"
    ACCURACY = "Accuracy"
    VERIFICATION = "Verification"


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

        if isinstance(source, Path):
            src_path: Path = source
            out: None | str = None
            if self.compiler is not None:
                compiled = self.compiler.compile_code(src_path, src_path.parent)
            else:
                compiled = src_path

            if isinstance(self.target, SSHHost):
                if compiled:
                    self.target.put_file(local_path=compiled, remote_path=".")
                    cmd = f"./{Path(compiled).name} {path_to_model.name}"
                    out = self.target.run_command(cmd)
            elif isinstance(self.target, SerialHost):
                path_to_executable = self.compiler.compile_code(source)
                if path_to_executable:
                    self.target.flash(local_path=path_to_executable)
                    out = self.target.receive()
            if out:
                return json.loads(out)
            else:
                return {metric.value: {"value": -1, "unit": "Error"}}

        err = TypeError(f"Unsupported source for metric {metric}. ")
        self.logger.error(err)
        raise err

    def prepare_dataset(
        self,
        dataset_spec: DatasetSpecification,
        quantization_scheme: QuantizationScheme,
    ):
        self.dataset_spec = dataset_spec
        self.quantization_scheme = quantization_scheme

    @abstractmethod
    def prepare_model(self, path_to_model: Path):
        pass

    def measure_metric(self, metric: Metric, path_to_model: Path) -> Dict:
        return self._invoke_metric_source(metric, path_to_model)


class RPiHWManager(HWManager):
    def __init__(self, target: RPiHost, compiler: Compiler):
        self.compiler = compiler
        self.docker_build_context = self.compiler.compiler_params.build_context
        self.target = target
        self.logger = logging.getLogger(
            "explorer.generator.deployment.hw_manager.RPiHWManager"
        )
        self.logger.info("Initializing PI Hardware Manager...")
        super().__init__(target, compiler)

    def prepare_measurement(self, source: Path | MetricFunction, metric: Metric):
        if isinstance(source, Callable):
            super().prepare_measurement(source, metric)
            return

        if source.is_relative_to(self.docker_build_context):
            relative_path = Path(
                "/" + str(source.relative_to(self.docker_build_context))
            )
        else:
            relative_path = Path("/" + str(source))
        path_to_executable = self.compiler.compile_code(relative_path)
        self._register_metric_to_source(metric, relative_path)
        self.target.put_file(path_to_executable, ".")

    def prepare_dataset(
        self,
        dataset_spec: DatasetSpecification,
        quantization_scheme: QuantizationScheme,
    ):
        super().prepare_dataset(dataset_spec, quantization_scheme)
        if dataset_spec.deployable_dataset_path:
            dataset_dir = dataset_spec.deployable_dataset_path
        else:
            raise Exception(
                f"There is no deployable dataset path. Cannot prepare the dataset."
            )
        archive_name = dataset_dir.with_suffix(".tar.gz")
        with tarfile.open(archive_name, "w:gz") as tar:
            tar.add(dataset_dir, arcname=dataset_dir.name)

        self.target.put_file(archive_name, ".")
        self.target.run_command(f"tar -xzf {archive_name.name} -C data")

    def prepare_model(self, path_to_model: Path):
        self.logger.info("Put model %s on target", path_to_model)
        self.target.put_file(path_to_model, ".")

    def measure_metric(self, metric: Metric, path_to_model: Path) -> dict:
        source = self._metric_to_source.get(metric)
        if not source:
            raise Exception(f"No source code registered for Metric: {metric}")
        _, tail = os.path.split(path_to_model)
        self.logger.info("Measure {} of model on device.".format(metric))

        measurement = self._invoke_metric_source(metric, path_to_model)

        self.logger.debug("Measurement on device: %s ", measurement)
        return measurement

    def build_command(self, name_of_executable: str, arguments: list[str]):
        builder = CommandBuilder(name_of_executable)
        for argument in arguments:
            builder.add_argument(argument)
        command = builder.build()
        return command


class CommandBuilder:
    def __init__(self, name_of_exec: str):
        self.command: list[str] = ["./{}".format(name_of_exec)]

    def add_argument(self, arg):
        self.command.append(arg)

    def build(self) -> str:
        return " ".join(self.command)


class PicoHWManager(HWManager):

    def __init__(self, target: PicoHost, compiler: Compiler):
        self.compiler = compiler
        self.docker_build_context = self.compiler.compiler_params.build_context
        self.target = target
        self.logger = logging.getLogger(
            "explorer.generator.deployment.hw_manager.PicoHWManager"
        )
        self.logger.info("Initializing Pico Hardware Manager...")
        super().__init__(target, compiler)

    def prepare_measurement(self, source: Path | MetricFunction, metric: Metric):

        # If the source contains the docker path, then make it relative to the docker context.
        if isinstance(source, Path) and source.is_relative_to(
            self.docker_build_context
        ):
            source = Path("/" + str(source.relative_to(self.docker_build_context)))

        # Else it assumes the path already was relative to docker context.
        elif isinstance(source, Path):
            source = Path("/" + str(source))

        super().prepare_measurement(source, metric)

    def prepare_dataset(
        self,
        dataset_spec: DatasetSpecification,
        quantization_scheme: QuantizationScheme,
    ):
        super().prepare_dataset(dataset_spec, quantization_scheme)
        target_dir = self.docker_build_context / "code/pico_crosscompiler/data"
        if not dataset_spec.deployable_dataset_path:
            raise ValueError(
                "For deployment on Pico the DatasetSpecification must have deployable_dataset_path set."
            )
        for file in dataset_spec.deployable_dataset_path.iterdir():
            if file.is_file():
                shutil.copyfile(file, target_dir / file.name)

    def _invoke_metric_source(self, metric: Metric, path_to_model: Path) -> Dict:
        source = self._metric_to_source.get(metric)
        if not source:
            raise Exception(f"No source code registered for Metric: {metric}")

        path_to_resolver = Path(
            str(self.docker_build_context) + f"{source}/resolver_ops.h"
        )
        tflite_to_resolver.generate_resolver_h(
            path_to_model,
            path_to_resolver,
        )
        return super()._invoke_metric_source(metric, path_to_model)

    def prepare_model(self, path_to_model: Path):
        shutil.copyfile(
            path_to_model.parent / (path_to_model.stem + ".cpp"),
            self.docker_build_context / "code/pico_crosscompiler/data/model.cpp",
        )
