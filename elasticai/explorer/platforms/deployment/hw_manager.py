import json
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import shutil

from elasticai.explorer.platforms.deployment.compiler import Compiler
from elasticai.explorer.platforms.deployment.device_communication import (
    Host,
    PicoHost,
    RPiHost,
)
from elasticai.explorer.platforms.generator import tflite_to_resolver
from settings import DOCKER_CONTEXT_DIR


class Metric(Enum):
    LATENCY = "Latency"
    ACCURACY = "Accuracy"


class HWManager(ABC):
    def __init__(self, target: Host, compiler: Compiler):
        self.compiler = compiler
        self.target: Host = target
        self._metric_to_source: dict[Metric, Path] = {}

    def _register_metric_to_source(self, metric: Metric, source: Path):
        self._metric_to_source.update({metric: source})

    @abstractmethod
    def install_code_on_target(self, source: Path, metric: Metric):
        pass

    @abstractmethod
    def install_dataset_on_target(self, path_to_dataset: Path):
        pass

    @abstractmethod
    def deploy_model(self, path_to_model: Path):
        pass

    @abstractmethod
    def measure_metric(self, metric: Metric, path_to_model: Path) -> dict:
        pass


class PIHWManager(HWManager):

    def __init__(self, target: RPiHost, compiler: Compiler):

        self.logger = logging.getLogger(
            "explorer.platforms.deployment.manager.PIHWManager"
        )
        self.logger.info("Initializing PI Hardware Manager...")
        super().__init__(target, compiler)

    def install_code_on_target(self, source: Path, metric: Metric):
        if source.is_relative_to(DOCKER_CONTEXT_DIR):
            relative_path = Path("/" + str(source.relative_to(DOCKER_CONTEXT_DIR)))
        else:
            relative_path = Path("/" + str(source))
        path_to_executable = self.compiler.compile_code(relative_path)
        self._register_metric_to_source(metric, relative_path)
        self.target.put_file(str(path_to_executable), ".")

    def install_dataset_on_target(self, path_to_dataset: Path):
        self.target.put_file(str(path_to_dataset), ".")
        self.target.run_command(
            f"unzip -q -o {os.path.split(path_to_dataset)[-1]} -d data"
        )

    def measure_metric(self, metric: Metric, path_to_model: Path) -> dict:
        source = self._metric_to_source.get(metric)
        if not source:
            raise Exception(f"No source code registered for Metric: {metric}")
        _, tail = os.path.split(path_to_model)
        self.logger.info("Measure {} of model on device.".format(metric))
        cmd = None

        match metric:
            case metric.ACCURACY:
                cmd = self.build_command(source.stem, [tail, "data"])
                print("acc")
            case metric.LATENCY:
                cmd = self.build_command(source.stem, [tail])
                print("lat")

        measurement = self.target.run_command(cmd)
        measurement = self._parse_measurement(measurement)

        self.logger.debug("Measurement on device: %s ", measurement)
        return measurement

    def deploy_model(self, path_to_model: Path):
        self.logger.info("Put model %s on target", path_to_model)
        self.target.put_file(str(path_to_model), ".")

    def _parse_measurement(self, result: str) -> dict:
        return json.loads(result)

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

        self.logger = logging.getLogger(
            "explorer.platforms.deployment.manager.PicoHWManager"
        )
        self.logger.info("Initializing Pico Hardware Manager...")
        super().__init__(target, compiler)

    def install_code_on_target(self, source: Path, metric: Metric):
        if source.is_relative_to(DOCKER_CONTEXT_DIR):
            relative_path = Path("/" + str(source.relative_to(DOCKER_CONTEXT_DIR)))
        else:
            relative_path = Path("/" + str(source))
        self._register_metric_to_source(metric, relative_path)

    def install_dataset_on_target(self, path_to_dataset: Path):
        # TODO make this more general
        shutil.copyfile(
            path_to_dataset / "mnist_images.h",
            DOCKER_CONTEXT_DIR / "code/pico_crosscompiler/data/mnist_images.h",
        )
        shutil.copyfile(
            path_to_dataset / "mnist_labels.h",
            DOCKER_CONTEXT_DIR / "code/pico_crosscompiler/data/mnist_labels.h",
        )

    def measure_metric(self, metric: Metric, path_to_model: Path) -> dict:

        self.deploy_model(path_to_model)
        source = self._metric_to_source.get(metric)
        if not source:
            self.logger.error(f"No source code registered for Metric: {metric}")
            exit(-1)
        path_to_resolver = Path(str(DOCKER_CONTEXT_DIR) + f"{source}/resolver_ops.h")
        tflite_to_resolver.generate_resolver_h(
            path_to_model,
            path_to_resolver,
        )

        path_to_executable = self.compiler.compile_code(source)
        self.measurements = self.target.put_file(str(path_to_executable), None)
        if self.measurements:
            measurement = self._parse_measurement(self.measurements)
        else:
            return self._parse_measurement(
                '{"' + metric.value + '": { "value": -1, "unit": "Error"}}'
            )

        self.logger.debug("Measurement on device: %s ", measurement)
        return measurement

    def deploy_model(self, path_to_model: Path):
        shutil.copyfile(
            path_to_model.parent / (path_to_model.stem + ".cpp"),
            DOCKER_CONTEXT_DIR / "code/pico_crosscompiler/data/model.cpp",
        )

    def _parse_measurement(self, result: str) -> dict:
        return json.loads(result)
