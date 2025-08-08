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
from settings import ROOT_DIR

CONTEXT_PATH = ROOT_DIR / "docker"


class Metric(Enum):
    LATENCY = 1
    ACCURACY = 2


class HWManager(ABC):
    def __init__(self, target: Host, compiler: Compiler):
        self.compiler = compiler
        self.target: Host = target

    @abstractmethod
    def install_code_on_target(self, sourcecode_identifier: str):
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

    def install_code_on_target(self, sourcecode_identifier: str):
        path_to_executable = self.compiler.compile_code(
            sourcecode_identifier + ".cpp", sourcecode_identifier
        )
        self.target.put_file(str(path_to_executable), ".")

    def install_dataset_on_target(self, path_to_dataset: Path):
        self.target.put_file(str(path_to_dataset), ".")
        self.target.run_command(
            f"unzip -q -o {os.path.split(path_to_dataset)[-1]} -d data"
        )

    def measure_metric(self, metric: Metric, path_to_model: Path) -> dict:
        _, tail = os.path.split(path_to_model)
        self.logger.info("Measure {} of model on device.".format(metric))
        cmd = None
        match metric:
            case metric.ACCURACY:
                cmd = self.build_command("measure_accuracy", [tail, "data"])
                print("acc")
            case metric.LATENCY:
                cmd = self.build_command("measure_latency", [tail])
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

    def install_code_on_target(self, sourcecode_identifier: str):
        self.name_of_executable = sourcecode_identifier + ".uf2"
        self.sourcecode_filename = sourcecode_identifier
        if not self.compiler.is_setup():
            self.compiler.setup()

    def install_dataset_on_target(self, path_to_dataset: Path):
        shutil.copyfile(
            path_to_dataset / "mnist_images.h",
            CONTEXT_PATH / "code/pico_crosscompiler/measure_accuracy/mnist_images.h",
        )
        shutil.copyfile(
            path_to_dataset / "mnist_labels.h",
            CONTEXT_PATH / "code/pico_crosscompiler/measure_accuracy/mnist_labels.h",
        )
        # TODO change measure_latency to need no dataset

    def measure_metric(self, metric: Metric, path_to_model: Path) -> dict:
        self.logger.info("Put model %s on target", path_to_model)

        if metric is Metric.LATENCY:
            path_to_executable = self.compiler.compile_code(
                "measure_latency.uf2", "measure_latency"
            )
            self.measurements = self.target.put_file(str(path_to_executable), None)
            if self.measurements:
                measurement = self._parse_measurement(self.measurements)
            else:
                return self._parse_measurement(
                    '{"Latency": { "value": -1, "unit": "microseconds"}}'
                )
        else:
            path_to_executable = self.compiler.compile_code(
                "measure_accuracy.uf2", "measure_accuracy"
            )
            self.measurements = self.target.put_file(str(path_to_executable), None)
            if self.measurements:
                measurement = self._parse_measurement(self.measurements)
            else:
                return self._parse_measurement(
                    '{"Accuracy": { "value":  -1, "unit": "percent"}}'
                )

        self.logger.debug("Measurement on device: %s ", measurement)
        return measurement

    def deploy_model(self, path_to_model: Path):
        shutil.copyfile(
            path_to_model.parent / (path_to_model.stem + ".cpp"),
            CONTEXT_PATH / "code/pico_crosscompiler/measure_accuracy/model.cpp",
        )
        shutil.copyfile(
            path_to_model.parent / (path_to_model.stem + ".cpp"),
            CONTEXT_PATH / "code/pico_crosscompiler/measure_latency/model.cpp",
        )

    def _parse_measurement(self, result: str) -> dict:
        return json.loads(result)
