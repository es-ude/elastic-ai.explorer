import json
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

from elasticai.explorer.platforms.deployment.compiler import RPICompiler
from elasticai.explorer.platforms.deployment.device_communication import Host
from settings import ROOT_DIR

CONTEXT_PATH = ROOT_DIR / "docker"


class Metric(Enum):
    LATENCY = 1
    ACCURACY = 2


class HWManager(ABC):

    def __init__(self, target: Host, compiler: RPICompiler):
        self.compiler = compiler
        self.target: Host = target

    @abstractmethod
    def install_code_on_target(self, name_of_executable: str, sourcecode_filename: str):
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

    def __init__(self, target: Host, compiler: RPICompiler):
        self.logger = logging.getLogger(
            "explorer.platforms.deployment.manager.PIHWManager"
        )
        self.logger.info("Initializing PI Hardware Manager...")
        super().__init__(target, compiler)

    def install_code_on_target(self, name_of_executable: str, sourcecode_filename: str):
        path_to_executable = self.compiler.compile_code(
            name_of_executable, sourcecode_filename
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

    def __init__(self, target: Host, compiler: RPICompiler):
        self.logger = logging.getLogger(
            "explorer.platforms.deployment.manager.PIHWManager"
        )
        self.logger.info("Initializing PI Hardware Manager...")
        super().__init__(target, compiler)

    def install_code_on_target(self, name_of_executable: str, sourcecode_filename: str):
        path_to_executable = self.compiler.compile_code(
            name_of_executable, sourcecode_filename
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
