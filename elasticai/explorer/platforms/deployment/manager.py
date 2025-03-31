import json
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

from elasticai.explorer.platforms.deployment.compile import Compiler
from elasticai.explorer.platforms.deployment.device_communication import Host
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
    def install_code_on_target(self, name_of_executable, path_to_code: str
                               ):
        pass

    @abstractmethod
    def install_dataset_on_target(self, path_to_dataset):
        pass

    @abstractmethod
    def deploy_model(
            self, path_to_model: str
    ):
        pass

    @abstractmethod
    def measure_metric(self, metric: Metric, path_to_model: Path, path_to_data: Path):
        pass


class PIHWManager(HWManager):

    def __init__(self, target: Host, compiler: Compiler):
        self.logger = logging.getLogger("explorer.platforms.deployment.manager.PIHWManager")
        self.logger.info("Initializing PI Hardware Manager...")
        super().__init__(target, compiler)

    def install_code_on_target(self, name_of_executable, path_to_code: str
                               ):
        path_to_executable = self.compiler.compile_code(name_of_executable, path_to_code)
        self.target.put_file(path_to_executable, ".")

    # todo: probably have to do paths differently
    def install_dataset_on_target(self, path_to_dataset):
        self.target.put_file(path_to_dataset, ".")
        self.target.run_command(f"unzip -q -o {os.path.split(path_to_dataset)[-1]}")

    # todo:measurement object was die parsefunktion beinhaltet
    def measure_latency(self, path_to_model: Path) -> (str, str):
        self.logger.info("Measure latency of model on device")
        _, tail = os.path.split(path_to_model)
        cmd = self.build_command("measure_latency", [tail])
        measurement = self.target.run_command(cmd)
        measurement = self._parse_measurement(measurement)
        self.logger.debug("Measured latency on device: %dus", measurement)
        return measurement

    def measure_metric(self, metric: Metric, path_to_model: Path, path_to_data: Path):
        _, tail = os.path.split(path_to_model)
        self.logger.info("Measure {} of model on device.".format(metric))
        cmd = None
        match metric:
            case metric.ACCURACY:
                _, data_tail = os.path.split(path_to_data)
                cmd = self.build_command("measure_accuracy", [tail, data_tail])
                print("acc")
            case metric.LATENCY:
                cmd = self.build_command("measure_latency", [tail])
                print("lat")

        measurement = self.target.run_command(cmd)
        measurement = self._parse_measurement(measurement)
        self.logger.debug("Measured %s on device: %0.2f\%", metric, measurement)
        return measurement

    def deploy_model(self, path_to_model: str):
        self.logger.info("Put model %s on target", path_to_model)
        self.target.put_file(path_to_model, ".")

    def _parse_measurement(self, result: str) -> dict:
        return json.loads(result)

    def build_command(self, name_of_program: str, arguments: list[str]):
        builder = CommandBuilder(name_of_program)
        for argument in arguments:
            builder.add_argument(argument)
        command = builder.build()
        return command


class CommandBuilder:
    def __init__(self, name_of_exec: str):
        self.command: list[str] = ["./{}".format(name_of_exec)]

    def add_argument(self, arg):
        self.command.append(arg)

    def build(self):
        return " ".join(self.command)
