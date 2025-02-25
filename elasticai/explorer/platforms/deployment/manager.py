import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path

from python_on_whales import docker

from elasticai.explorer.platforms.deployment.device_communication import Host
from settings import ROOT_DIR

CONTEXT_PATH = ROOT_DIR / "docker"


class HWManager(ABC):

    @abstractmethod
    def install_latency_measurement_on_target(self, path_to_program=None):
        pass

    @abstractmethod
    def install_accuracy_measurement_on_target(self, path_to_program: str = None):
        pass

    @abstractmethod
    def measure_accuracy(self, path_to_model: str, path_to_data: str) -> int:
        pass

    @abstractmethod
    def measure_latency(self, path_to_model) -> int:
        pass

    @abstractmethod
    def install_code_on_target(self, path_to_program: str = None
                               ):
        pass

    @abstractmethod
    def deploy_model(self, path_to_model: str) -> int:
        pass


class Compiler:
    def __init__(self):
        self.tag: str = "cross"
        if not self.is_setup():
            self.setup()

    def is_setup(self) -> bool:
        return docker.images(self.tag)

    def setup(self) -> None:
        self.logger.info("Crosscompiler has not been Setup. Setup Crosscompiler...")
        docker.build(
            CONTEXT_PATH, file=CONTEXT_PATH / "Dockerfile.picross", tags=self.tag
        )
        self.logger.debug("Crosscompiler available now.")

    def compile_code(self, path_to_program: str):
        docker.build(
            CONTEXT_PATH,
            file=CONTEXT_PATH / "Dockerfile.loader",
            output={"type": "local", "dest": CONTEXT_PATH / "bin"},
        )
        self.logger.info("Compilation finished. Program available in %s", CONTEXT_PATH / "bin")


class PIHWManager(HWManager):

    def __init__(self, target: Host, compiler: Compiler):
        self.logger = logging.getLogger("explorer.platforms.deployment.manager.PIHWManager")
        self.logger.info("Initializing PI Hardware Manager...")
        self.compiler = compiler
        self.target: Host = target

    def install_code_on_target(self, path_to_program: str
                               ):
        # if path_to_program is None:
        #     path_to_program = str(CONTEXT_PATH) + "/bin/measure_latency"

        self.compiler.compile_code(path_to_program)
        self.target.put_file(path_to_program, ".")

    def install_latency_measurement_on_target(self, path_to_program: str = None):
        self.logger.info("Install latency measurement code on target...")
        if path_to_program is None:
            self.logger.info("Latency measurement is not compiled yet...")
            path_to_program = str(CONTEXT_PATH) + "/bin/measure_latency"
            self.logger.info("Compile latency measurement code.")
            self.compiler.compile_code(path_to_program)

            self.target.put_file(path_to_program, ".")
            self.logger.info("Latency measurements available on Target")

    # todo: don't compile both scripts twice...
    def install_accuracy_measurement_on_target(self, path_to_program: str = None):
        self.logger.info("Install accuracy measurement code on target...")
        if path_to_program is None:
            self.logger.info("Accuracy measurement is not compiled yet...")
            path_to_program = str(CONTEXT_PATH) + "/bin/measure_accuracy"
            self.logger.info("Compile accuracy measurement code.")
            self.compiler.compile_code(path_to_program)
        path_to_data = None  # todo: tf is this
        if path_to_data is None:
            path_to_data = str(CONTEXT_PATH) + "/data/mnist.zip"
            self.logger.info("No path to dataset given. Set dataset path to:  %s", path_to_data)

        with self._getConnection() as conn:
            self.logger.info("Install accuracy measurement on target. Hostname: %s - User: %s", conn.host,
                             conn.user)
            conn.put(path_to_program)
            self.logger.info("Put dataset on target ")
            conn.put(path_to_data)
            conn.run(f"unzip -q -o {os.path.split(path_to_data)[-1]}")
            self.logger.info("Accuracy measurements available on target")

    # todo:measurement object was die parsefunktion beinhaltet
    def measure_latency(self, path_to_model: Path) -> (str, str):
        self.logger.info("Measure latency of model on device")
        _, tail = os.path.split(path_to_model)
        cmd = self.build_command("measure_latency", [tail])
        measurement = self.target.run_command(cmd)
        measurement = self._parse_measurement(measurement)
        self.logger.debug("Measured latency on device: %dus", measurement)
        return measurement

    def measure_accuracy(self, path_to_model: Path, path_to_data: str) -> (str, str):
        self.logger.info("Measure accuracy of model on device.")
        _, model_tail = os.path.split(path_to_model)
        _, data_tail = os.path.split(path_to_data)
        cmd = self.build_command("measure_accuracy", [model_tail, data_tail])
        measurement = self.target.run_command(cmd)
        measurement = self._parse_measurement(measurement)

        self.logger.debug("Measured accuracy on device: %0.2f\%", measurement)
        return measurement

    def deploy_model(self, path_to_model: str):
        with self._getConnection() as conn:
            self.logger.info("Put model %s on target", path_to_model)
            conn.put(path_to_model)

    def _parse_measurement(self, result) -> tuple[str, str]:
        experiment_result = re.search("(.*): (.*)", result)
        measurement = (experiment_result.group(1), experiment_result.group(2))
        return measurement

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
