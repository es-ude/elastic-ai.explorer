import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from fabric import Connection
from invoke import Result
from python_on_whales import docker

from elasticai.explorer.config import ConnectionConfig
from settings import ROOT_DIR

CONTEXT_PATH = ROOT_DIR / "docker"


class HWManager(ABC):

    @abstractmethod
    def measure_latency(
            self, connection_conf: ConnectionConfig, path_to_model
    ) -> int:
        pass

    @abstractmethod
    def install_latency_measurement_on_target(
            self, connection_conf: ConnectionConfig, path_to_program=None
    ):
        pass

    @abstractmethod
    def install_accuracy_measurement_on_target(self, connection_conf: ConnectionConfig, path_to_program: str = None
                                               ):
        pass

    @abstractmethod
    def measure_accuracy(
            self, connection_conf: ConnectionConfig, path_to_model: str, path_to_data: str
    ) -> float:
        pass

    @abstractmethod
    def deploy_model(
            self, connection_conf: ConnectionConfig, path_to_model: str
    ) -> int:
        pass


class PIHWManager(HWManager):

    def __init__(self):
        self.logger = logging.getLogger("explorer.platforms.deployment.manager.PIHWManager")
        self.logger.info("Initializing PI Hardware Manager...")
        if not docker.images("cross"):
            self.setup_crosscompiler()

    def setup_crosscompiler(self):
        self.logger.info("Crosscompiler has not been Setup. Setup Crosscompiler...")
        docker.build(
            CONTEXT_PATH, file=CONTEXT_PATH / "Dockerfile.picross", tags="cross"
        )
        self.logger.info("Crosscompiler available now.")

    def compile_code(self):

        docker.build(
            CONTEXT_PATH,
            file=CONTEXT_PATH / "Dockerfile.loader",
            output={"type": "local", "dest": CONTEXT_PATH / "bin"},
        )
        self.logger.info("Compilation finished. Programs available in %s", CONTEXT_PATH / "bin")

    def install_latency_measurement_on_target(
            self, connection_conf: ConnectionConfig, path_to_program: str = None
    ):
        self.logger.info("Install latency measurement code on target...")
        if path_to_program is None:
            self.logger.info("Latency measurement is not compiled yet...")
            path_to_program = str(CONTEXT_PATH) + "/bin/measure_latency"
            self.logger.info("Compile latency measurement code.")
            self.compile_code()

        with Connection(host=connection_conf.target_name, user=connection_conf.target_user) as conn:
            self.logger.info("Install program on target. Hostname: %s - User: %s", connection_conf.target_name,
                             connection_conf.target_user)
            conn.put(path_to_program)
            self.logger.info("Latency measurements available on Target")

    # todo: don't compile both scripts twice...
    def install_accuracy_measurement_on_target(self, connection_conf: ConnectionConfig, path_to_program: str = None,
                                               path_to_data: str = None):
        self.logger.info("Install accuracy measurement code on target...")
        if path_to_program is None:
            self.logger.info("Accuracy measurement is not compiled yet...")
            path_to_program = str(CONTEXT_PATH) + "/bin/measure_accuracy"
            self.logger.info("Compile accuracy measurement code.")
            self.compile_code()

        if path_to_data is None:
            path_to_data = str(CONTEXT_PATH) + "/data/mnist.zip"
            self.logger.info("No path to dataset given. Set dataset path to:  %s", path_to_data)

        with Connection(host=connection_conf.target_name, user=connection_conf.target_user) as conn:
            self.logger.info("Install accuracy measurement on target. Hostname: %s - User: %s", connection_conf.target_name,
                             connection_conf.target_user)
            conn.put(path_to_program)
            self.logger.info("Put dataset on target ")
            conn.put(path_to_data)
            conn.run(f"unzip -q -o {os.path.split(path_to_data)[-1]}")
            self.logger.info("Accuracy measurements available on target")

    def measure_latency(
            self, connection_conf: ConnectionConfig, path_to_model: str
    ) -> int:
        self.logger.info("Measure latency of model on device")
        with Connection(host=connection_conf.target_name, user=connection_conf.target_user) as conn:
            measurement = self._run_latency(conn, path_to_model)
        self.logger.debug("Measured latency on device: %dus", measurement)
        return measurement

    def measure_accuracy(
            self, connection_conf: ConnectionConfig, path_to_model: str, path_to_data: str
    ) -> float:
        self.logger.info("Measure accuracy of model on device.")
        with Connection(host=connection_conf.target_name, user=connection_conf.target_user) as conn:
            measurement = self._run_accuracy(conn, path_to_model, path_to_data)
        self.logger.debug("Measured accuracy on device: %0.2f\%", measurement)
        return measurement

    def deploy_model(self, connection_conf: ConnectionConfig, path_to_model: str):
        with Connection(host=connection_conf.target_name, user=connection_conf.target_user) as conn:
            self.logger.info("Put model %s on target", path_to_model)
            conn.put(path_to_model)

    def _run_accuracy(self, conn: Connection, path_to_model: str, path_to_data: str) -> float:

        _, model_tail = os.path.split(path_to_model)
        _, data_tail = os.path.split(path_to_data)
        command = "./measure_accuracy {} {}".format(model_tail, data_tail)

        result = conn.run(command, hide=True)
        if self._wasSuccessful(result):
            experiment_result = re.search("Accuracy: (.*)", result.stdout)
            measurement = float(experiment_result.group(1))
        else:
            raise Exception(result.stderr)
        return float(measurement)

    def _run_latency(self, conn: Connection, path_to_model: str) -> int:

        _, tail = os.path.split(path_to_model)
        command = "./measure_latency {}".format(tail)

        result = conn.run(command, hide=True)
        if self._wasSuccessful(result):
            experiment_result = re.search("Inference Time: (.*) us", result.stdout)
            measurement = int(experiment_result.group(1))
        else:
            raise Exception(result.stderr)
        return measurement

    def _wasSuccessful(self, result: Result) -> bool:
        return result.ok
