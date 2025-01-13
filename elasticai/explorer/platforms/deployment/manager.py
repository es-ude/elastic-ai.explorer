import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from fabric import Connection
from invoke import Result
from python_on_whales import docker

from settings import ROOT_DIR

CONTEXT_PATH = ROOT_DIR / "docker"


@dataclass
class ConnectionData:
    host: str
    user: str


class HWManager(ABC):

    @abstractmethod
    def deploy_model_and_evaluate(
        self, connection_info: ConnectionData, path_to_model
    ) -> int:
        pass

    @abstractmethod
    def install_model_runner_on_target(
        self, connection_info: ConnectionData, path_to_program=None
    ):
        pass

    @abstractmethod
    def install_accuracy_measurement_on_target(self, connection_info: ConnectionData, path_to_program: str = None
    ):
        pass

    @abstractmethod
    def deploy_model_and_measure_accuracy(
        self, connection_info: ConnectionData, path_to_model: str,  path_to_data: str
    ) -> int:
        pass


class PIHWManager(HWManager):

    def __init__(self):
        if not docker.images("cross"):
            self.setup_crosscompiler()

    def setup_crosscompiler(self):
        docker.build(
            CONTEXT_PATH, file=CONTEXT_PATH / "Dockerfile.picross", tags="cross"
        )

    def compile_code(self):
        docker.build(
            CONTEXT_PATH,
            file=CONTEXT_PATH / "Dockerfile.loader",
            output={"type": "local", "dest": CONTEXT_PATH / "bin"},
        )

    def install_model_runner_on_target(
        self, connection_info: ConnectionData, path_to_program: str = None
    ):
        if path_to_program is None:
            path_to_program = str(CONTEXT_PATH) + "/bin/measure_latency"
            self.compile_code()

        with Connection(host=connection_info.host, user=connection_info.user) as conn:
            conn.put(path_to_program)


    def install_accuracy_measurement_on_target(self, connection_info: ConnectionData, path_to_program: str = None, path_to_data: str = None):
        if path_to_program is None:
            path_to_program = str(CONTEXT_PATH) + "/bin/measure_accuracy"
            self.compile_code()

        if path_to_data is None:
            path_to_data = str(CONTEXT_PATH) + "/data/mnist.zip"
            
        with Connection(host=connection_info.host, user=connection_info.user) as conn:
            conn.put(path_to_program)
            conn.put(path_to_data)
            conn.run(f"unzip -q -o {os.path.split(path_to_data)[-1]}")

    

    def deploy_model_and_evaluate(
        self, connection_info: ConnectionData, path_to_model: str
    ) -> int:
        
        self._deploy_model(connection_info, path_to_model)
        with Connection(host=connection_info.host, user=connection_info.user) as conn:
            result = conn.run(self._getcommand_eval(path_to_model), hide=True)
            if self._wasSuccessful(result):
                measurement = self._parse_measurement_eval(result)
            else:
                raise Exception(result.stderr)
        return measurement

    def deploy_model_and_measure_accuracy(
        self, connection_info: ConnectionData, path_to_model: str, path_to_data: str
    ) -> int:
        
        self._deploy_model(connection_info, path_to_model)
        with Connection(host=connection_info.host, user=connection_info.user) as conn:
            result = conn.run(self._getcommand_verify(path_to_model, path_to_data), hide=True)
            if self._wasSuccessful(result):
                measurement = self._parse_measurement_verify(result)
            else:
                raise Exception(result.stderr)
        return measurement
    
    def _deploy_model(self, connection_info: ConnectionData, path_to_model: str):
        
        with Connection(host=connection_info.host, user=connection_info.user) as conn:
            conn.put(path_to_model)
            

    def _getcommand_eval(self, path_to_model: str):
        _, tail = os.path.split(path_to_model)
        return "./measure_latency {}".format(tail)

    def _getcommand_verify(self, path_to_model: str, path_to_data: str):
        _, model_tail = os.path.split(path_to_model)
        _, data_tail = os.path.split(path_to_data)
        return "./measure_accuracy {} {}".format(model_tail, data_tail)

    def _parse_measurement_eval(self, result: Result) -> int:
        experiment_result = re.search("Inference Time: (.*) us", result.stdout)
        return int(experiment_result.group(1))

    def _parse_measurement_verify(self, result: Result) -> float:
        experiment_result = re.search("Accuracy: (.*)", result.stdout)
        return experiment_result.group(1)

    def _wasSuccessful(self, result: Result) -> bool:
        return result.ok
