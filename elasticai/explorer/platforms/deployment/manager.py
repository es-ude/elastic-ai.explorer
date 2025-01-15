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
    def measure_latency(
        self, connection_info: ConnectionData, path_to_model
    ) -> int:
        pass

    @abstractmethod
    def install_latency_measurement_on_target(
        self, connection_info: ConnectionData, path_to_program=None
    ):
        pass

    @abstractmethod
    def install_accuracy_measurement_on_target(self, connection_info: ConnectionData, path_to_program: str = None
    ):
        pass

    @abstractmethod
    def measure_accuracy(
        self, connection_info: ConnectionData, path_to_model: str,  path_to_data: str
    ) -> int:
        pass
    @abstractmethod
    def deploy_model(
        self, connection_info: ConnectionData, path_to_model: str
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

    def install_latency_measurement_on_target(
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

    

    def measure_latency(
        self, connection_info: ConnectionData, path_to_model: str
    ) -> int:
    
        with Connection(host=connection_info.host, user=connection_info.user) as conn:
            measurement = self._run_latency(conn, path_to_model)
        return measurement

    def measure_accuracy(
        self, connection_info: ConnectionData, path_to_model: str, path_to_data: str
    ) -> int:
        
        with Connection(host=connection_info.host, user=connection_info.user) as conn:
            measurement = self._run_accuracy(conn, path_to_model, path_to_data)
        return measurement
    

    def deploy_model(self, connection_info: ConnectionData, path_to_model: str):
        with Connection(host=connection_info.host, user=connection_info.user) as conn:
            conn.put(path_to_model)


    def _run_accuracy(self, conn: Connection, path_to_model: str, path_to_data: str) -> float:

        _, model_tail = os.path.split(path_to_model)
        _, data_tail = os.path.split(path_to_data)
        command  = "./measure_accuracy {} {}".format(model_tail, data_tail)
    
        result = conn.run(command, hide=True)
        if self._wasSuccessful(result):
            experiment_result = re.search("Accuracy: (.*)", result.stdout)
            measurement = experiment_result.group(1)
        else:
            raise Exception(result.stderr)
        return measurement
    


    def _run_latency(self, conn: Connection, path_to_model: str) -> int:

        _, tail = os.path.split(path_to_model)
        command =  "./measure_latency {}".format(tail)
    
        result = conn.run(command, hide=True)
        if self._wasSuccessful(result):
            experiment_result = re.search("Inference Time: (.*) us", result.stdout)
            measurement = int(experiment_result.group(1))
        else:
            raise Exception(result.stderr)
        return measurement
    

    # def _getcommand_latency(self, path_to_model: str):
    #     _, tail = os.path.split(path_to_model)
    #     return "./measure_latency {}".format(tail)

    # def _getcommand_accuracy(self, path_to_model: str, path_to_data: str):
    #     _, model_tail = os.path.split(path_to_model)
    #     _, data_tail = os.path.split(path_to_data)
    #     return "./measure_accuracy {} {}".format(model_tail, data_tail)

    # def _parse_measurement_latency(self, result: Result) -> int:
    #     experiment_result = re.search("Inference Time: (.*) us", result.stdout)
    #     return int(experiment_result.group(1))

    # def _parse_measurement_accuracy(self, result: Result) -> float:
    #     experiment_result = re.search("Accuracy: (.*)", result.stdout)
    #     return experiment_result.group(1)

    def _wasSuccessful(self, result: Result) -> bool:
        return result.ok
