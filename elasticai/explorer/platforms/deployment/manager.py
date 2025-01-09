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
            self.compile_code()
            path_to_program = CONTEXT_PATH + "/bin"
        with Connection(host=connection_info.host, user=connection_info.user) as conn:
            conn.put(path_to_program)

    def deploy_model_and_evaluate(
            self, connection_info: ConnectionData, path_to_model: str
    ) -> int:
        with Connection(host=connection_info.host, user=connection_info.user) as conn:
            conn.put(path_to_model)

            result = conn.run(self._getcommand(path_to_model), hide=False)
            if self._wasSuccessful(result):
                measurement = self._parse_measurement(result)
            else:
                raise Exception(result.stderr)
        return measurement

    def _getcommand(self, path_to_model: str):
        _, tail = os.path.split(path_to_model)
        return "./measure_latency {}".format(tail)

    def _parse_measurement(self, result: Result) -> int:
        experiment_result = re.search("Inference Time: (.*) us", result.stdout)
        return int(experiment_result.group(1))

    def _wasSuccessful(self, result: Result) -> bool:
        return result.ok
