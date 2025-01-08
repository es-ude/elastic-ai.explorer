import os
import re
from abc import ABC, abstractmethod

from fabric import Connection
from python_on_whales import docker

from settings import ROOT_DIR

CONTEXT_PATH = ROOT_DIR / "docker"


class HWManager(ABC):
    @abstractmethod
    def deploy_model(self, path_to_model) -> int:
        pass

    @abstractmethod
    def deploy_program(self, path_to_program=None):
        pass


class PIHWManager(HWManager):

    def __init__(self):
        if not docker.images("cross"):
            self.build_compiler()

    def build_compiler(self):
        docker.build(
            CONTEXT_PATH, file=CONTEXT_PATH / "Dockerfile.picross", tags="cross"
        )

    def compile_code(self):
        docker.build(
            CONTEXT_PATH,
            file=CONTEXT_PATH / "Dockerfile.loader",
            output={"type": "local", "dest": CONTEXT_PATH / "bin"},
        )

    def deploy_program(self, path_to_program=None):
        if path_to_program is None:
            self.compile_code()
            path_to_program = CONTEXT_PATH + "/bin"
        with Connection(host="transpi5.local", user="ies") as conn:
            conn.put(path_to_program)

    def deploy_model(self, path_to_model: str) -> int:
        with Connection(host="transpi5.local", user="ies") as conn:
            conn.put(path_to_model)

            results = conn.run(self._getcommand(path_to_model), hide=False)
            if self._wasSuccessful(results):
                measurement = self._parse_measurement(results)
            else:
                raise Exception(results.stderr)
        return measurement

    def _getcommand(self, path_to_model):
        _, tail = os.path.split(path_to_model)
        return "./measure_latency {}".format(tail)

    def _parse_measurement(self, results) -> int:
        experiment_result = re.search("Inference Time: (.*) us", results.stdout)
        return int(experiment_result.group(1))

    def _wasSuccessful(self, results) -> bool:
        return results.ok
