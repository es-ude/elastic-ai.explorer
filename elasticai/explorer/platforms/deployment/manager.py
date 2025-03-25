import logging
import os
from pathlib import Path
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from fabric import Connection
from invoke import Result
from python_on_whales import docker

from elasticai.explorer.config import DeploymentConfig
from settings import ROOT_DIR

CONTEXT_PATH = ROOT_DIR / "docker"


class HWManager(ABC):

    @abstractmethod
    def measure_latency(
            self, deploy_dfg: DeploymentConfig, path_to_model: Path
    ) -> int:
        pass

    @abstractmethod
    def install_latency_measurement_on_target(
            self, deploy_dfg: DeploymentConfig, path_to_executable: Path = None, rebuild: bool = True
    ):
        """Installs latency measuremt on target specified in DeploymentConfig. 
        To do that, additionalen resources can be specified.

        Args:
            deploy_dfg (DeploymentConfig): Config that parameterizes the deployment.
            path_to_executable (Path, optional): Path on host to binary for the execution of the latency measurement. Defaults to None.
            rebuild (bool, optional): If true rebuilds the binary by linking to compiled library. Defaults to True.
        """
        pass

    @abstractmethod
    def install_accuracy_measurement_on_target(self, deploy_cfg: DeploymentConfig, path_to_test_data: Path = None, path_to_compiled_library: Path = None, rebuild: bool = True
                                               ):
        """Installs accuracy measuremt on target specified in DeploymentConfig. 
        To do that, additionalen resources can be specified.

        Args:
            deploy_dfg (DeploymentConfig): Config that parameterizes the deployment.
            path_to_executable (Path, optional): Path on host to binary for the execution of the latency measurement. Defaults to None.
            path_to_test_data (Path, optional): Path to test data on which to measure accuracy. Defaults to None. 
            rebuild (bool, optional): If true rebuilds the binary by linking to compiled library, else uses given executable or default executable. Defaults to True.
        """
        pass

    @abstractmethod
    def measure_accuracy(
            self, deploy_dfg: DeploymentConfig, path_to_model: Path, path_to_data: Path
    ) -> float:
        pass

    @abstractmethod
    def deploy_model(
            self, deploy_dfg: DeploymentConfig, path_to_model: Path
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

    def compile_code(self, path_to_libtorch: Path):
        docker.build(
            CONTEXT_PATH,
            file=CONTEXT_PATH / "Dockerfile.loader",
            output={"type": "local", "dest": CONTEXT_PATH / "bin"},
            build_args={"HOST_LIBTORCH_PATH": f"{path_to_libtorch}"},
        
        )
        self.logger.info("Compilation finished. Programs available in %s", CONTEXT_PATH / "bin")

    def install_latency_measurement_on_target(
            self,
        deploy_dfg: DeploymentConfig,
        path_to_executable: Path = None,
        rebuild: bool = True
    ):
        self.logger.info("Install latency measurement code on target...")
        if path_to_executable is None:
            
            path_to_executable = CONTEXT_PATH / "bin/measure_latency"
            if rebuild:
                self.logger.info("Latency measurement is not compiled yet...")
                self.logger.info("Compile latency measurement code.")
                self.compile_code(deploy_dfg.compiled_libary_path)

        with Connection(host=deploy_dfg.target_name, user=deploy_dfg.target_user) as conn:
            self.logger.info("Install program on target. Hostname: %s - User: %s", deploy_dfg.target_name,
                             deploy_dfg.target_user)
            conn.put(path_to_executable)
            self.logger.info("Latency measurements available on Target")


    def install_accuracy_measurement_on_target(self, deploy_dfg: DeploymentConfig, path_to_executable: Path = None, path_to_test_data: Path = None, rebuild: bool = True):
        self.logger.info("Install accuracy measurement code on target...")
        if path_to_executable is None:
            path_to_executable = CONTEXT_PATH / "bin/measure_accuracy"
            if rebuild:
                self.logger.info("Accuracy measurement is not compiled yet...")
                self.logger.info("Compile accuracy measurement code.")
                self.compile_code(deploy_dfg.compiled_libary_path)

        if path_to_test_data is None:
            path_to_test_data = CONTEXT_PATH / "data/mnist.zip"
            
        with Connection(host=deploy_dfg.target_name, user=deploy_dfg.target_user) as conn:
            conn.put(path_to_executable)
            self.logger.info("Put dataset on target ")
            conn.put(path_to_test_data)
            conn.run(f"unzip -q -o {os.path.split(path_to_test_data)[-1]}")
            self.logger.info("Accuracy measurements available on target")

    def measure_latency(
            self, deploy_dfg: DeploymentConfig, path_to_model: Path
    ) -> int:
        self.logger.info("Measure latency of model on device")
        with Connection(host=deploy_dfg.target_name, user=deploy_dfg.target_user) as conn:
            measurement = self._run_latency(conn, path_to_model)
        self.logger.debug("Measured latency on device: %dus", measurement)
        return measurement

    def measure_accuracy(
            self, deploy_dfg: DeploymentConfig, path_to_model: Path, path_to_data: Path
    ) -> float:
        self.logger.info("Measure accuracy of model on device.")
        with Connection(host=deploy_dfg.target_name, user=deploy_dfg.target_user) as conn:
            measurement = self._run_accuracy(conn, path_to_model, path_to_data)
        self.logger.debug("Measured accuracy on device: %0.2f\%", measurement)
        return measurement

    def deploy_model(self, deploy_dfg: DeploymentConfig, path_to_model: Path):
        with Connection(host=deploy_dfg.target_name, user=deploy_dfg.target_user) as conn:
            self.logger.info("Put model %s on target", path_to_model)
            conn.put(path_to_model)

    def _run_accuracy(self, conn: Connection, path_to_model: Path, path_to_data: Path) -> float:

        _, model_tail = os.path.split(path_to_model)
        _, data_tail = os.path.split(path_to_data)
        command = "./measure_accuracy {} {}".format(model_tail, data_tail)

        result = conn.run(command, hide=True)
        if self._wasSuccessful(result):
            experiment_result = re.search("Accuracy: (.*)", result.stdout)
            measurement = float(experiment_result.group(1))
        else:
            raise Exception(result.stderr)
        return measurement

    def _run_latency(self, conn: Connection, path_to_model: Path) -> int:

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
