import math
import os
import shutil
from elasticai.explorer.config import HWNASConfig, ModelConfig, DeploymentConfig
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import HWPlatform, KnowledgeRepository
from elasticai.explorer.platforms.deployment.compiler import PicoCompiler
from elasticai.explorer.platforms.deployment.manager import (
    Metric,
    PicoHWManager,
)
from elasticai.explorer.platforms.generator.generator import PicoGenerator
from elasticai.explorer.platforms.deployment.device_communication import (
    PicoHost,
)
from pathlib import Path

from elasticai.explorer.utils import setup_mnist_for_cpp
from settings import ROOT_DIR


class TestPicoDeploymentAndMeasurement:
    def setup_class(self):
        self.hwnas_cfg = HWNASConfig(
            config_path=ROOT_DIR
            / Path("tests/system_tests/test_configs/hwnas_config.yaml")
        )
        self.deploy_cfg = DeploymentConfig(
            config_path=ROOT_DIR
            / Path("tests/system_tests/test_configs/deployment_config_pico.yaml")
        )
        self.model_cfg = ModelConfig()
        knowledge_repository = KnowledgeRepository()
        knowledge_repository.register_hw_platform(
            HWPlatform(
                "pico",
                "Pico with RP2040 MCU and 2MB control memory",
                PicoGenerator,
                PicoHWManager,
                PicoHost,
                PicoCompiler,
            )
        )
        self.pico_explorer = Explorer(knowledge_repository)
        self.pico_explorer.experiment_dir = ROOT_DIR / Path(
            "tests/system_tests/test_experiment"
        )
        self.pico_explorer._model_dir = ROOT_DIR / Path("tests/system_tests/samples")
        self.pico_explorer.choose_target_hw(self.deploy_cfg)
        self.model_name = "ts_model_0.tflite"
        root_dir_mnist = str(ROOT_DIR / "data/mnist")
        root_dir_cpp_mnist = str(ROOT_DIR / "data/cpp-mnist")
        setup_mnist_for_cpp(root_dir_mnist, root_dir_cpp_mnist)

        metric_to_program_id = {
            Metric.ACCURACY: "measure_accuracy",
            Metric.LATENCY: "measure_latency",
        }
        self.pico_explorer.hw_setup_on_target(metric_to_program_id, Path(root_dir_cpp_mnist))

    def test_pico_accuracy_measurment(self):
        assert math.isclose(
            self.pico_explorer.run_measurement(
                Metric.ACCURACY, model_name=self.model_name
            )["Accuracy"]["value"],
            86.719,
            abs_tol=0.01,
        )

    def test_run_latency_measurement(self):
        assert (
            type(
                self.pico_explorer.run_measurement(
                    Metric.LATENCY,
                    model_name=self.model_name,
                )["Latency"]["value"]
            )
            == int
        )
