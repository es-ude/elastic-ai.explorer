import math
from elasticai.explorer.config import HWNASConfig, DeploymentConfig
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import HWPlatform, KnowledgeRepository
from elasticai.explorer.platforms.deployment.compiler import PicoCompiler
from elasticai.explorer.platforms.deployment.hw_manager import (
    Metric,
    PicoHWManager,
)
from elasticai.explorer.platforms.generator.generator import PicoGenerator
from elasticai.explorer.platforms.deployment.device_communication import (
    PicoHost,
)
from pathlib import Path

from elasticai.explorer.training import data
from elasticai.explorer.utils.data_utils import setup_mnist_for_cpp
from settings import DOCKER_CONTEXT_DIR, ROOT_DIR
from torchvision import transforms


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
        root_dir_mnist = ROOT_DIR / "data/mnist"

        metric_to_source = {
            Metric.ACCURACY: Path(
                "code/pico_crosscompiler/measure_accuracy"
            ),  # test relative path
            Metric.LATENCY: (
                DOCKER_CONTEXT_DIR / Path("code/pico_crosscompiler/measure_latency")
            ),  # test absolute path
        }
        transf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        path_to_dataset = ROOT_DIR / "data/mnist"
        path_to_deployable_dataset = ROOT_DIR / "data/cpp-mnist"

        setup_mnist_for_cpp(
            root_dir_mnist=path_to_dataset,
            root_dir_cpp_mnist=path_to_deployable_dataset,
            transf=transf,
        )

        self.dataset_spec = data.DatasetSpecification(
            data.MNISTWrapper, path_to_dataset, path_to_deployable_dataset, transf
        )
        self.pico_explorer.hw_setup_on_target(metric_to_source, self.dataset_spec)

    def test_pico_accuracy_measurement(self):
        assert math.isclose(
            self.pico_explorer.run_measurement(
                Metric.ACCURACY, model_name=self.model_name
            )["Accuracy"]["value"],
            78.516,
            abs_tol=0.01,
        )

    def test_pico_latency_measurement(self):
        assert (
            type(
                self.pico_explorer.run_measurement(
                    Metric.LATENCY,
                    model_name=self.model_name,
                )["Latency"]["value"]
            )
            == int
        )
