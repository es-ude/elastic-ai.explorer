import shutil
from elasticai.explorer.config import HWNASConfig, DeploymentConfig
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import HWPlatform, KnowledgeRepository
from elasticai.explorer.platforms.deployment.compiler import RPICompiler
from elasticai.explorer.platforms.deployment.hw_manager import (
    DOCKER_CONTEXT_DIR,
    PIHWManager,
    Metric,
)
from elasticai.explorer.platforms.generator.generator import PIGenerator
from elasticai.explorer.platforms.deployment.device_communication import RPiHost
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from pathlib import Path

from settings import ROOT_DIR


class TestDeploymentAndMeasurement:
    def setup_class(self):
        self.hwnas_cfg = HWNASConfig(
            config_path=ROOT_DIR
            / Path("tests/system_tests/test_configs/hwnas_config.yaml")
        )
        self.deploy_cfg = DeploymentConfig(
            config_path=ROOT_DIR
            / Path("tests/system_tests/test_configs/deployment_config_pi.yaml")
        )
        knowledge_repository = KnowledgeRepository()
        knowledge_repository.register_hw_platform(
            HWPlatform(
                "rpi5",
                "Raspberry PI 5 with A76 processor and 8GB RAM",
                PIGenerator,
                PIHWManager,
                RPiHost,
                RPICompiler,
            )
        )
        self.RPI5explorer = Explorer(knowledge_repository)
        self.RPI5explorer.experiment_dir = ROOT_DIR / Path(
            "tests/system_tests/test_experiment"
        )
        self.RPI5explorer._model_dir = ROOT_DIR / Path("tests/system_tests/samples")
        self.RPI5explorer.choose_target_hw(self.deploy_cfg)
        self.model_name = "ts_model_0.pt"
        transf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        path_to_dataset = Path(ROOT_DIR / "data/mnist")
        MNIST(path_to_dataset, download=True, transform=transf)

        path_to_data_docker = str(ROOT_DIR / "data/mnist")
        shutil.make_archive(path_to_data_docker, "zip", path_to_dataset)

        metric_to_source = {
            Metric.ACCURACY: Path("code/measure_accuracy.cpp"), #test relative path
            Metric.LATENCY: DOCKER_CONTEXT_DIR / Path("code/measure_latency.cpp"), # test absolute path
        }
        self.RPI5explorer.hw_setup_on_target(
            metric_to_source, Path(path_to_data_docker + ".zip")
        )

    def test_pi_accuracy_measurement(self):
        assert (
            type(
                self.RPI5explorer.run_measurement(
                    Metric.ACCURACY, model_name=self.model_name
                )["Accuracy"]["value"]
            )
            == float
        )

    def test_pi_latency_measurement(self):
        assert (
            type(
                self.RPI5explorer.run_measurement(
                    Metric.LATENCY,
                    model_name=self.model_name,
                )["Latency"]["value"]
            )
            == int
        )

    def teardown_class(self):
        shutil.rmtree(self.RPI5explorer.experiment_dir)
