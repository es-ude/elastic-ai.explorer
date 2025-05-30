import shutil
from elasticai.explorer.config import HWNASConfig, ModelConfig, DeploymentConfig
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import HWPlatform, KnowledgeRepository
from elasticai.explorer.platforms.deployment.compiler import Compiler
from elasticai.explorer.platforms.deployment.manager import PIHWManager, Metric
from elasticai.explorer.platforms.generator.generator import PIGenerator
from elasticai.explorer.platforms.deployment.device_communication import Host
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from pathlib import Path


class TestDeploymentAndMeasurement:
    def setUp(self):
        self.hwnas_cfg = HWNASConfig(
            config_path=Path("tests/system_tests/test_configs/hwnas_config.yaml")
        )
        self.deploy_cfg = DeploymentConfig(
            config_path=Path("tests/system_tests/test_configs/deployment_config.yaml")
        )
        self.model_cfg = ModelConfig()
        knowledge_repository = KnowledgeRepository()
        knowledge_repository.register_hw_platform(
            HWPlatform(
                "rpi5",
                "Raspberry PI 5 with A76 processor and 8GB RAM",
                PIGenerator,
                PIHWManager,
                Host,
                Compiler,
            )
        )
        self.RPI5explorer = Explorer(knowledge_repository)
        self.RPI5explorer.experiment_dir = Path("tests/system_tests/test_experiment")
        self.RPI5explorer._model_dir = Path("tests/system_tests/samples")
        self.RPI5explorer.choose_target_hw(self.deploy_cfg)
        self.model_name = "ts_model_0.pt"
        transf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        path_to_dataset = Path("tests/system_tests/samples/data/mnist")
        MNIST(path_to_dataset, download=True, transform=transf)
        
        path_to_data_docker = "docker/data/mnist"
        shutil.make_archive(path_to_data_docker, "zip", path_to_dataset)
        self.RPI5explorer.hw_setup_on_target(Path(path_to_data_docker + ".zip"))

    def test_run_accuracy_measurement(self):
        self.setUp()
        assert (
            type(
                self.RPI5explorer.run_measurement(
                    Metric.ACCURACY, model_name=self.model_name
                )["Accuracy"]["value"]
            )
            == float
        )

    def test_run_latency_measurement(self):
        self.setUp()
        assert (
            type(
                self.RPI5explorer.run_measurement(
                    Metric.LATENCY,
                    model_name=self.model_name,
                )["Latency"]["value"]
            )
            == int
        )
