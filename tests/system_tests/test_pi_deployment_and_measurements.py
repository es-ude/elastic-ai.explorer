import shutil
from elasticai.explorer.config import HWNASConfig, DeploymentConfig
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import Generator, KnowledgeRepository
from elasticai.explorer.generator.deployment.compiler import RPICompiler
from elasticai.explorer.generator.deployment.hw_manager import (
    DOCKER_CONTEXT_DIR,
    RPiHWManager,
    Metric,
)
from elasticai.explorer.generator.model_generator.model_generator import (
    TorchscriptModelGenerator,
)
from elasticai.explorer.generator.deployment.device_communication import RPiHost
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from pathlib import Path

from elasticai.explorer.training import data
from elasticai.explorer.utils.data_utils import setup_mnist_for_cpp
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
            Generator(
                "rpi5",
                "Raspberry PI 5 with A76 processor and 8GB RAM",
                TorchscriptModelGenerator,
                RPiHWManager,
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
        transf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        root_dir_cpp_mnist = ROOT_DIR / Path("data/cpp-mnist")
        setup_mnist_for_cpp(path_to_dataset, root_dir_cpp_mnist, transf)
        metric_to_source = {
            Metric.ACCURACY: Path(
                "code/measure_accuracy_mnist.cpp"
            ),  # test relative path
            Metric.LATENCY: (
                DOCKER_CONTEXT_DIR / Path("code/measure_latency.cpp")
            ),  # test absolute path
        }
        self.RPI5explorer.hw_setup_on_target(
            metric_to_source,
            data.DatasetSpecification(
                dataset_type=data.MNISTWrapper,
                dataset_location=path_to_dataset,
                deployable_dataset_path=root_dir_cpp_mnist,
                transform=transf,
            ),
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
