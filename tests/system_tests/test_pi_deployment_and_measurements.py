import tomllib

import pytest
import torch
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.hw_nas.estimators import LatencyEstimator
from elasticai.explorer.knowledge_repository import HWPlatform, KnowledgeRepository
from elasticai.explorer.platforms.deployment.compiler import CompilerParams, RPICompiler
from elasticai.explorer.platforms.deployment.hw_manager import (
    DOCKER_CONTEXT_DIR,
    RPiHWManager,
    Metric,
)
from elasticai.explorer.platforms.generator.generator import RPiGenerator
from elasticai.explorer.platforms.deployment.device_communication import (
    RPiHost,
    SSHParams,
)
from tests.integration_tests.samples.sample_MLP import SampleMLP
from torchvision.transforms import transforms
from pathlib import Path

from elasticai.explorer.training import data
from elasticai.explorer.utils.data_utils import setup_mnist_for_cpp
from settings import ROOT_DIR


class TestDeploymentAndMeasurement:
    def setup_class(self):
        with open("./tests/system_tests/system_test_settings.toml", "rb") as f:
            config = tomllib.load(f)

        ssh_params = SSHParams(
            hostname=config["RPI_HOSTNAME"], username=config["RPI_USERNAME"]
        )  # <-- Set the credentials of your RPi
        compiler_params = CompilerParams()
        knowledge_repository = KnowledgeRepository()
        knowledge_repository.register_hw_platform(
            HWPlatform(
                "rpi5",
                "Raspberry PI 5 with A76 processor and 8GB RAM",
                RPiGenerator,
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
        self.RPI5explorer.choose_target_hw(
            "rpi5", communication_params=ssh_params, compiler_params=compiler_params
        )
        self.model_name = "ts_model_0.pt"
        transf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        path_to_dataset = Path(ROOT_DIR / "data/mnist")
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
                dataset=data.MNISTWrapper(
                    path_to_dataset,
                    transform=transf,
                ),
                deployable_dataset_path=root_dir_cpp_mnist,
            ),
        )

    @pytest.mark.hardware
    def test_pi_accuracy_measurement(self):
        assert (
            type(
                self.RPI5explorer.run_measurement(
                    Metric.ACCURACY, model_name=self.model_name
                )["Accuracy"]["value"]
            )
            == float
        )

    @pytest.mark.hardware
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


class TestLatencyEstimator:
    def setup_class(self):
        with open("./tests/system_tests/system_test_settings.toml", "rb") as f:
            config = tomllib.load(f)

        ssh_params = SSHParams(
            hostname=config["RPI_HOSTNAME"], username=config["RPI_USERNAME"]
        )
        compiler_params = CompilerParams()

        host = RPiHost(ssh_params)
        compiler = RPICompiler(compiler_params)
        self.hw_manager = RPiHWManager(host, compiler)
        self.generator = RPiGenerator()

        self.hw_manager.install_code_on_target(
            DOCKER_CONTEXT_DIR / "code/measure_latency.cpp",
            Metric.LATENCY,
        )

        self.model_output_path = ROOT_DIR / "tests/system_tests/samples/test_model.pt"
        self.data_sample = torch.ones(1, 1, 28, 28)

    @pytest.mark.hardware
    def test_latency_estimator_returns_int_latency(self):
        model = SampleMLP(input_dim=28 * 28)
        estimator = LatencyEstimator(
            hw_manager=self.hw_manager,
            generator=self.generator,
            model_output_path=self.model_output_path,
            data_sample=self.data_sample,
        )
        latency, intermediates = estimator.estimate(model)
        assert isinstance(latency, int)
        assert intermediates == []
