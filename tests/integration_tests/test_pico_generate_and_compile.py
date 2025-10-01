import os
from pathlib import Path

from elasticai.explorer.config import DeploymentConfig, HWNASConfig
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import Generator, KnowledgeRepository
from elasticai.explorer.generator.deployment.compiler import PicoCompiler
from elasticai.explorer.generator.deployment.device_communication import RPiHost
from elasticai.explorer.generator.deployment.hw_manager import (
    DOCKER_CONTEXT_DIR,
    PicoHWManager,
)
from elasticai.explorer.generator.model_generator import tflite_to_resolver
from elasticai.explorer.generator.model_generator.model_generator import (
    TFliteModelGenerator,
)
from torchvision import transforms
from elasticai.explorer.training.data import DatasetSpecification, MNISTWrapper
from elasticai.explorer.utils.data_utils import setup_mnist_for_cpp
from settings import ROOT_DIR
from tests.integration_tests.samples import sample_MLP


class TestPicoGenerateAndCompile:
    def setup_method(self):
        knowledge_repository = KnowledgeRepository()
        knowledge_repository.register_hw_platform(
            Generator(
                "pico",
                "Pico mit RP2040",
                TFliteModelGenerator,
                PicoHWManager,
                RPiHost,
                PicoCompiler,
            )
        )
        self.RPI5explorer = Explorer(knowledge_repository)
        self.RPI5explorer.experiment_dir = ROOT_DIR / Path(
            "tests/integration_tests/test_experiment"
        )
        self.model_name = "model"

        self.hwnas_cfg = HWNASConfig(
            config_path=ROOT_DIR
            / Path("tests/integration_tests/test_configs/hwnas_config.yaml")
        )
        self.deploy_cfg = DeploymentConfig(
            config_path=ROOT_DIR
            / Path("tests/integration_tests/test_configs/deployment_config_pico.yaml")
        )

        self.RPI5explorer.choose_target_hw(self.deploy_cfg)
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

        self.dataset_spec = DatasetSpecification(
            dataset_type=MNISTWrapper,
            dataset_location=path_to_dataset,
            deployable_dataset_path=path_to_deployable_dataset,
            transform=transf,
        )

    def test_generate_for_hw_platform(self):
        self.RPI5explorer.choose_target_hw(self.deploy_cfg)
        model = sample_MLP.SampleMLP(28 * 28)

        self.RPI5explorer.generate_for_hw_platform(
            model=model, model_name=self.model_name, dataset_spec=self.dataset_spec
        )
        assert (
            os.path.exists(self.RPI5explorer.model_dir / (self.model_name + ".tflite"))
            == True
        )
        assert (
            os.path.exists(self.RPI5explorer.model_dir / (self.model_name + ".cpp"))
            == True
        )

    def test_pico_docker_compile(self):

        expected_name_of_executable = "measure_accuracy.uf2"
        self.path_to_executable = (
            DOCKER_CONTEXT_DIR / "bin" / expected_name_of_executable
        )

        compiler = PicoCompiler(deploy_cfg=self.deploy_cfg)
        if not compiler.is_setup():
            compiler.setup()
        compiler.compile_code(Path("code/pico_crosscompiler/measure_accuracy"))
        if not Path(self.path_to_executable).resolve().is_file():
            raise AssertionError(
                "File does not exist: %s" % str(self.path_to_executable)
            )

    def test_tflite_to_resolver(self):
        self.RPI5explorer.choose_target_hw(self.deploy_cfg)
        model = sample_MLP.SampleMLP(28 * 28)

        self.RPI5explorer.generate_for_hw_platform(
            model=model, model_name=self.model_name, dataset_spec=self.dataset_spec
        )
        sample_model_path = self.RPI5explorer.model_dir / (self.model_name + ".tflite")
        tflite_to_resolver.generate_resolver_h(
            sample_model_path, self.RPI5explorer.experiment_dir / "resolver_ops.h"
        )

        assert (
            os.path.exists(self.RPI5explorer.experiment_dir / "resolver_ops.h") == True
        )

    def teardown_method(self):

        try:
            os.remove(
                self.RPI5explorer.model_dir / (self.model_name + ".tflite"),
            )
        except:
            pass

        try:
            os.remove(self.RPI5explorer.model_dir / (self.model_name + ".cpp"))
        except:
            pass

        try:
            os.remove(self.path_to_executable)
        except:
            pass
        try:
            os.remove(self.RPI5explorer.experiment_dir / "resolver_ops.h")
        except:
            pass
