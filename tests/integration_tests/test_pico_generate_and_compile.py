import os
from pathlib import Path
from python_on_whales import docker

from elasticai.explorer.config import DeploymentConfig, HWNASConfig
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import HWPlatform, KnowledgeRepository
from elasticai.explorer.platforms.deployment.compiler import PicoCompiler, RPICompiler
from elasticai.explorer.platforms.deployment.device_communication import RPiHost
from elasticai.explorer.platforms.deployment.manager import CONTEXT_PATH, PicoHWManager
from elasticai.explorer.platforms.generator.generator import PicoGenerator
from settings import ROOT_DIR
from tests.integration_tests.samples.sample_MLP import sample_MLP


class TestPicoGenerateAndCompile:
    def setup_method(self):
        knowledge_repository = KnowledgeRepository()
        knowledge_repository.register_hw_platform(
            HWPlatform(
                "pico",
                "Pico mit RP2040",
                PicoGenerator,
                PicoHWManager,
                RPiHost,
                PicoCompiler,
            )
        )
        self.RPI5explorer = Explorer(knowledge_repository)
        self.RPI5explorer.experiment_dir = Path(
            "tests/integration_tests/test_experiment"
        )
        self.model_name = "model"

        self.hwnas_cfg = HWNASConfig(
            config_path=Path("tests/integration_tests/test_configs/hwnas_config.yaml")
        )
        self.deploy_cfg = DeploymentConfig(
            config_path=Path(
                "tests/integration_tests/test_configs/deployment_config_pico.yaml"
            )
        )

        self.RPI5explorer.choose_target_hw(self.deploy_cfg)

    def test_generate_for_hw_platform(self):
        self.RPI5explorer.choose_target_hw(self.deploy_cfg)
        model = sample_MLP()

        self.RPI5explorer.generate_for_hw_platform(
            model=model, model_name=self.model_name
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

        expected_name_of_executable = "app_full_precision.uf2"
        path_to_executable = CONTEXT_PATH / "bin" / expected_name_of_executable

        compiler = PicoCompiler(deploy_cfg=self.deploy_cfg)
        if not compiler.is_setup():
            compiler.setup()
        compiler.compile_code(expected_name_of_executable, "app_full_precision")
        if not Path(path_to_executable).resolve().is_file():
            raise AssertionError("File does not exist: %s" % str(path_to_executable))

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
