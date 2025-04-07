from elasticai.explorer.config import HWNASConfig, ModelConfig, DeploymentConfig
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import HWPlatform, KnowledgeRepository
from elasticai.explorer.platforms.deployment.compiler import Compiler
from elasticai.explorer.platforms.deployment.manager import PIHWManager, Metric
from elasticai.explorer.platforms.generator.generator import PIGenerator
from elasticai.explorer.platforms.deployment.device_communication import Host
from pathlib import Path


class TestDeploymentAndMeasurement:
    def setUp(self):
        self.hwnas_cfg = HWNASConfig(
            config_path=Path("tests/system_tests/test_configs/hwnas_config.yaml")
        )
        self.deploy_cfg = DeploymentConfig(
            config_path=Path("tests/system_tests/test_configs/deployment_config.yaml")
        )
        self.model_cfg = ModelConfig(
            config_path=Path("tests/system_tests/test_configs/model_config.yaml")
        )
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
        self.RPI5explorer = Explorer(knowledge_repository, "only_for_system_tests")
        self.RPI5explorer.choose_target_hw(self.deploy_cfg)
        self.RPI5explorer._model_dir = Path("tests/system_tests/samples")
        self.model_name = "ts_model_0.pt"
        self.RPI5explorer.hw_setup_on_target()

    def test_run_accuracy_measurement(self):
        self.setUp()
        assert (
            type(
                self.RPI5explorer.run_measurement(
                    Metric.ACCURACY,
                    model_name=self.model_name,
                    path_to_data=Path("docker/data"),
                )["Accuracy"]["value"]
            )
            == float
        )
        self.tearDown()

    def test_run_latency_measurement(self):
        self.setUp()
        assert (
            type(
                self.RPI5explorer.run_measurement(
                    Metric.LATENCY,
                    model_name=self.model_name,
                    path_to_data=Path("docker/data"),
                )["Latency"]["value"]
            )
            == int
        )
        self.tearDown()

    def tearDown(self):

        self.RPI5explorer.clear_experiment_folder()
        del self.RPI5explorer
