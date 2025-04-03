from elasticai.explorer.config import HWNASConfig, ModelConfig
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import HWPlatform, KnowledgeRepository
from elasticai.explorer.platforms.deployment.manager import (
    ConnectionConfig,
    PIHWManager,
)
from elasticai.explorer.platforms.generator.generator import PIGenerator


class TestDeploymentAndMeasurement:
    def setUp(self):
        self.hwnas_cfg = HWNASConfig(
            config_path="tests/integration_tests/test_configs/hwnas_config.yaml"
        )
        self.connection_cfg = ConnectionConfig(
            config_path="tests/integration_tests/test_configs/connection_config.yaml"
        )
        self.model_cfg = ModelConfig(
            config_path="tests/integration_tests/test_configs/model_config.yaml"
        )
        knowledge_rep = KnowledgeRepository()
        knowledge_rep.register_hw_platform(
            HWPlatform(
                "rpi5",
                "Raspberry PI 5 with A76 processor and 8GB RAM",
                PIGenerator,
                PIHWManager,
            )
        )
        self.RPI5explorer = Explorer(knowledge_rep, "only_for_integration_tests1")
        self.RPI5explorer.choose_target_hw("rpi5")
        self.model_name = "ts_model_0.pt"
        self.RPI5explorer.hw_setup_on_target(self.connection_cfg)

    def test_run_accuracy_measurement(self):
        self.setUp()
        assert (
            type(
                self.RPI5explorer.run_accuracy_measurement(
                    model_name=self.model_name, path_to_data="docker/data"
                )
            )
            == float
        )
        self.tearDown()

    def test_run_latency_measurement(self):
        self.setUp()
        assert (
            type(self.RPI5explorer.run_latency_measurement(model_name=self.model_name))
            == int
        )
        self.tearDown()

    def tearDown(self):
        del self.RPI5explorer
