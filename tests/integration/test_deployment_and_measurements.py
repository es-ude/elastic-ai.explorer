import os
import pytest
import torch

from elasticai.explorer import search_space
from elasticai.explorer.config import HWNASConfig, ModelConfig
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import HWPlatform, KnowledgeRepository
from elasticai.explorer.platforms.deployment.manager import ConnectionConfig, PIHWManager
from elasticai.explorer.platforms.generator.generator import PIGenerator
from torch import nn
from settings import ROOT_DIR
from tests.samples.sample_MLP import sample_MLP

SAMPLE_PATH = ROOT_DIR / "tests/samples"
OUTPUT_PATH = ROOT_DIR / "tests/outputs"

class TestDeploymentAndMeasurement():
    def setUp(self):
        self.hwnas_cfg = HWNASConfig(config_path="configs/hwnas_config.yaml")
        self.connection_cfg = ConnectionConfig(config_path="configs/connection_config.yaml")
        self.model_cfg = ModelConfig(config_path="configs/model_config.yaml")
        knowledge_rep = KnowledgeRepository()
        knowledge_rep.register_hw_platform(
        HWPlatform(
            "rpi5",
            "Raspberry PI 5 with A76 processor and 8GB RAM",
            PIGenerator,
            PIHWManager,
        )
        )
        self.RPI5explorer = Explorer(knowledge_rep)
        self.RPI5explorer.choose_target_hw("rpi5")
        self.model_path = SAMPLE_PATH / "models/ts_models/model_0.pt"
        self.RPI5explorer.hw_setup_on_target(self.connectionData)

    def test_run_accuracy_measurement(self):
        self.setUp()
        self.assertIsInstance(self.RPI5explorer.run_accuracy_measurement(self.connectionData, path_to_model= self.model_path, path_to_data="data"), float)


    def test_run_latency_measurement(self):
        self.setUp()
        self.assertIsInstance(self.RPI5explorer.run_latency_measurement(self.connectionData, path_to_model= self.model_path), int)
