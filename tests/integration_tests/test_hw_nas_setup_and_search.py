import os
import unittest
import torch
from elasticai.explorer import search_space
from elasticai.explorer.config import HWNASConfig
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import HWPlatform, KnowledgeRepository
from elasticai.explorer.platforms.deployment.manager import PIHWManager
from elasticai.explorer.platforms.generator.generator import PIGenerator
from torch import nn
from settings import ROOT_DIR
from tests.integration_tests.samples.sample_MLP import sample_MLP

SAMPLE_PATH = ROOT_DIR / "tests/samples"
OUTPUT_PATH = ROOT_DIR / "tests/outputs"


class TestHWNasSetupAndSearch:
    """Integration test of the Explorer HW-NAS pipeline without a target device."""

    def setUp(self):
        knowledge_rep = KnowledgeRepository()
        knowledge_rep.register_hw_platform(
            HWPlatform(
                "rpi5",
                "Raspberry PI 5 with A76 processor and 8GB RAM",
                PIGenerator,
                PIHWManager,
            )
        )
        self.RPI5explorer = Explorer(knowledge_rep, "only_for_integration_tests2")
        self.model_name = "ts_model_0.pt"
        self.hwnas_cfg = HWNASConfig(
            config_path="tests/integration/test_configs/hwnas_config.yaml"
        )

    def test_search(self):
        self.setUp()
        self.RPI5explorer.generate_search_space()
        top_k_models = self.RPI5explorer.search(self.hwnas_cfg)
        assert len(top_k_models) == 1
        assert type(top_k_models[0]) == search_space.MLP
        self.tearDown()

    def test_generate_for_hw_platform(self):
        self.setUp()
        self.RPI5explorer.choose_target_hw("rpi5")
        model = sample_MLP()

        self.RPI5explorer.generate_for_hw_platform(
            model=model, model_name=self.model_name
        )
        assert (
            os.path.exists(
                "experiments/only_for_integration_tests2/models/" + self.model_name
            )
            == True
        )
        assert (
            type(
                torch.jit.load(
                    "experiments/only_for_integration_tests2/models/" + self.model_name
                )
            )
            == torch.jit._script.RecursiveScriptModule
        )
        self.tearDown()

    def test_set_default_model(self):
        self.setUp()
        model = sample_MLP()
        self.RPI5explorer.set_default_model(model)
        assert self.RPI5explorer.default_model == model
        self.tearDown()

    def tearDown(self):
        del self.RPI5explorer
        if os.path.exists(
            "experiments/only_for_integration_tests2/models/" + self.model_name
        ):
            os.remove(
                "experiments/only_for_integration_tests2/models/" + self.model_name
            )
