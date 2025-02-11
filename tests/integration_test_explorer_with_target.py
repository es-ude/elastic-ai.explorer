import os
import unittest

import torch

from elasticai.explorer import search_space
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import HWPlatform, KnowledgeRepository
from elasticai.explorer.platforms.deployment.manager import ConnectionData, PIHWManager
from elasticai.explorer.platforms.generator.generator import PIGenerator
from torch import nn
from settings import ROOT_DIR
from tests.samples.sample_MLP import sample_MLP

SAMPLE_PATH = ROOT_DIR / "tests/samples"
OUTPUT_PATH = ROOT_DIR / "tests/outputs"

class TestExplorerWithTarget(unittest.TestCase):

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
        self.RPI5explorer = Explorer(knowledge_rep)
        self.connectionData = ConnectionData("transfair.local", "robin")    

    def test_hw_setup_on_target(self):
        model_path = SAMPLE_PATH / "models/ts_models/model_0.pt"
        self.RPI5explorer.hw_setup_on_target(self.connectionData)
        self.assertAlmostEqual(self.RPI5explorer.run_accuracy_measurement(self.connectionData, path_to_model= model_path), 0)
        self.RPI5explorer.run_latency_measurement(self.connectionData, path_to_model= model_path)
        print("hi")


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    
    unittest.main()
        