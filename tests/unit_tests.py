import os
import unittest

import torch

from elasticai.explorer import search_space
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import HWPlatform, KnowledgeRepository
from elasticai.explorer.platforms.deployment.manager import PIHWManager
from elasticai.explorer.platforms.generator.generator import PIGenerator
from torch import nn
from settings import ROOT_DIR
from tests.samples.test_model import test_MLP

SAMPLE_PATH = ROOT_DIR / "tests/samples"



class TestExplorerWithoutTarget(unittest.TestCase):

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
    def test_search(self):
        self.RPI5explorer.generate_search_space()
        top_k_models = self.RPI5explorer.search(1,2)
        self.assertEqual(len(top_k_models), 1)
        self.assertEqual(type(top_k_models[0]), search_space.MLP)
        
    def test_generate_for_hw_platform(self):
        self.RPI5explorer.choose_target_hw("rpi5")
        model = test_MLP()
        ts_model_path = SAMPLE_PATH / "models/test.pt"
        self.RPI5explorer.generate_for_hw_platform( model= model , path = ts_model_path)
        self.assertTrue(os.path.exists(ts_model_path))
        self.assertTrue(torch.jit.load(SAMPLE_PATH / "models/test.pt")!= None)

    def test_set_default_model(self):
        model = test_MLP()
        self.RPI5explorer.set_default_model(model)
        self.assertEquals(self.RPI5explorer.default_model, model)
    
def only_test_search():
    suite = unittest.TestSuite()
    suite.addTest(TestExplorerWithoutTarget('test_search'))
    return suite
def only_test_generate_for_hw_platform():
    suite = unittest.TestSuite()
    suite.addTest(TestExplorerWithoutTarget('test_generate_for_hw_platform'))
    return suite
def only_test_set_default_model():
    suite = unittest.TestSuite()
    suite.addTest(TestExplorerWithoutTarget('test_set_default_model'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    #runner.run(only_test_generate_for_hw_platform())
    #runner.run(only_test_search())
    #runner.run(only_test_set_default_model())
    unittest.main()
    