from functools import partial
import optuna
import pytest
import torch
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
import yaml

from elasticai.explorer.config import HWNASConfig, DeploymentConfig
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.hw_nas.hw_nas import objective_wrapper
from elasticai.explorer.hw_nas.search_space.construct_sp import SearchSpace, yml_to_dict
from elasticai.explorer.knowledge_repository import HWPlatform, KnowledgeRepository
from elasticai.explorer.platforms.deployment.compiler import Compiler
from elasticai.explorer.platforms.generator.generator import PIGenerator
from elasticai.explorer.platforms.deployment.device_communication import Host
from elasticai.explorer.platforms.deployment.manager import PIHWManager
from settings import ROOT_DIR
from pathlib import Path

SAMPLE_PATH = ROOT_DIR / "tests/samples"
OUTPUT_PATH = ROOT_DIR / "tests/outputs"


class TestHWNasSetupAndSearch:
    """Integration test of the Explorer HW-NAS pipeline without a target device."""

    def setup_class(self):
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
        self.RPI5explorer = Explorer(knowledge_repository)
        self.RPI5explorer.experiment_dir = Path(
            "tests/integration_tests/test_experiment"
        )
        self.model_name = "ts_model_0.pt"
        self.hwnas_cfg = HWNASConfig(
            Path("tests/integration_tests/test_configs/hwnas_config.yaml"),
        )
        self.deploy_cfg = DeploymentConfig(
            Path("tests/integration_tests/test_configs/deployment_config.yaml")
        )
        self.search_space_cfg = yml_to_dict(
            Path("elasticai/explorer/hw_nas/search_space/search_space.yml")
        )

    def test_search(self):
        self.RPI5explorer.generate_search_space(self.search_space_cfg)
        top_k_models = self.RPI5explorer.search(self.hwnas_cfg)
        assert len(top_k_models) == 2

    def test_frozentrial_to_model(self):
        study = optuna.create_study(
            sampler=optuna.samplers.RandomSampler(),
            direction="maximize",
        )
        study.optimize(
            partial(
                objective_wrapper,
                search_space_cfg=self.search_space_cfg,
                device=self.hwnas_cfg.host_processor,
            ),
            callbacks=[
                MaxTrialsCallback(
                    self.hwnas_cfg.max_search_trials,
                    states=(
                        TrialState.COMPLETE,
                        TrialState.RUNNING,
                        TrialState.WAITING,
                    ),
                )
            ],
            n_jobs=1,
            show_progress_bar=True,
        )

        best_model = study.best_trial
        search_space = SearchSpace(self.search_space_cfg)
        model = search_space.create_model_sample(best_model)
        input = torch.randn(1, 1, 28, 28).to(self.hwnas_cfg.host_processor)
        model.eval()
        model.to(self.hwnas_cfg.host_processor)
        result = model(input)
        assert len(result[0]) == 10
     
