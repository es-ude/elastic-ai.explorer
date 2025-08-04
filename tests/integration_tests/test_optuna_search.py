from elasticai.explorer.config import HWNASConfig, DeploymentConfig
from functools import partial
import optuna
import torch
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from elasticai.explorer.explorer import Explorer
from elasticai.explorer.hw_nas import hw_nas
from elasticai.explorer.hw_nas.hw_nas import objective_wrapper
from elasticai.explorer.hw_nas.search_space.construct_search_space import (
    SearchSpace,
    yaml_to_dict,
)
from elasticai.explorer.knowledge_repository import HWPlatform, KnowledgeRepository
from elasticai.explorer.platforms.deployment.compiler import Compiler
from elasticai.explorer.platforms.generator.generator import PIGenerator
from elasticai.explorer.platforms.deployment.device_communication import Host
from elasticai.explorer.platforms.deployment.manager import PIHWManager
from settings import ROOT_DIR
from pathlib import Path

SAMPLE_PATH = ROOT_DIR / "tests/samples"
OUTPUT_PATH = ROOT_DIR / "tests/outputs"


class TestFrozenTrialToModel:
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
        self.hwnas_cfg = HWNASConfig(
            Path(ROOT_DIR / "tests/integration_tests/test_configs/hwnas_config.yaml")
        )
        self.deploy_cfg = DeploymentConfig(
            Path(
                ROOT_DIR / "tests/integration_tests/test_configs/deployment_config.yaml"
            )
        )
        self.search_space_cfg = yaml_to_dict(
            Path(ROOT_DIR / "elasticai/explorer/hw_nas/search_space/search_space.yaml")
        )
        self.search_space = SearchSpace(self.search_space_cfg)

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
                n_estimation_epochs=self.hwnas_cfg.n_estimation_epochs,
                flops_weight=self.hwnas_cfg.flops_weight,
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
            n_trials=self.hwnas_cfg.max_search_trials,
            show_progress_bar=True,
        )
        best_model = study.best_trial
        model = self.search_space.create_model_sample(best_model)
        input = torch.randn(1, 1, 28, 28).to(self.hwnas_cfg.host_processor)
        model.eval()
        model.to(self.hwnas_cfg.host_processor)
        result = model(input)
        assert len(result[0]) == 10

    def test_hw_nas_search(self):
        top_models, model_parameters, metrics = hw_nas.search(
            self.search_space_cfg, self.hwnas_cfg
        )
        assert len(top_models) == 2
        assert len(model_parameters) == 2
        assert len(metrics) == 2
        assert type(metrics[0]["accuracy"]) is float
