import datetime
import logging
from pathlib import Path
from typing import Optional, Any, Type

from torch import nn
from torch.nn import Module
from nni.nas.nn.pytorch import ModelSpace

from elasticai.explorer import hw_nas, utils
from elasticai.explorer.config import DeploymentConfig, ModelConfig, HWNASConfig
from elasticai.explorer.knowledge_repository.knowledge_repository import KnowledgeRepository, HWPlatform
from elasticai.explorer.platforms.deployment.manager import HWManager, Metric
from elasticai.explorer.platforms.generator.generator import Generator
from elasticai.explorer.search_space import MLP
from settings import MAIN_EXPERIMENT_DIR, ROOT_DIR


class Explorer:
    """
    The explorer class manages the HW-NAS and the deployment on hardware.
    """

    def __init__(
        self,
        knowledge_repository: KnowledgeRepository,
        experiment_name: str | None = None,
    ):
        """
        Args:
            knowledge_repository
            experiment_name (str, optional): The name of the current experiment. Defaults to timestamp at instantiation.
              This defines in which directory the results are stored inside MAIN_EXPERIMENT_DIR (from settings.py).
        """
        self.logger = logging.getLogger("explorer")
        self.default_model: Optional[nn.Module] = None
        self.target_hw_platform: Optional[HWPlatform] = None
        self.knowledge_repository: KnowledgeRepository = knowledge_repository
        self.generator: Optional[Generator] = None
        self.hw_manager: Optional[HWManager] = None
        self.search_space: Optional[Type[ModelSpace] | Module] = None
        self.model_cfg: Optional[ModelConfig] = None

        if not experiment_name:
            self.experiment_name: str = f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}"
        else:
            self.experiment_name: str = experiment_name

    @property
    def experiment_name(self):
        return self._experiment_name

    @property
    def experiment_dir(self):
        return self._experiment_dir

    @property
    def model_dir(self):
        return self._model_dir

    @property
    def metric_dir(self):
        return self._metric_dir

    @property
    def plot_dir(self):
        return self._plot_dir

    @experiment_name.setter
    def experiment_name(self, value: str):
        """Setting experiment name updates the experiment pathes aswell."""
        self._experiment_name: str = value
        self._experiment_dir: Path = MAIN_EXPERIMENT_DIR / self._experiment_name
        self._update_experiment_pathes()
    

    @experiment_dir.setter
    def experiment_dir(self, value: Path):
        """Setting the experiment directory updates the experiment name to the Path-Stem."""
        self._experiment_dir: Path = value
        self._experiment_name: str = self._experiment_dir.stem
        self._update_experiment_pathes()
       

    def set_default_model(self, model: nn.Module):
        self.default_model = model

    def set_model_cfg(self, model_cfg: ModelConfig):
        self.model_cfg = model_cfg
        self.model_cfg.dump_as_yaml(self._model_dir / "model_config.yaml")

    def generate_search_space(self):
        self.search_space = MLP()
        self.logger.info("Generated search space:\n %s", self.search_space)

    def search(self, hwnas_cfg: HWNASConfig) -> list[Any]:

        self.logger.info(
            "Start Hardware NAS with %d number of trials for top %d models ",
            hwnas_cfg.max_search_trials,
            hwnas_cfg.top_n_models,
        )

        top_models, model_parameters, metrics = hw_nas.search(
            self.search_space, hwnas_cfg
        )

        utils.save_list_to_json(
            model_parameters, path_to_dir=self._model_dir, filename="models.json"
        )
        utils.save_list_to_json(
            metrics, path_to_dir=self._metric_dir, filename="metrics.json"
        )
        hwnas_cfg.dump_as_yaml(self._experiment_dir / "hwnas_config.yaml")

        return top_models

    def choose_target_hw(self, deploy_cfg: DeploymentConfig):
        self.target_hw_platform = self.knowledge_repository.fetch_hw_info(
            deploy_cfg.target_platform_name
        )
        self.generator = self.target_hw_platform.model_generator()
        self.hw_manager = self.target_hw_platform.platform_manager(
            self.target_hw_platform.communication_protocol(deploy_cfg),
            self.target_hw_platform.compiler(deploy_cfg),
        )
        self.logger.info(
            "Configure chosen Target Hardware Platform. Name: %s, HW PLatform:\n%s",
            deploy_cfg.target_platform_name,
            self.target_hw_platform,
        )
        deploy_cfg.dump_as_yaml(self._experiment_dir / "deployment_config.yaml")

    def hw_setup_on_target(self, path_to_testdata: Path | None):
        """
        Args:
            path_to_testdata: Path to zipped testdata relative to docker context. Testdata has to be in docker context.
        """
        self.logger.info("Setup Hardware target for experiments.")
        if self.hw_manager:
            self.hw_manager.install_code_on_target(
                "measure_latency", "measure_latency.cpp"
            )
            if path_to_testdata:
                self.hw_manager.install_dataset_on_target(path_to_testdata)
            self.hw_manager.install_code_on_target(
                "measure_accuracy", "measure_accuracy.cpp"
            )
        else:
            self.logger.error(
                "HwManager is not initialized! First run choose_target_hw(deploy_cfg), before hw_setup_on_target()"
            )
            exit(-1)

    def run_measurement(self, metric: Metric, model_name: str) -> dict:
        model_path = self._model_dir / model_name
        if self.hw_manager:
            self.hw_manager.deploy_model(model_path)
            measurement = self.hw_manager.measure_metric(metric, model_path)
            self.logger.info(measurement)
        else:
            self.logger.error(
                "HwManager is not initialized! First run choose_target_hw(deploy_cfg) and hw_setup_on_target(path_to_testdata), before run_measurement()."
            )
            exit(-1)
        return measurement

    def generate_for_hw_platform(self, model: nn.Module, model_name: str) -> Any:
        model_path = self._model_dir / model_name
        if self.generator:
            return self.generator.generate(model, model_path)
        else:
            self.logger.error(
                "Generator is not initialized! First run choose_target_hw(deploy_cfg), before generate_for_hw_platform()"
            )
            exit(-1)

    def _update_experiment_pathes(self):
        self._model_dir: Path = self._experiment_dir / "models"
        self._metric_dir: Path = self._experiment_dir / "metrics"
        self._plot_dir: Path = self._experiment_dir / "plots"
        self.logger.info(f"Experiment directory changed to {self._experiment_dir} and experiment name to {self._experiment_name}")