import datetime
import logging
from pathlib import Path
from typing import Optional, Any, Type
from torch import nn

from elasticai.explorer.hw_nas import hw_nas

from elasticai.explorer.hw_nas.search_space.utils import yaml_to_dict
from elasticai.explorer.knowledge_repository import KnowledgeRepository, HWPlatform
from elasticai.explorer.platforms.deployment.compiler import DockerParams
from elasticai.explorer.platforms.deployment.device_communication import (
    SSHParams,
    SerialParams,
)
from elasticai.explorer.platforms.deployment.hw_manager import (
    HWManager,
    Metric,
)
from elasticai.explorer.platforms.generator.generator import Generator
from elasticai.explorer.training.trainer import Trainer
from elasticai.explorer.training import data
from elasticai.explorer.utils import data_utils
from settings import MAIN_EXPERIMENT_DIR


class Explorer:
    """
    The explorer class manages the HW-NAS and the deployment on hardware.
    """

    def __init__(
        self,
        knowledge_repository: KnowledgeRepository,
        experiment_name: Optional[str] = None,
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
        self.search_space_cfg: Optional[dict] = None

        if not experiment_name:
            self.experiment_name: str = f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}"
        else:
            self.experiment_name: str = experiment_name

    @property
    def experiment_name(self):  # type: ignore
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
    def experiment_name(self, value: str):  # type: ignore
        """Setting experiment name updates the experiment paths as well."""
        self._experiment_name: str = value
        self._experiment_dir: Path = MAIN_EXPERIMENT_DIR / self._experiment_name
        self._update_experiment_paths()

    @experiment_dir.setter
    def experiment_dir(self, value: Path):
        """Setting the experiment directory updates the experiment name to the Path-Stem."""
        self._experiment_dir: Path = value
        self._experiment_name: str = self._experiment_dir.stem
        self._update_experiment_paths()

    def generate_search_space(self, path_to_searchspace: Path):
        self.search_space_cfg = yaml_to_dict(path_to_searchspace)
        self.logger.info("Generated search space:\n %s", self.search_space_cfg)

    def search(
        self,
        dataset_spec: data.DatasetSpecification,
        trainer_class: Type[Trainer],
        search_algorithm: hw_nas.SearchAlgorithm = hw_nas.SearchAlgorithm.RANDOM_SEARCH,
        hardware_constraints: hw_nas.HardwareConstraints = hw_nas.HardwareConstraints(),
        hw_nas_parameters: hw_nas.HWNASParameters = hw_nas.HWNASParameters(),
    ) -> list[Any]:

        self.logger.info(
            "Start Hardware NAS with %d number of trials for top %d models ",
            hw_nas_parameters.max_search_trials,
            hw_nas_parameters.top_n_models,
        )
        if self.search_space_cfg:
            top_models, model_parameters, metrics = hw_nas.search(
                search_space_cfg=self.search_space_cfg,
                dataset_spec=dataset_spec,
                trainer_class=trainer_class,
                search_algorithm=search_algorithm,
                hw_nas_parameters=hw_nas_parameters,
                hardware_constraints=hardware_constraints,
            )
        else:
            self.logger.error(
                "Generate a searchspace before starting the HW-NAS with Explorer.search()!"
            )
            exit(-1)

        data_utils.save_list_to_json(
            model_parameters, path_to_dir=self._model_dir, filename="models.json"
        )
        data_utils.save_list_to_json(
            metrics, path_to_dir=self._metric_dir, filename="metrics.json"
        )

        return top_models

    def choose_target_hw(
        self,
        target_platform_name: str,
        docker_params: DockerParams,
        communication_params: SSHParams | SerialParams,
    ):
        self.target_hw_platform = self.knowledge_repository.fetch_hw_info(
            target_platform_name
        )
        self.generator = self.target_hw_platform.model_generator()
        self.hw_manager = self.target_hw_platform.platform_manager(
            self.target_hw_platform.communication_protocol(communication_params),
            self.target_hw_platform.compiler(docker_params),
        )
        self.logger.info(
            "Configure chosen Target Hardware Platform. Name: %s, HW PLatform:\n%s",
            target_platform_name,
            self.target_hw_platform,
        )

    def hw_setup_on_target(
        self, metric_to_source: dict[Metric, Path], data_spec: data.DatasetSpecification
    ):
        """
        Args:
            path_to_testdata: Path to testdata. Format depends on the HWManager implementation.
            metric_to_source: Dictionary mapping Metric to source code Path inside the docker context.
              E.g.: metric_to_source = {Metric.ACCURACY: Path("/path/to/measure_accuracy.cpp")}
        """
        self.logger.info("Setup Hardware target for experiments.")

        if not self.hw_manager:
            self.logger.error(
                "HwManager is not initialized! First run choose_target_hw(deploy_cfg) before hw_setup_on_target()."
            )
            exit(-1)

        self.hw_manager.install_dataset_on_target(data_spec)

        for metric, source in metric_to_source.items():
            self.logger.info(f"Installing program for {metric.name}: {source}")
            self.hw_manager.install_code_on_target(source, metric)

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

    def generate_for_hw_platform(
        self, model: nn.Module, model_name: str, dataset_spec: data.DatasetSpecification
    ) -> Any:
        model_path = self._model_dir / model_name

        dataset = dataset_spec.dataset_type(
            dataset_spec.dataset_location,
            transform=dataset_spec.transform,
        )
        sample_input, _ = next(iter(dataset))
        if self.generator:
            return self.generator.generate(model, model_path, sample_input)
        else:
            self.logger.error(
                "Generator is not initialized! First run choose_target_hw(deploy_cfg), before generate_for_hw_platform()"
            )
            exit(-1)

    def _update_experiment_paths(self):
        self._model_dir: Path = self._experiment_dir / "models"
        self._metric_dir: Path = self._experiment_dir / "metrics"
        self._plot_dir: Path = self._experiment_dir / "plots"
        self.logger.info(
            f"Experiment directory changed to {self._experiment_dir} and experiment name to {self._experiment_name}"
        )
