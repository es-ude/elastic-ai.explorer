import datetime
import logging
from pathlib import Path
from typing import Optional, Any
from torch import nn

from elasticai.explorer.hw_nas import hw_nas
from elasticai.explorer.hw_nas.optimization_criteria import (
    OptimizationCriteria,
)
from elasticai.explorer.hw_nas.hw_nas import HWNASParameters, SearchStrategy
from elasticai.explorer.hw_nas.search_space.utils import yaml_to_dict
from elasticai.explorer.knowledge_repository import KnowledgeRepository, HWPlatform
from elasticai.explorer.platforms.deployment.compiler import CompilerParams
from elasticai.explorer.platforms.deployment.device_communication import (
    SSHParams,
    SerialParams,
)
from elasticai.explorer.platforms.deployment.hw_manager import (
    HWManager,
    Metric,
)
from elasticai.explorer.platforms.generator.generator import Generator
from elasticai.explorer.training import data
from elasticai.explorer.utils import data_utils
from elasticai.explorer.utils.logging_utils import (
    dataclass_instance_to_toml,
    opt_crit_registry_to_toml,
)
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
        search_strategy: SearchStrategy = SearchStrategy.RANDOM_SEARCH,
        optimization_criteria: OptimizationCriteria = OptimizationCriteria(),
        hw_nas_parameters: HWNASParameters = HWNASParameters(),
        dump_configuration: bool = True,
    ) -> list[Any]:

        self.logger.info(
            "Start Hardware NAS with %d number of trials searching for top %d models. ",
            hw_nas_parameters.max_search_trials,
            hw_nas_parameters.top_n_models,
        )
        if self.search_space_cfg:
            top_models, model_parameters, metrics = hw_nas.search(
                search_space_cfg=self.search_space_cfg,
                search_strategy=search_strategy,
                hw_nas_parameters=hw_nas_parameters,
                optimization_criteria=optimization_criteria,
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

        if dump_configuration:
            data_utils.save_to_toml(
                dataclass_instance_to_toml(
                    hw_nas_parameters,
                    additional_info={"search_strategy": search_strategy.value},
                ),
                self._experiment_dir,
                "hw_nas_params.toml",
            )
            data_utils.save_to_toml(
                opt_crit_registry_to_toml(optimization_criteria),
                self._experiment_dir,
                "optimization_criteria.toml",
            )

        return top_models

    def choose_target_hw(
        self,
        target_platform_name: str,
        compiler_params: CompilerParams,
        communication_params: SSHParams | SerialParams,
    ):
        self.target_hw_platform = self.knowledge_repository.fetch_hw_info(
            target_platform_name
        )
        self.generator = self.target_hw_platform.model_generator()
        self.hw_manager = self.target_hw_platform.platform_manager(
            self.target_hw_platform.communication_protocol(communication_params),
            self.target_hw_platform.compiler(compiler_params),
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
            path_to_testdata: Path to testdata. Format depends on the HWManager implementation. This is not here anymore
            metric_to_source: Dictionary mapping Metric to source code Path inside the docker context. this doesn't explain anything
              E.g.: metric_to_source = {Metric.ACCURACY: Path("/path/to/measure_accuracy.cpp")}
              :param data_spec: this is missing

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
