from elasticai.explorer.krepo.API import KnowledgeRepoAPI
from torch import nn

from abc import ABC, abstractmethod
from typing import Any

class MetricLogger(ABC):
    @abstractmethod
    def log_metrics(
            self,
            metrics: dict[str, Any],
            model_id: str | None = None,
            step: int | None = None,
            run_name: str | None = None
    ):
        pass

    @abstractmethod
    def log_model(
            self,
            parameters: dict,
            model_architecture: dict,
            model: nn.Module | None = None,
            run_name: str | None = None
    ) -> str | None:
        pass

    def log_metric(
            self,
            name: str,
            value: Any,
            model_id: str | None = None,
            step=None,
            run_name: str | None = None
    ):
        self.log_metrics(
            metrics={name: value},
            model_id=model_id,
            step=step,
            run_name=run_name
        )


class KrepoMetricLogger(MetricLogger):
    def __init__(
            self,
            server_ip: str,
            server_port: int,

            experiment_name: str | None = None,
            nas_config: dict | None = None,
            search_space_config: dict | None = None,
            hw_platform: str | None = None,

            run_name: str | None = None,
            run_id: str | None = None,
    ):
        self.krepo = KnowledgeRepoAPI(
            server_ip=server_ip,
            server_port=server_port,
        )
        self.experiment = self.krepo.set_experiment(
            experiment_name=experiment_name,
            nas_config=nas_config,
            search_space_config=search_space_config,
            hw_platform=hw_platform,
        )
        self.run = None
        if run_id or run_name:
            self.run = self.krepo.start_run(
                run_name=run_name,
                run_id=run_id,
            )

    def log_metrics(
            self,
            metrics: dict,
            model_id: str | None = None,
            step: int | None = None,
            run_name: str | None = None
    ):
        self._create_run_if_not_exist(
            run_name=run_name,
        )
        self.krepo.log_metrics(metrics=metrics, model_id=model_id, step=step)

    def log_model(
            self,
            parameters: dict,
            model_architecture: dict,
            model: nn.Module | None = None,
            run_name: str | None = None
    ) -> str | None:
        self._create_run_if_not_exist(
            run_name=run_name,
        )
        model_id = self.krepo.log_model(
            parameters=parameters,
            model_architecture=model_architecture,
            model=model
        )
        return model_id

    @property
    def experiment_id(self):
        return self.experiment.experiment_id

    @property
    def run_id(self):
        return self.run.info.run_id if self.run else None

    @property
    def run_name(self):
        return self.run.info.run_name if self.run else None

    def _create_run_if_not_exist(self, run_name: str | None = None):
        if self.run is None:
            self.run = self.krepo.start_run(
                run_name=run_name,
            )
        elif run_name and run_name != self.run_name:
            self.run = self.krepo.start_run(
                run_name=run_name,
            )