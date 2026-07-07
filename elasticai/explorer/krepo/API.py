import mlflow
from mlflow.entities import Experiment, Run
from typing import TYPE_CHECKING, Any
import subprocess
import json

if TYPE_CHECKING:
    from torch.nn import Module as TorchModule
else:
    TorchModule = Any


class KnowledgeRepoAPI:
    def __init__(self, server_ip: str, server_port: int):
        self.tracking_uri = f"http://localhost:{server_port}"
        self.server_ip = server_ip
        self.server_port = server_port

        self.experiment_id = None
        self.active_run = None
        self.nas_config = None
        self.search_space_config = None
        self.hw_platform = None

        self._ssh_process = None
        self._forward_port()
        mlflow.set_tracking_uri(self.tracking_uri)

    # --- Connection ---

    def _forward_port(self):
        if self._ssh_process:
            return

        self._ssh_process = subprocess.Popen(
            [
                "ssh", "-N",
                "-L", f"{self.server_port}:localhost:{self.server_port}",
                f"krepo@{self.server_ip}",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # Experiments, Runs

    def set_experiment(
            self,
            experiment_name: str | None = None,
            nas_config: dict | None = None,
            search_space_config: dict | None = None,
            hw_platform: str | None = None
    ) -> Experiment:
        """Create or retrieve the experiment and set it as the active context."""
        experiment = mlflow.set_experiment(experiment_name)
        if experiment:
            self.experiment_id = experiment.experiment_id
            self.nas_config = nas_config if nas_config else {}
            self.search_space_config = search_space_config if search_space_config else {}
            self.hw_platform = hw_platform if hw_platform else ""
            return experiment
        else:
            raise RuntimeError(f"KnowledgeRepoAPI: Could not find or create experiment: {experiment_name}")

    def start_run(self, run_name: str | None = None, run_id: str | None = None) -> Run:
        if not self.experiment_id:
            raise RuntimeError("KnowledgeRepoAPI: Experiment not set. Call set_experiment() first.")

        self.end_run()
        self.active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            run_id=run_id
        )
        mlflow.log_dict(self.nas_config, "nas_config.json")
        mlflow.log_dict(self.search_space_config, "search_space_config.json")
        self.set_tag("hw_platform", self.hw_platform)
        return self.active_run

    def end_run(self) -> None:
        if self.active_run:
            mlflow.end_run()
            self.active_run = None

    def set_tag(self, key: str, value: Any) -> None:
        self._require_active_run()
        mlflow.set_tag(key, value)

    def set_tags(self, tags: dict[str, Any]) -> None:
        self._require_active_run()
        for key, value in tags.items():
            mlflow.set_tag(key, value)

    def log_metrics(self, metrics: dict, model_id: str | None = None, step: int = 0) -> None:
        self._require_active_run()
        mlflow.log_metrics(metrics, model_id=model_id, step=step)

    def _require_active_run(self) -> None:
        if not self.active_run:
            raise RuntimeError("KnowledgeRepoAPI: Run not started. Use start_run() first.")

    # Model Logging

    def log_model(self, parameters: dict, model_architecture: dict, model: TorchModule | None = None) -> str | None:
        """
        Log model parameters and architecture.
        Optionally log a PyTorch model object.

        Returns:
            Model UID if a model object was logged, None otherwise.
        """
        self._require_active_run()

        mlflow.log_params(parameters)
        mlflow.log_dict(model_architecture, "model_architecture.json")
        mlflow.set_tag("model_type", "default")
        mlflow.set_tag("model_architecture", json.dumps(model_architecture))

        if model:
            model_info = mlflow.pytorch.log_model(model)
            return model_info.model_id

        return None

    def load_model_by_uid(self, uid: str) -> TorchModule:
        """Load and return a PyTorch model by its UID."""
        for logged_model in self.get_all_models():
            if logged_model.model_id == uid:
                return mlflow.pytorch.load_model(logged_model.artifact_location)
        raise LookupError(f"No model found matching UID: {uid}")

    @staticmethod
    def get_all_models():
        all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
        return mlflow.search_logged_models(
            experiment_ids=all_experiments,
            output_format="list"
        )

    # Estimator

    @staticmethod
    def get_training_data_for_estimator(
        target_metric: str, hw_platform: str, tags: dict[str, Any] | None = None
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """
        Returns all model architectures and target metrics corresponding to
        the given hardware platform, in the form of:
        [(model_architecture_dict, {metric: value}), ...]
        """
        filter_parts = [
            f'tags."hw_platform" = "{hw_platform}"',
            'tags."model_type" != "estimator"',
        ]
        if tags:
            for key, value in tags.items():
                filter_parts.append(f'tags."{key}" = "{value}"')

        runs = mlflow.search_runs(
            search_all_experiments=True,
            filter_string=" AND ".join(filter_parts),
        )

        training_data: list[tuple[dict, dict]] = []
        for _, run in runs.iterrows():
            metric_value = run[f"metrics.{target_metric}"]
            architecture_json = run.get("tags.model_architecture")
            if metric_value is None or architecture_json is None:
                continue
            training_data.append((json.loads(architecture_json), {target_metric: metric_value}))

        return training_data

    def save_estimator(
        self, estimator: TorchModule, hw_platform: str, metric: str,
        validation_loss: float, tags: dict[str, Any] | None = None
    ) -> str:
        """
        Save an estimator to a new run and return its UID.
        The UID can later be used with load_model_by_uid().
        """
        if self.active_run:
            self.end_run()

        self.start_run("Estimator")
        self.set_tags({
            "model_type": "estimator",
            "metric": metric,
            "hw_platform": hw_platform,
        })
        self.log_metrics({"validation_loss": validation_loss})

        if tags:
            self.set_tags(tags)

        model_info = mlflow.pytorch.log_model(pytorch_model=estimator, name="estimator_model")
        return model_info.model_id

    def load_estimator(self, hw_platform: str, metric: str, tags: dict[str, Any] | None = None) -> TorchModule:
        """
            Retrieve and load the best-performing estimator for the given hardware platform and metric.
        """
        if not self.experiment_id:
            raise RuntimeError("KnowledgeRepoAPI: Experiment not set. Call set_experiment() first.")

        run_id = self._find_best_estimator_run_id(hw_platform, metric, tags)
        model_uri = self._find_model_uri_for_run(run_id)
        return mlflow.pytorch.load_model(model_uri)

    def _find_best_estimator_run_id(
        self, hw_platform: str, metric: str, tags: dict[str, Any] | None
    ) -> str:
        filter_parts = [
            f"tags.hw_platform = '{hw_platform}'",
            "tags.model_type = 'estimator'",
            f"tags.metric = '{metric}'",
        ]
        if tags:
            for key, value in tags.items():
                filter_parts.append(f"tags.{key} = '{value}'")

        runs = mlflow.search_runs(
            search_all_experiments=True,
            filter_string=" AND ".join(filter_parts),
            order_by=["metrics.validation_loss ASC"],
            max_results=1,
        )

        all_models = self.get_all_models()
        if not all_models or runs.empty:
            raise LookupError(
                f"No estimator found matching criteria: HW='{hw_platform}', Metric='{metric}', Tags='{tags}'"
            )

        return runs.iloc[0]["run_id"]

    def _find_model_uri_for_run(self, run_id: str) -> str:
        for model in self.get_all_models():
            if model.source_run_id == run_id:
                return model.artifact_location
        raise LookupError(f"No model found for run_id: {run_id}")