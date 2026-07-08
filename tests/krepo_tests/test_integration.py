import json

import mlflow
import pytest
import torch
import torch.nn as nn

from elasticai.explorer.krepo.API import KnowledgeRepoAPI


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def api(tmp_path):
    """KnowledgeRepoAPI backed by a local file-based MLflow store."""
    instance = KnowledgeRepoAPI("127.0.0.1", 59999)
    # Redirect MLflow to a temp directory instead of the SSH tunnel
    tracking_uri = str(tmp_path / "mlruns")
    instance.tracking_uri = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    yield instance
    instance.end_run()
    if instance._ssh_process and instance._ssh_process.poll() is None:
        instance._ssh_process.terminate()


@pytest.fixture
def api_with_experiment(api):
    api.set_experiment(
        "integration-test",
        nas_config={"algo": "evo"},
        search_space_config={"layers": [1, 2]},
        hw_platform="gpu",
    )
    return api


@pytest.fixture
def api_with_run(api_with_experiment):
    api_with_experiment.start_run("test-run")
    return api_with_experiment


class TestExperimentSetup:
    def test_creates_experiment(self, api):
        api.set_experiment("new-experiment", {}, {}, "cpu")
        assert api.experiment_id is not None
        experiment = mlflow.get_experiment_by_name("new-experiment")
        assert experiment is not None
        assert experiment.experiment_id == api.experiment_id

    def test_reuses_existing_experiment(self, api):
        api.set_experiment("reuse-test", {}, {}, "cpu")
        first_id = api.experiment_id
        api.set_experiment("reuse-test", {"new": True}, {}, "gpu")
        assert api.experiment_id == first_id


class TestRunLifecycle:
    def test_start_and_end_run(self, api_with_experiment):
        api = api_with_experiment
        api.start_run("lifecycle-run")
        assert api.active_run is not None
        run_id = api.active_run.info.run_id
        api.end_run()
        assert api.active_run is None

        run = mlflow.get_run(run_id)
        assert run.info.status == "FINISHED"

    def test_start_run_ends_previous(self, api_with_experiment):
        api = api_with_experiment
        api.start_run("first")
        first_run_id = api.active_run.info.run_id

        api.start_run("second")
        assert api.active_run.info.run_id != first_run_id

        run = mlflow.get_run(first_run_id)
        assert run.info.status == "FINISHED"


class TestMetricsAndTags:
    def test_metrics_persisted(self, api_with_run):
        api = api_with_run
        api.log_metrics({"accuracy": 0.95, "loss": 0.05}, step=1)

        run = mlflow.get_run(api.active_run.info.run_id)
        assert run.data.metrics["accuracy"] == 0.95
        assert run.data.metrics["loss"] == 0.05

    def test_set_tag_persisted(self, api_with_run):
        api = api_with_run
        api.set_tag("custom_tag", "hello")

        run = mlflow.get_run(api.active_run.info.run_id)
        assert run.data.tags["custom_tag"] == "hello"

    def test_set_tags_persisted(self, api_with_run):
        api = api_with_run
        api.set_tags({"tag_a": "1", "tag_b": "2"})

        run = mlflow.get_run(api.active_run.info.run_id)
        assert run.data.tags["tag_a"] == "1"
        assert run.data.tags["tag_b"] == "2"


class TestLogModel:
    def test_without_pytorch_model(self, api_with_run):
        api = api_with_run
        arch = {"layers": 3, "hidden": 64}
        result = api.log_model(
            parameters={"lr": "0.001"},
            model_architecture=arch,
        )

        assert result is None
        run = mlflow.get_run(api.active_run.info.run_id)
        assert run.data.params["lr"] == "0.001"
        assert run.data.tags["model_architecture"] == json.dumps(arch)
        assert run.data.tags["model_type"] == "default"

    def test_with_pytorch_model_returns_uid(self, api_with_run):
        api = api_with_run
        uid = api.log_model(
            parameters={"lr": "0.01"},
            model_architecture={"type": "simple"},
            model=SimpleModel(),
        )

        assert uid is not None
        assert isinstance(uid, str)


class TestEstimatorRoundTrip:
    def test_save_and_load_estimator(self, api_with_experiment):
        api = api_with_experiment
        original = SimpleModel()

        uid = api.save_estimator(
            estimator=original,
            hw_platform="gpu",
            metric="latency",
            validation_loss=0.03,
        )
        assert uid is not None
        api.end_run()

        loaded = api.load_estimator(hw_platform="gpu", metric="latency")
        assert isinstance(loaded, nn.Module)
        assert torch.equal(original.linear.weight.data, loaded.linear.weight.data)
        assert torch.equal(original.linear.bias.data, loaded.linear.bias.data)

    def test_save_estimator_persists_tags(self, api_with_experiment):
        api = api_with_experiment
        api.save_estimator(
            estimator=SimpleModel(),
            hw_platform="cpu",
            metric="throughput",
            validation_loss=0.1,
            tags={"version": "2"},
        )
        run_id = api.active_run.info.run_id
        api.end_run()

        run = mlflow.get_run(run_id)
        assert run.data.tags["model_type"] == "estimator"
        assert run.data.tags["metric"] == "throughput"
        assert run.data.tags["version"] == "2"
        assert run.data.metrics["validation_loss"] == 0.1

    def test_load_estimator_picks_lowest_validation_loss(self, api_with_experiment):
        api = api_with_experiment
        worse = SimpleModel()
        better = SimpleModel()

        api.save_estimator(worse, "gpu", "latency", validation_loss=0.5)
        api.end_run()
        api.save_estimator(better, "gpu", "latency", validation_loss=0.01)
        api.end_run()

        loaded = api.load_estimator(hw_platform="gpu", metric="latency")
        assert torch.equal(better.linear.weight.data, loaded.linear.weight.data)

    def test_get_model_by_uid(self, api_with_experiment):
        api = api_with_experiment
        original = SimpleModel()
        uid = api.save_estimator(
            estimator=original,
            hw_platform="gpu",
            metric="latency",
            validation_loss=0.01,
        )
        api.end_run()

        loaded = api.load_model_by_uid(uid)
        assert isinstance(loaded, nn.Module)
        assert torch.equal(original.linear.weight.data, loaded.linear.weight.data)

    def test_load_model_by_uid_unknown_raises(self, api_with_experiment):
        with pytest.raises(LookupError, match="No model found matching UID"):
            api_with_experiment.load_model_by_uid("nonexistent-uid")


class TestGetTrainingData:
    def test_retrieves_logged_architectures(self, api_with_experiment):
        api = api_with_experiment

        api.start_run("run-1")
        api.log_model(
            parameters={"lr": "0.01"},
            model_architecture={"layers": 2, "hidden": 32},
        )
        api.log_metrics({"accuracy": 0.9})
        api.end_run()

        api.start_run("run-2")
        api.log_model(
            parameters={"lr": "0.001"},
            model_architecture={"layers": 4, "hidden": 64},
        )
        api.log_metrics({"accuracy": 0.95})
        api.end_run()

        data = KnowledgeRepoAPI.get_training_data_for_estimator("accuracy", "gpu")
        assert len(data) == 2

        architectures = [arch for arch, _ in data]
        metrics = [m["accuracy"] for _, m in data]
        assert {"layers": 2, "hidden": 32} in architectures
        assert {"layers": 4, "hidden": 64} in architectures
        assert 0.9 in metrics
        assert 0.95 in metrics

    def test_excludes_estimator_runs(self, api_with_experiment):
        api = api_with_experiment

        # Log a regular model
        api.start_run("regular")
        api.log_model(
            parameters={"lr": "0.01"},
            model_architecture={"layers": 1},
        )
        api.log_metrics({"accuracy": 0.8})
        api.end_run()

        # Log an estimator (tagged model_type=estimator)
        api.save_estimator(SimpleModel(), "gpu", "accuracy", 0.05)
        api.end_run()

        data = KnowledgeRepoAPI.get_training_data_for_estimator("accuracy", "gpu")
        assert len(data) == 1
        assert data[0][0] == {"layers": 1}

    def test_filters_by_hw_platform(self, api_with_experiment):
        api = api_with_experiment

        # Log model under "gpu" (the experiment's hw_platform)
        api.start_run("gpu-run")
        api.log_model(parameters={"lr": "0.01"}, model_architecture={"a": 1})
        api.log_metrics({"loss": 0.1})
        api.end_run()

        # Query for "cpu" should return nothing
        data = KnowledgeRepoAPI.get_training_data_for_estimator("loss", "cpu")
        assert data == []

        # Query for "gpu" should return the run
        data = KnowledgeRepoAPI.get_training_data_for_estimator("loss", "gpu")
        assert len(data) == 1
