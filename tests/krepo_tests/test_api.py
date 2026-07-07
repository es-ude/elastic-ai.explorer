import json
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from elasticai.explorer.krepo import KnowledgeRepoAPI

Mocks = namedtuple("Mocks", ["api", "mlflow", "subprocess"])


@pytest.fixture
def mocks():
    """Create a KnowledgeRepoAPI with subprocess and mlflow mocked out."""
    with patch("krepo.API.subprocess") as mock_sub, \
         patch("krepo.API.mlflow") as mock_mlflow:
        instance = KnowledgeRepoAPI("10.0.0.1", 5000)
        yield Mocks(api=instance, mlflow=mock_mlflow, subprocess=mock_sub)


@pytest.fixture
def mocks_with_experiment(mocks):
    """Mocks with an experiment already set."""
    mocks.mlflow.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="exp-1")
    mocks.api.set_experiment(
        "test-experiment",
        nas_config={"algo": "evo"},
        search_space_config={"layers": [1, 2]},
        hw_platform="gpu",
    )
    return mocks


@pytest.fixture
def mocks_with_run(mocks_with_experiment):
    """Mocks with an active run."""
    m = mocks_with_experiment
    m.mlflow.start_run.return_value = SimpleNamespace(
        info=SimpleNamespace(run_id="run-1")
    )
    m.api.start_run("test-run")
    return m


class TestInit:
    def test_tracking_uri(self, mocks):
        assert mocks.api.tracking_uri == "http://localhost:5000"
        mocks.mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")

    def test_forward_port_launches_ssh(self, mocks):
        mocks.subprocess.Popen.assert_called_once()
        args, kwargs = mocks.subprocess.Popen.call_args
        assert args[0] == ["ssh", "-N", "-L", "5000:localhost:5000", "krepo@10.0.0.1"]
        assert "stdin" in kwargs
        assert "stdout" in kwargs
        assert "stderr" in kwargs

    def test_initial_state_is_none(self, mocks):
        assert mocks.api.experiment_id is None
        assert mocks.api.active_run is None
        assert mocks.api.nas_config is None
        assert mocks.api.search_space_config is None
        assert mocks.api.hw_platform is None


class TestSetExperiment:
    def test_sets_experiment_fields(self, mocks_with_experiment):
        api = mocks_with_experiment.api
        assert api.experiment_id == "exp-1"
        assert api.nas_config == {"algo": "evo"}
        assert api.search_space_config == {"layers": [1, 2]}
        assert api.hw_platform == "gpu"

    def test_raises_when_experiment_not_found(self, mocks):
        mocks.mlflow.get_experiment_by_name.return_value = None
        with pytest.raises(RuntimeError, match="Could not find or create experiment"):
            mocks.api.set_experiment("missing", {}, {}, "cpu")


class TestRunLifecycle:
    def test_start_run_without_experiment_raises(self, mocks):
        with pytest.raises(RuntimeError, match="Experiment not set"):
            mocks.api.start_run("r1")

    def test_start_run_logs_configs(self, mocks_with_run):
        mlf = mocks_with_run.mlflow
        mlf.log_dict.assert_any_call({"algo": "evo"}, "nas_config.json")
        mlf.log_dict.assert_any_call({"layers": [1, 2]}, "search_space_config.json")

    def test_start_run_sets_hw_platform_tag(self, mocks_with_run):
        mocks_with_run.mlflow.set_tag.assert_any_call("hw_platform", "gpu")

    def test_end_run_clears_current_run(self, mocks_with_run):
        api = mocks_with_run.api
        assert api.active_run is not None
        api.end_run()
        assert api.active_run is None
        mocks_with_run.mlflow.end_run.assert_called()

    def test_end_run_noop_when_no_run(self, mocks):
        mocks.api.end_run()  # should not raise
        mocks.mlflow.end_run.assert_not_called()

    def test_start_run_ends_previous_run(self, mocks_with_experiment):
        m = mocks_with_experiment
        m.mlflow.start_run.return_value = SimpleNamespace(info=SimpleNamespace(run_id="r1"))
        m.api.start_run("first")

        m.mlflow.start_run.return_value = SimpleNamespace(info=SimpleNamespace(run_id="r2"))
        m.api.start_run("second")

        m.mlflow.end_run.assert_called()


class TestTags:
    def test_set_tag_without_run_raises(self, mocks):
        with pytest.raises(RuntimeError, match="Run not started"):
            mocks.api.set_tag("k", "v")

    def test_set_tag_delegates_to_mlflow(self, mocks_with_run):
        mocks_with_run.mlflow.set_tag.reset_mock()
        mocks_with_run.api.set_tag("foo", "bar")
        mocks_with_run.mlflow.set_tag.assert_called_with("foo", "bar")

    def test_set_tags_without_run_raises(self, mocks):
        with pytest.raises(RuntimeError, match="Run not started"):
            mocks.api.set_tags({"a": "1"})

    def test_set_tags_delegates_each_pair(self, mocks_with_run):
        mocks_with_run.mlflow.set_tag.reset_mock()
        mocks_with_run.api.set_tags({"a": "1", "b": "2"})
        mocks_with_run.mlflow.set_tag.assert_any_call("a", "1")
        mocks_with_run.mlflow.set_tag.assert_any_call("b", "2")


class TestLogMetrics:
    def test_without_run_raises(self, mocks):
        with pytest.raises(RuntimeError, match="Run not started"):
            mocks.api.log_metrics({"loss": 0.5})

    def test_delegates_to_mlflow(self, mocks_with_run):
        mocks_with_run.api.log_metrics({"loss": 0.5}, step=3)
        mocks_with_run.mlflow.log_metrics.assert_called_with({"loss": 0.5}, step=3)


class TestLogModel:
    def test_without_run_raises(self, mocks):
        with pytest.raises(RuntimeError, match="Run not started"):
            mocks.api.log_model({}, {})

    def test_logs_params_and_architecture(self, mocks_with_run):
        m = mocks_with_run
        arch = {"layers": 3, "hidden": 128}

        m.api.log_model({"lr": 0.01}, arch)

        m.mlflow.log_params.assert_called_with({"lr": 0.01})
        m.mlflow.log_dict.assert_any_call(arch, "model_architecture.json")
        m.mlflow.set_tag.assert_any_call("model_architecture", json.dumps(arch))
        m.mlflow.set_tag.assert_any_call("model_type", "default")

    def test_returns_none_without_model(self, mocks_with_run):
        result = mocks_with_run.api.log_model({"lr": 0.01}, {"layers": 1})
        assert result is None

    def test_returns_uid_with_model(self, mocks_with_run):
        m = mocks_with_run
        mock_model = MagicMock()
        m.mlflow.pytorch.log_model.return_value = SimpleNamespace(model_id="uid-42")

        result = m.api.log_model({"lr": 0.01}, {"layers": 1}, model=mock_model)

        assert result == "uid-42"
        m.mlflow.pytorch.log_model.assert_called_once_with(mock_model)


class TestGetTrainingData:
    @staticmethod
    def _make_runs_df(rows):
        return pd.DataFrame(rows)

    def test_returns_architecture_and_metric(self, mocks):
        arch = {"layers": 2, "hidden": 64}
        df = self._make_runs_df([{
            "run_id": "r1",
            "metrics.accuracy": 0.95,
            "tags.model_architecture": json.dumps(arch),
        }])
        mocks.mlflow.search_runs.return_value = df

        result = KnowledgeRepoAPI.get_training_data_for_estimator("accuracy", "gpu")
        assert len(result) == 1
        assert result[0] == (arch, {"accuracy": 0.95})

    def test_skips_rows_without_metric(self, mocks):
        df = self._make_runs_df([{
            "run_id": "r1",
            "metrics.accuracy": None,
            "tags.model_architecture": '{"a":1}',
        }])
        mocks.mlflow.search_runs.return_value = df

        result = KnowledgeRepoAPI.get_training_data_for_estimator("accuracy", "gpu")
        assert result == []

    def test_skips_rows_without_architecture_tag(self, mocks):
        df = self._make_runs_df([{
            "run_id": "r1",
            "metrics.accuracy": 0.9,
        }])
        mocks.mlflow.search_runs.return_value = df

        result = KnowledgeRepoAPI.get_training_data_for_estimator("accuracy", "gpu")
        assert result == []

    def test_multiple_runs(self, mocks):
        df = self._make_runs_df([
            {"run_id": "r1", "metrics.loss": 0.1, "tags.model_architecture": '{"a":1}'},
            {"run_id": "r2", "metrics.loss": 0.2, "tags.model_architecture": '{"b":2}'},
        ])
        mocks.mlflow.search_runs.return_value = df

        result = KnowledgeRepoAPI.get_training_data_for_estimator("loss", "cpu")
        assert len(result) == 2

    def test_filter_includes_custom_tags(self, mocks):
        df = self._make_runs_df([])
        mocks.mlflow.search_runs.return_value = df

        KnowledgeRepoAPI.get_training_data_for_estimator(
            "loss", "gpu", tags={"task": "classification"}
        )

        filter_string = mocks.mlflow.search_runs.call_args[1]["filter_string"]
        assert '"task" = "classification"' in filter_string


class TestSaveEstimator:
    def test_saves_and_returns_uid(self, mocks_with_experiment):
        m = mocks_with_experiment
        m.mlflow.start_run.return_value = SimpleNamespace(info=SimpleNamespace(run_id="run-e"))
        m.mlflow.pytorch.log_model.return_value = SimpleNamespace(model_id="uid-99")

        uid = m.api.save_estimator(MagicMock(), "gpu", "latency", 0.05)

        assert uid == "uid-99"
        m.mlflow.set_tag.assert_any_call("model_type", "estimator")
        m.mlflow.set_tag.assert_any_call("metric", "latency")
        m.mlflow.log_metrics.assert_called_with({"validation_loss": 0.05}, step=0)

    def test_ends_existing_run_before_saving(self, mocks_with_run):
        m = mocks_with_run
        m.mlflow.pytorch.log_model.return_value = SimpleNamespace(model_id="uid-1")

        m.api.save_estimator(MagicMock(), "gpu", "latency", 0.1)
        m.mlflow.end_run.assert_called()

    def test_sets_extra_tags(self, mocks_with_experiment):
        m = mocks_with_experiment
        m.mlflow.start_run.return_value = SimpleNamespace(info=SimpleNamespace(run_id="run-e"))
        m.mlflow.pytorch.log_model.return_value = SimpleNamespace(model_id="uid-1")

        m.api.save_estimator(MagicMock(), "gpu", "latency", 0.1, tags={"version": "2"})
        m.mlflow.set_tag.assert_any_call("version", "2")


class TestLoadEstimator:
    def test_raises_without_experiment(self, mocks):
        with pytest.raises(RuntimeError, match="Experiment not set"):
            mocks.api.load_estimator("gpu", "latency")

    def test_raises_when_no_runs(self, mocks_with_experiment):
        m = mocks_with_experiment
        m.mlflow.search_runs.return_value = pd.DataFrame()
        m.mlflow.search_experiments.return_value = []
        m.mlflow.search_logged_models.return_value = []

        with pytest.raises(LookupError, match="No estimator found"):
            m.api.load_estimator("gpu", "latency")

    def test_raises_when_no_model_matches_run(self, mocks_with_experiment):
        m = mocks_with_experiment
        m.mlflow.search_runs.return_value = pd.DataFrame([{"run_id": "run-x"}])
        m.mlflow.search_experiments.return_value = [SimpleNamespace(experiment_id="e1")]
        m.mlflow.search_logged_models.return_value = [
            SimpleNamespace(source_run_id="run-other", artifact_location="s3://other")
        ]

        with pytest.raises(LookupError, match="No model found"):
            m.api.load_estimator("gpu", "latency")

    def test_loads_matching_model(self, mocks_with_experiment):
        m = mocks_with_experiment
        m.mlflow.search_runs.return_value = pd.DataFrame([{"run_id": "run-x"}])
        m.mlflow.search_experiments.return_value = [SimpleNamespace(experiment_id="e1")]
        m.mlflow.search_logged_models.return_value = [
            SimpleNamespace(source_run_id="run-x", artifact_location="s3://model")
        ]
        sentinel = object()
        m.mlflow.pytorch.load_model.return_value = sentinel

        result = m.api.load_estimator("gpu", "latency")

        assert result is sentinel
        m.mlflow.pytorch.load_model.assert_called_once_with("s3://model")


class TestLoadModelByUid:
    def test_returns_model_for_matching_uid(self, mocks_with_experiment):
        m = mocks_with_experiment
        m.mlflow.search_experiments.return_value = [SimpleNamespace(experiment_id="e1")]
        m.mlflow.search_logged_models.return_value = [
            SimpleNamespace(model_id="uid-1", artifact_location="s3://m1"),
            SimpleNamespace(model_id="uid-2", artifact_location="s3://m2"),
        ]
        sentinel = object()
        m.mlflow.pytorch.load_model.return_value = sentinel

        result = m.api.load_model_by_uid("uid-2")

        assert result is sentinel
        m.mlflow.pytorch.load_model.assert_called_once_with("s3://m2")

    def test_raises_for_unknown_uid(self, mocks_with_experiment):
        m = mocks_with_experiment
        m.mlflow.search_experiments.return_value = [SimpleNamespace(experiment_id="e1")]
        m.mlflow.search_logged_models.return_value = []

        with pytest.raises(LookupError, match="No model found matching UID"):
            m.api.load_model_by_uid("nonexistent")
            