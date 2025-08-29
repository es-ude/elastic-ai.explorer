from pathlib import Path
from unittest.mock import MagicMock, Mock

from elasticai.explorer.platforms.deployment.manager import (
    CommandBuilder,
    PIHWManager,
    DOCKER_CONTEXT_DIR,
    Metric,
)


class TestPiHWManager:

    def testCommandBuilder(self):
        builder = CommandBuilder("measure_latency")
        builder.add_argument("model_0.pt")
        builder.add_argument("dataset")
        command = builder.build()
        assert "./measure_latency model_0.pt dataset" == command

    def test_run_latency_measurements(self):
        target = MagicMock()
        compiler = Mock()
        output = '{ "Latency": { "value": 57474 , "unit": "microseconds"}}'

        expected = {"Latency": {"value": 57474, "unit": "microseconds"}}
        attr = {"run_command.return_value": output}
        target.configure_mock(**attr)
        self.hwmanager = PIHWManager(target, compiler)
        self.hwmanager._register_metric_to_source(Metric.LATENCY, "measure_latency")
        path: Path = Path(str(DOCKER_CONTEXT_DIR)) / "bin" / "measure_latency"
        metric = Metric.LATENCY
        result = self.hwmanager.measure_metric(metric, path_to_model=path)
        assert expected == result

    def test_run_accuracy_measurements(self):
        target = MagicMock()
        compiler = Mock()
        output = '{ "Accuracy": { "value": 94.8 , "unit": "percent"}}'

        expected = {"Accuracy": {"value": 94.8, "unit": "percent"}}
        attr = {"run_command.return_value": output}
        target.configure_mock(**attr)
        self.hwmanager = PIHWManager(target, compiler)
        self.hwmanager._register_metric_to_source(Metric.ACCURACY, "measure_accuracy")
        path: Path = Path(str(DOCKER_CONTEXT_DIR)) / "bin" / "measure_accuracy"
        metric = Metric.ACCURACY
        result = self.hwmanager.measure_metric(metric, path_to_model=path)
        assert expected == result
