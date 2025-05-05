import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock

from elasticai.explorer.platforms.deployment.manager import (
    CommandBuilder,
    PIHWManager,
    CONTEXT_PATH,
    Metric,
)


class TestPiHWManager(unittest.TestCase):

    def testCommandBuilder(self):
        builder = CommandBuilder("measure_latency")
        builder.add_argument("model_0.pt")
        builder.add_argument("dataset")
        command = builder.build()
        self.assertEqual("./measure_latency model_0.pt dataset", command)

    def test_run_latency_measurements(self):
        target = MagicMock()
        compiler = Mock()
        output = '{ "Latency": { "value": 57474 , "unit": "microseconds"}}'

        expected = {"Latency": {"value": 57474, "unit": "microseconds"}}
        attr = {"run_command.return_value": output}
        target.configure_mock(**attr)
        self.hwmanager = PIHWManager(target, compiler)
        path: Path = Path(str(CONTEXT_PATH)) / "bin" / "measure_latency"
        metric = Metric.LATENCY
        result = self.hwmanager.measure_metric(metric, path_to_model=path)
        self.assertEqual(expected, result)

    def test_run_accuracy_measurements(self):
        target = MagicMock()
        compiler = Mock()
        output = '{ "Accuracy": { "value": 94.8 , "unit": "percent"}}'

        expected = {"Accuracy": {"value": 94.8, "unit": "percent"}}
        attr = {"run_command.return_value": output}
        target.configure_mock(**attr)
        self.hwmanager = PIHWManager(target, compiler)
        path: Path = Path(str(CONTEXT_PATH)) / "bin" / "measure_accuracy"
        metric = Metric.ACCURACY
        result = self.hwmanager.measure_metric(metric, path_to_model=path)
        self.assertEqual(expected, result)


if __name__ == "__main__":
    unittest.main()
