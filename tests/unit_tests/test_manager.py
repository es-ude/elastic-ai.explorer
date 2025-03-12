import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock

from elasticai.explorer.platforms.deployment.manager import CommandBuilder, PIHWManager, CONTEXT_PATH


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
        expected = "Latency in us: 574747"
        attr = {"run_command.return_value": expected}
        target.configure_mock(**attr)
        self.hwmanager = PIHWManager(target, compiler)
        path: Path = Path(str(CONTEXT_PATH)) / "bin" / "measure_latency"

        result = self.hwmanager.measure_latency(path_to_model=path)
        self.assertEqual(("Latency in us", "574747"), result)

    def test_run_accuracy_measurements(self):
        target = MagicMock()
        compiler = Mock()
        expected = "Accuracy: 94.5"
        attr = {"run_command.return_value": expected}
        target.configure_mock(**attr)
        self.hwmanager = PIHWManager(target, compiler)
        path: Path = Path(str(CONTEXT_PATH)) / "bin" / "measure_accuracy"

        result = self.hwmanager.measure_accuracy(path_to_model=path, path_to_data="/data")
        self.assertEqual(("Accuracy", "94.5"), result)


if __name__ == '__main__':
    unittest.main()
