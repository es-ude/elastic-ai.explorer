from pathlib import Path
from unittest.mock import MagicMock, Mock

from elasticai.explorer.generator.deployment.compiler import RPICompiler
from elasticai.explorer.generator.deployment.device_communication import SSHHost
from elasticai.explorer.generator.deployment.hw_manager import (
    CommandBuilder,
    RPiHWManager,
    Metric,
)
from settings import DOCKER_CONTEXT_DIR


class TestPiHWManager:

    def testCommandBuilder(self):
        builder = CommandBuilder("measure_latency")
        builder.add_argument("model_0.pt")
        builder.add_argument("dataset")
        command = builder.build()
        assert "./measure_latency model_0.pt dataset" == command

    def test_run_latency_measurements(self):
        target = MagicMock(spec=SSHHost)
        compiler = Mock(spec=RPICompiler)
        output = '{ "Latency": { "value": 57474 , "unit": "microseconds"}}'

        expected = {"Latency": {"value": 57474, "unit": "microseconds"}}

        attr = {"run_command.return_value": output}
        target.configure_mock(**attr)
        attr = {
            "compile_code.return_value": ".",
        }
        compiler.configure_mock(**attr)
        compiler.compiler_params = MagicMock()
        compiler.compiler_params.base_dockerfile_path = ""

        self.hw_manager = RPiHWManager(target, compiler)
        path: Path = Path(str(DOCKER_CONTEXT_DIR)) / "bin" / "measure_latency"
        self.hw_manager._register_metric_to_source(
            Metric.LATENCY, Path("measure_latency")
        )

        metric = Metric.LATENCY
        result = self.hw_manager.measure_metric(metric, path_to_model=path)
        assert expected == result

    def test_run_accuracy_measurements(self):
        target = MagicMock(spec=SSHHost)
        compiler = Mock()
        output = '{ "Accuracy": { "value": 94.8 , "unit": "percent"}}'

        expected = {"Accuracy": {"value": 94.8, "unit": "percent"}}
        target_attr = {"run_command.return_value": output}
        target.configure_mock(**target_attr)
        compiler_attr = {"compile_code.return_value": "."}
        compiler.configure_mock(**compiler_attr)
        self.hw_manager = RPiHWManager(target, compiler)
        self.hw_manager._register_metric_to_source(
            Metric.ACCURACY, Path("measure_accuracy")
        )
        path: Path = Path(str(DOCKER_CONTEXT_DIR)) / "bin" / "measure_accuracy"
        metric = Metric.ACCURACY
        result = self.hw_manager.measure_metric(metric, path_to_model=path)
        assert expected == result
