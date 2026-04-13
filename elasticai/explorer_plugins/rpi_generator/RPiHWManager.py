from elasticai.explorer.generator.deployment.compiler import Compiler
from elasticai.explorer.generator.deployment.device_communication import SSHHost
from elasticai.explorer.generator.deployment.hw_manager import (
    CommandBuilder,
    HWManager,
    Metric,
    MetricFunction,
)
from elasticai.explorer.hw_nas.search_space.quantization import QuantizationScheme
from elasticai.explorer.training.data import DatasetSpecification
from elasticai.explorer_plugins.rpi_generator.RPiHost import RPiHost


import json
import logging
import os
import tarfile
from pathlib import Path
from typing import Callable, Dict


class RPiHWManager(HWManager):
    def __init__(self, target: RPiHost, compiler: Compiler):
        self.compiler = compiler
        self.docker_build_context = self.compiler.compiler_params.build_context
        self.target = target
        self.logger = logging.getLogger(
            "explorer.generator.deployment.hw_manager.RPiHWManager"
        )
        self.logger.info("Initializing PI Hardware Manager...")
        super().__init__(target, compiler)

    def prepare_measurement(self, source: Path | MetricFunction, metric: Metric):
        if isinstance(source, Callable):
            super().prepare_measurement(source, metric)
            return

        relative_path = self._create_relative_path(self.docker_build_context, source)
        path_to_executable = self.compiler.compile_code(relative_path)
        self._register_metric_to_source(metric, relative_path)
        self.target.put_file(path_to_executable, ".")

    def prepare_dataset(
        self,
        dataset_spec: DatasetSpecification,
        quantization_scheme: QuantizationScheme | None,
    ):
        super().prepare_dataset(dataset_spec, quantization_scheme)
        if dataset_spec.deployable_dataset_path:
            dataset_dir = dataset_spec.deployable_dataset_path
        else:
            raise Exception(
                f"There is no deployable dataset path. Cannot prepare the dataset."
            )
        archive_name = dataset_dir.with_suffix(".tar.gz")
        with tarfile.open(archive_name, "w:gz") as tar:
            tar.add(dataset_dir, arcname=dataset_dir.name)

        self.target.put_file(archive_name, ".")
        self.target.run_command(f"tar -xzf {archive_name.name} -C data")

    def prepare_model(self, path_to_model: Path):
        self.logger.info("Put model %s on target", path_to_model)
        self.target.put_file(path_to_model, ".")

    def measure_metric(self, metric: Metric, path_to_model: Path) -> dict:
        source = self._metric_to_source.get(metric)
        if not source:
            raise Exception(f"No source code registered for Metric: {metric}")
        _, tail = os.path.split(path_to_model)
        self.logger.info("Measure {} of model on device.".format(metric))

        measurement = self._invoke_metric_source(metric, path_to_model)

        self.logger.debug("Measurement on device: %s ", measurement)
        return measurement

    def build_command(self, name_of_executable: str, arguments: list[str]):
        builder = CommandBuilder(name_of_executable)
        for argument in arguments:
            builder.add_argument(argument)
        command = builder.build()
        return command

    def _invoke_metric_source(self, metric: Metric, path_to_model: Path) -> Dict:
        results = super()._invoke_metric_source(metric, path_to_model)
        if results:
            return results
        source = self._get_metric_source(metric)
        out: str | None = None
        if isinstance(source, Path):
            src_path: Path = source
            out: None | str = None
            if self.compiler is not None:
                compiled = self.compiler.compile_code(src_path, src_path.parent)
            else:
                compiled = src_path

            if isinstance(self.target, SSHHost):
                if compiled:
                    self.target.put_file(local_path=compiled, remote_path=".")
                    cmd = f"./{Path(compiled).name} {path_to_model.name}"
                    out = self.target.run_command(cmd)
            else:
                err = TypeError(
                    f"Only SSHHost is supported and not {self.target.__class__.__name__}"
                )
                self.logger.error(err)
                raise err

            if out:
                return json.loads(out)
            else:
                return {metric.value: {"value": -1, "unit": "Error"}}

        err = TypeError(f"Unsupported source for metric {metric}. ")
        self.logger.error(err)
        raise err
