from elasticai.explorer.generator.deployment.compiler import Compiler, CompilerParams
from elasticai.explorer.generator.deployment.device_communication import SerialHost
from elasticai.explorer.generator.deployment.hw_manager import (
    HWManager,
    Metric,
    MetricFunction,
)
from elasticai.explorer.hw_nas.search_space.quantization import QuantizationScheme
from elasticai.explorer.training.data import DatasetSpecification
from elasticai.explorer_impl.pico_generator import tflite_to_resolver
from elasticai.explorer_impl.pico_generator.host import PicoHost


import json
import logging
import shutil
from pathlib import Path
from typing import Dict


class PicoHWManager(HWManager):

    def __init__(self, target: PicoHost, compiler: Compiler):
        self.compiler = compiler
        self.target = target
        self.logger = logging.getLogger(
            "explorer.generator.deployment.hw_manager.PicoHWManager"
        )
        self.logger.info("Initializing Pico Hardware Manager...")
        if type(self.compiler.compiler_params) != CompilerParams: 
            err = ValueError(f"Only CompilerParams are supported with this HWManager and not {type(self.compiler.compiler_params)}.")
            self.logger.error(err)
            raise err
        self.docker_build_context = self.compiler.compiler_params.build_context
        super().__init__(target, compiler)

    def prepare_measurement(self, source: Path | MetricFunction, metric: Metric):

        relative_path = self._create_relative_path(self.docker_build_context, source)
        super().prepare_measurement(relative_path, metric)

    def prepare_dataset(
        self,
        dataset_spec: DatasetSpecification,
        quantization_scheme: QuantizationScheme | None,
    ):
        super().prepare_dataset(dataset_spec, quantization_scheme)
        target_dir = self.docker_build_context / "code/pico_crosscompiler/data"
        if not dataset_spec.deployable_dataset_path:
            raise ValueError(
                "For deployment on Pico the DatasetSpecification must have deployable_dataset_path set."
            )
        for file in dataset_spec.deployable_dataset_path.iterdir():
            if file.is_file():
                shutil.copyfile(file, target_dir / file.name)

    def _invoke_metric_source(self, metric: Metric, path_to_model: Path) -> Dict:
        results = super()._invoke_metric_source(metric, path_to_model)
        if results:
            return results

        source = self._metric_to_source.get(metric)
        if not source:
            raise Exception(f"No source code registered for Metric: {metric}")

        path_to_resolver = Path(
            str(self.docker_build_context) + f"{source}/resolver_ops.h"
        )
        tflite_to_resolver.generate_resolver_h(
            path_to_model,
            path_to_resolver,
        )
        if isinstance(source, Path):
            out: None | str = None
            if isinstance(self.target, SerialHost):
                path_to_executable = self.compiler.compile_code(source)
                if path_to_executable:
                    self.target.flash(local_path=path_to_executable)
                    out = self.target.receive()
            else:
                err = TypeError(
                    f"Only SerialHost is supported and not {self.target.__class__.__name__}"
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

    def prepare_model(self, path_to_model: Path):
        shutil.copyfile(
            path_to_model.parent / (path_to_model.stem + ".cpp"),
            self.docker_build_context / "code/pico_crosscompiler/data/model.cpp",
        )
