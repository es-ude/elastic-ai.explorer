import json
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import shutil
import tarfile
from elasticai.explorer.generator.deployment.compiler import Compiler
from elasticai.explorer.generator.deployment.device_communication import (
    ENv5Host,
    Host,
    PicoHost,
    RPiHost,
)
from elasticai.explorer.generator.model_compiler import tflite_to_resolver
from elasticai.explorer.hw_nas.search_space.quantization import (
    FixedPointInt8Scheme,
    QuantizationScheme,
    parse_bytearray_to_fxp_tensor,
)
from elasticai.explorer.training.data import DatasetSpecification
from elasticai.explorer.hw_nas.search_space.quantization import (
    parse_fxp_tensor_to_bytearray,
)
from settings import DOCKER_CONTEXT_DIR
from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)
from torch.utils.data import DataLoader, random_split
from torch import nn


class Metric(Enum):
    LATENCY = "Latency"
    ACCURACY = "Accuracy"
    VERIFICATION = "Verification"


class HWManager(ABC):
    def __init__(self, target: Host, compiler: Compiler):
        self._metric_to_source: dict[Metric, Path] = {}

    def _register_metric_to_source(self, metric: Metric, source: Path):
        self._metric_to_source.update({metric: source})

    @abstractmethod
    def install_code_on_target(self, source: Path, metric: Metric):
        pass

    @abstractmethod
    def prepare_dataset(
        self,
        dataset_spec: DatasetSpecification,
        quantization_scheme: QuantizationScheme,
    ):
        pass

    @abstractmethod
    def deploy_model(self, path_to_model: Path):
        pass

    @abstractmethod
    def measure_metric(
        self, metric: Metric, path_to_model: Path, model: nn.Module
    ) -> dict:
        pass


class RPiHWManager(HWManager):

    def __init__(self, target: RPiHost, compiler: Compiler):
        self.compiler = compiler
        self.target = target
        self.logger = logging.getLogger(
            "explorer.generator.deployment.manager.RPIHWManager"
        )
        self.logger.info("Initializing PI Hardware Manager...")
        super().__init__(target, compiler)

    def install_code_on_target(self, source: Path, metric: Metric):
        if source.is_relative_to(DOCKER_CONTEXT_DIR):
            relative_path = Path("/" + str(source.relative_to(DOCKER_CONTEXT_DIR)))
        else:
            relative_path = Path("/" + str(source))
        path_to_executable = self.compiler.compile_code(relative_path)
        self._register_metric_to_source(metric, relative_path)
        self.target.put_file(path_to_executable, ".")

    def prepare_dataset(
        self,
        dataset_spec: DatasetSpecification,
        quantization_scheme: QuantizationScheme,
    ):

        if dataset_spec.deployable_dataset_path:
            dataset_dir = dataset_spec.deployable_dataset_path
        else:
            dataset_dir = dataset_spec.dataset_location
        archive_name = dataset_dir.with_suffix(".tar.gz")
        with tarfile.open(archive_name, "w:gz") as tar:
            tar.add(dataset_dir, arcname=dataset_dir.name)

        self.target.put_file(archive_name, ".")
        self.target.run_command(f"tar -xzf {archive_name.name} -C data")

    def measure_metric(
        self, metric: Metric, path_to_model: Path, model: nn.Module
    ) -> dict:
        source = self._metric_to_source.get(metric)
        if not source:
            raise Exception(f"No source code registered for Metric: {metric}")
        _, tail = os.path.split(path_to_model)
        self.logger.info("Measure {} of model on device.".format(metric))
        cmd = None

        match metric:
            case metric.ACCURACY:
                cmd = self.build_command(source.stem, [tail])
                print("acc")
            case metric.LATENCY:
                cmd = self.build_command(source.stem, [tail])
                print("lat")

        measurement = self.target.run_command(cmd)
        measurement = self._parse_measurement(measurement)

        self.logger.debug("Measurement on device: %s ", measurement)
        return measurement

    def deploy_model(self, path_to_model: Path):
        self.logger.info("Put model %s on target", path_to_model)
        self.target.put_file(path_to_model, ".")

    def _parse_measurement(self, result: str) -> dict:
        return json.loads(result)

    def build_command(self, name_of_executable: str, arguments: list[str]):
        builder = CommandBuilder(name_of_executable)
        for argument in arguments:
            builder.add_argument(argument)
        command = builder.build()
        return command


class CommandBuilder:
    def __init__(self, name_of_exec: str):
        self.command: list[str] = ["./{}".format(name_of_exec)]

    def add_argument(self, arg):
        self.command.append(arg)

    def build(self) -> str:
        return " ".join(self.command)


class PicoHWManager(HWManager):

    def __init__(self, target: PicoHost, compiler: Compiler):
        self.compiler = compiler
        self.target = target
        self.logger = logging.getLogger(
            "explorer.generator.deployment.manager.PicoHWManager"
        )
        self.logger.info("Initializing Pico Hardware Manager...")
        super().__init__(target, compiler)

    def install_code_on_target(self, source: Path, metric: Metric):
        if source.is_relative_to(DOCKER_CONTEXT_DIR):
            relative_path = Path("/" + str(source.relative_to(DOCKER_CONTEXT_DIR)))
        else:
            relative_path = Path("/" + str(source))
        self._register_metric_to_source(metric, relative_path)

    def prepare_dataset(
        self,
        dataset_spec: DatasetSpecification,
        quantization_scheme: QuantizationScheme,
    ):
        target_dir = DOCKER_CONTEXT_DIR / "code/pico_crosscompiler/data"
        if not dataset_spec.deployable_dataset_path:
            raise ValueError(
                "For deployment on Pico the DatasetSpecification must have deployable_dataset_path set."
            )
        for file in dataset_spec.deployable_dataset_path.iterdir():
            if file.is_file():
                shutil.copyfile(file, target_dir / file.name)

    def measure_metric(
        self, metric: Metric, path_to_model: Path, model: nn.Module
    ) -> dict:

        self.deploy_model(path_to_model)
        source = self._metric_to_source.get(metric)
        if not source:
            self.logger.error(f"No source code registered for Metric: {metric}")
            exit(-1)
        path_to_resolver = Path(str(DOCKER_CONTEXT_DIR) + f"{source}/resolver_ops.h")
        tflite_to_resolver.generate_resolver_h(
            path_to_model,
            path_to_resolver,
        )

        path_to_executable = self.compiler.compile_code(source)
        self.measurements = self.target.put_file(path_to_executable, None)
        if self.measurements:
            measurement = self._parse_measurement(self.measurements)
        else:
            return self._parse_measurement(
                '{"' + metric.value + '": { "value": -1, "unit": "Error"}}'
            )

        self.logger.debug("Measurement on device: %s ", measurement)
        return measurement

    def deploy_model(self, path_to_model: Path):
        shutil.copyfile(
            path_to_model.parent / (path_to_model.stem + ".cpp"),
            DOCKER_CONTEXT_DIR / "code/pico_crosscompiler/data/model.cpp",
        )

    def _parse_measurement(self, result: str) -> dict:
        return json.loads(result)


class ENv5HWManager(HWManager):
    def __init__(self, target: ENv5Host, compiler: Compiler):
        self.compiler = compiler
        self.target = target
        self.logger = logging.getLogger(
            "explorer.generator.deployment.manager.RPIHWManager"
        )
        self.logger.info("Initializing PI Hardware Manager...")
        super().__init__(target, compiler)

    def install_code_on_target(self, source: Path, metric: Metric):

        self._register_metric_to_source(metric, source)

    def prepare_dataset(
        self,
        dataset_spec: DatasetSpecification,
        quantization_scheme: QuantizationScheme,
    ):

        if not isinstance(quantization_scheme, FixedPointInt8Scheme):
            err = TypeError(f"{quantization_scheme} is not supported by ENv5HWManager!")
            self.logger.error(err)
            raise err

        self.num_output_bytes = quantization_scheme.num_output_bytes
        self.frac_bits = quantization_scheme.frac_bits
        self.total_bits = quantization_scheme.total_bits
        self.dataset_spec = dataset_spec
        fxp_params = FxpParams(
            total_bits=quantization_scheme.total_bits,
            frac_bits=quantization_scheme.frac_bits,
            signed=True,
        )
        fxp_conf = FxpArithmetic(fxp_params)
        self.dataset = self.dataset_spec.dataset_type(
            dataset_spec.dataset_location,
            dataset_spec.transform,
            target_transform=lambda x: fxp_conf.as_rational(fxp_conf.cut_as_integer(x)),
        )
        train_subset, test_subset, val_subset = random_split(
            self.dataset,
            dataset_spec.test_train_val_ratio,
        )
        self.batch_size = 16
        self.test_loader = DataLoader(
            test_subset, batch_size=self.batch_size, shuffle=dataset_spec.shuffle
        )

    def measure_metric(
        self, metric: Metric, path_to_model: Path, model: nn.Module
    ) -> dict:
        self.deploy_model(path_to_model)
        source = self._metric_to_source.get(metric)

        if not source:
            self.logger.error(f"No source code registered for Metric: {metric}")
            exit(-1)

        path_to_executable = self.compiler.compile_code(source)
        self.measurements = self.target.put_file(path_to_executable, None)

        accuracy = self._run_accuracy_test()
        return {"Accuracy on device": accuracy}

    def deploy_model(self, path_to_model: Path):
        self.target.put_file(local_path=path_to_model, remote_path=None)

    def _run_accuracy_test(self) -> float:

        correct = 0
        total = 0
        for inputs_rational, target in self.test_loader:
            data_bytearray = parse_fxp_tensor_to_bytearray(
                inputs_rational, self.total_bits, self.frac_bits
            )
            batch_results_bytes = []
            for sample in data_bytearray:
                result_bytes = self.target.send_data_bytes(
                    sample=sample, num_bytes_outputs=self.num_output_bytes
                )
                batch_results_bytes.append(result_bytes)
            result = parse_bytearray_to_fxp_tensor(
                batch_results_bytes,
                self.total_bits,
                self.frac_bits,
                (self.batch_size, 1, self.num_output_bytes),
            )
            pred = result.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        return 100.0 * correct / total
