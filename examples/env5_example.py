import logging.config
from pathlib import Path
import numpy as np
import torch
from typing import Any, Callable

from elasticai.explorer.training.data import BaseDataset, DatasetSpecification
from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)


from elasticai.explorer.generator.model_builder.model_builder import CreatorModelBuilder
from elasticai.explorer.hw_nas.hw_nas import HWNASParameters, SearchStrategy
from elasticai.explorer.hw_nas.search_space.quantization import (
    FixedPointInt8Scheme,
    parse_bytearray_to_fxp_tensor,
    parse_fxp_tensor_to_bytearray,
)

from elasticai.explorer.explorer import Explorer
from elasticai.explorer.generator.deployment.compiler import VivadoParams
from elasticai.explorer.generator.deployment.device_communication import (
    Host,
    SerialHost,
    SerialParams,
)
from elasticai.explorer.generator.deployment.hw_manager import (
    HWManager,
    Metric,
)

from examples.example_helpers import (
    measure_on_device,
    setup_example_optimization_criteria,
    setup_knowledge_repository,
)

logging.config.fileConfig("logging.conf", disable_existing_loggers=False)

logger = logging.getLogger("explorer.main")


device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
BATCH_SIZE = 64
INPUT_DIM = 6


class NotAndDataset(BaseDataset):
    def __init__(
        self,
        root: str | Path,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(root, transform, target_transform, *args, **kwargs)

        self.data = torch.randint(0, 2, (BATCH_SIZE * 25, INPUT_DIM))
        summed = self.data.sum(dim=1)
        self.targets = torch.empty(BATCH_SIZE * 25, dtype=torch.long)
        self.targets[summed == 2] = 0
        self.targets[summed < 2] = 1

    def __getitem__(self, idx) -> Any:
        data, target = self.data[idx], self.targets[idx]
        if self.transform is not None:
            data = self.transform(self.data[idx])

        if self.target_transform is not None:
            target = self.target_transform(self.targets[idx])
        return data, target

    def __len__(self) -> int:
        return len(self.data)


class SumDataset(BaseDataset):
    def __init__(
        self,
        root,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        thresholds=[-0.5, 0.0, 0.5],
        noise_std=0.0,
        *args,
        **kwargs,
    ):
        super().__init__(root, transform, target_transform, *args, **kwargs)
        self.data = torch.randn(BATCH_SIZE * 100, INPUT_DIM)
        summed = self.data.sum(dim=1)
        if noise_std > 0:
            summed = summed + noise_std * torch.randn_like(summed)

        self.targets = torch.empty(BATCH_SIZE * 100, dtype=torch.long)
        self.targets[summed <= thresholds[0]] = 0
        self.targets[(summed > thresholds[0]) & (summed <= thresholds[1])] = 1
        self.targets[(summed > thresholds[1]) & (summed <= thresholds[2])] = 2
        self.targets[summed > thresholds[2]] = 3

    def __getitem__(self, idx) -> Any:
        data, target = self.data[idx], self.targets[idx]
        if self.transform is not None:
            data = self.transform(self.data[idx])

        if self.target_transform is not None:
            target = self.target_transform(self.targets[idx])
        return data, target

    def __len__(self) -> int:
        return len(self.data)


def create_example_dataset_spec(quantization_scheme):

    fxp_params = FxpParams(
        total_bits=quantization_scheme.total_bits,
        frac_bits=quantization_scheme.frac_bits,
        signed=quantization_scheme.signed,
    )
    fxp_conf = FxpArithmetic(fxp_params)
    return DatasetSpecification(
        dataset_type=SumDataset,
        dataset_location=Path(""),
        deployable_dataset_path=None,
        transform=lambda x: 
            fxp_conf.as_rational(fxp_conf.cut_as_integer(x)),
        target_transform=None,
    )


def _run_accuracy_test(host: Host, hw_manager: HWManager) -> dict[str, dict]:
    correct = 0
    total = 0
    num_bytes_outputs = 4
    if not hw_manager.test_loader:
        raise TypeError("Testloader not defined.")
    if not hw_manager.quantization_scheme:
        raise TypeError("Quantization Scheme is not defined")
    if not isinstance(host, SerialHost):
        raise TypeError("Need Serialhost for this test.")

    for inputs_rational, target in hw_manager.test_loader:

        data_bytearray = parse_fxp_tensor_to_bytearray(
            inputs_rational,
            hw_manager.quantization_scheme.total_bits,
            hw_manager.quantization_scheme.frac_bits,
        )
        batch_results_bytes = []
        for sample in data_bytearray:
            result_bytes = host.send_data_bytes(
                sample=sample,
                num_bytes_outputs=num_bytes_outputs,
            )
            batch_results_bytes.append(result_bytes)

        result = parse_bytearray_to_fxp_tensor(
            batch_results_bytes,
            hw_manager.quantization_scheme.total_bits,
            hw_manager.quantization_scheme.frac_bits,
            (BATCH_SIZE, num_bytes_outputs),
        )
        pred = result.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    return {Metric.ACCURACY.value: {"value": 100.0 * correct / total, "unit": "%"}}


def _run_validation_test(host: Host, hw_manager: HWManager) -> dict[str, dict]:
    correct = 0
    total = 0
    num_bytes_outputs = 2
    if not hw_manager.test_loader:
        raise TypeError("Testloader not defined.")
    if not hw_manager.quantization_scheme:
        raise TypeError("Quantization Scheme is not defined")
    if not isinstance(host, SerialHost):
        raise TypeError("Need Serialhost for this test.")
    val_input = torch.Tensor(
        [
            [5.2500, 14.2500],
            [6.7500, -8.0000],
            [-4.2500, 5.0000],
            [15.0000, 3.5000],
            [13.5000, 8.0000],
            [9.5000, -11.5000],
            [-5.0000, -10.2500],
            [4.7500, -4.2500],
            [-0.7500, 3.5000],
            [13.2500, 4.5000],
            [9.5000, -6.7500],
            [-3.2500, -11.2500],
            [4.5000, 10.2500],
            [-10.5000, -6.7500],
            [2.2500, 11.7500],
            [5.0000, -5.0000],
            [-11.7500, 14.0000],
            [-15.0000, 1.2500],
            [8.5000, -12.5000],
            [0.0000, 2.7500],
        ]
    )
    val_output = torch.Tensor(
        [
            [31.7500, -31.2500],
            [-1.5000, 31.7500],
            [6.0000, -31.2500],
            [31.7500, 21.0000],
            [31.7500, -10.0000],
            [-4.7500, 31.7500],
            [-28.7500, 31.7500],
            [7.0000, 31.7500],
            [17.5000, -22.2500],
            [31.7500, 10.0000],
            [20.0000, 31.7500],
            [-28.7500, 31.7500],
            [31.7500, -31.2500],
            [-28.7500, 12.2500],
            [31.7500, -31.2500],
            [4.5000, 31.7500],
            [12.0000, -31.2500],
            [-28.7500, -31.2500],
            [-15.5000, 31.7500],
            [17.5000, -15.7500],
        ]
    )
    fxp = FxpArithmetic(
        FxpParams(
            total_bits=hw_manager.quantization_scheme.total_bits,
            frac_bits=hw_manager.quantization_scheme.frac_bits,
        )
    )

    data_bytearray = parse_fxp_tensor_to_bytearray(
        val_input,
        hw_manager.quantization_scheme.total_bits,
        hw_manager.quantization_scheme.frac_bits,
    )
    batch_results_bytes = []
    for sample in data_bytearray:
        result_bytes = host.send_data_bytes(
            sample=sample,
            num_bytes_outputs=num_bytes_outputs,
        )
        batch_results_bytes.append(result_bytes)

    print(batch_results_bytes)

    result = parse_bytearray_to_fxp_tensor(
        batch_results_bytes,
        hw_manager.quantization_scheme.total_bits,
        hw_manager.quantization_scheme.frac_bits,
        (20, num_bytes_outputs),
    )
    print(result)
    return {Metric.ACCURACY.value: {"value": 100.0, "unit": "%"}}


def _run_latency_test(host: Host, hw_manager: HWManager) -> dict[str, dict]:
    import time

    elapsed_ns = 0
    total = 0
    num_bytes_outputs = 4
    if not hw_manager.test_loader:
        raise TypeError("Testloader not defined.")
    if not hw_manager.quantization_scheme:
        raise TypeError("Quantization Scheme is not defined")
    if not isinstance(host, SerialHost):
        raise TypeError("Need Serialhost for this test.")

    for inputs_rational, target in hw_manager.test_loader:
        data_bytearray = parse_fxp_tensor_to_bytearray(
            inputs_rational,
            hw_manager.quantization_scheme.total_bits,
            hw_manager.quantization_scheme.frac_bits,
        )
        for sample in data_bytearray:
            start_time = time.time_ns()
            result_bytes = host.send_data_bytes(
                sample=sample,
                num_bytes_outputs=num_bytes_outputs,
            )
            end_time = time.time_ns()
            elapsed_ns = elapsed_ns + (end_time - start_time)
            total += 1

    return {Metric.LATENCY.value: {"value": (elapsed_ns / total) / 1000, "unit": "us"}}


def search_generate_measure_for_env5(
    explorer: Explorer,
    fpga_type: str,
    serial_params: SerialParams,
    compiler_params: VivadoParams,
    search_space_path: Path,
    retrain_epochs: int = 4,
    max_search_trials: int = 4,
    top_n_models: int = 2,
):

    quantization_scheme = FixedPointInt8Scheme()
    explorer.choose_target_hw(fpga_type, compiler_params, serial_params)
    explorer.generate_search_space(search_space_path)
    dataset_spec = create_example_dataset_spec(quantization_scheme)
    optimization_criteria = setup_example_optimization_criteria(
        dataset_spec, device, (1, INPUT_DIM)
    )
    top_models, top_quantization_schemes = explorer.search(
        search_strategy=SearchStrategy.RANDOM_SEARCH,
        optimization_criteria=optimization_criteria,
        hw_nas_parameters=HWNASParameters(max_search_trials, top_n_models),
    )

    metric_to_source = {
        Metric.ACCURACY: _run_accuracy_test,
        # Metric.LATENCY: _run_latency_test,
    }
    df = measure_on_device(
        explorer=explorer,
        top_models=top_models,
        metric_to_source=metric_to_source,
        retrain_epochs=retrain_epochs,
        retrain_device="cpu",
        dataset_spec=dataset_spec,
        model_suffix="",
        top_quantization_schemes=top_quantization_schemes,
    )
    logger.info("Models:\n %s", df)


if __name__ == "__main__":
    max_search_trials = 4
    top_n_models = 4
    retrain_epochs = 10
    hw_platform = "env5_s50"

    # TODO obscure this
    compiler_params = VivadoParams(
        "/home/vivado/robin-build/", "65.108.38.237", "vivado", hw_platform
    )

    serial_params = SerialParams(
        device_path=Path(""), serial_port="/dev/ttyACM0", baud_rate=9600
    )

    knowledge_repo = setup_knowledge_repository()
    explorer = Explorer(knowledge_repo)
    search_space = Path("examples/search_space_examples/env5_search_space.yaml")
    search_generate_measure_for_env5(
        explorer,
        hw_platform,
        serial_params,
        compiler_params,
        search_space,
        retrain_epochs,
        max_search_trials,
        top_n_models,
    )
