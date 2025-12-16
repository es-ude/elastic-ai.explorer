import logging.config
from pathlib import Path
import torch

from elasticai.explorer.generator.generator import Generator
from elasticai.explorer.generator.model_builder.model_builder import CreatorModelBuilder
from elasticai.explorer.hw_nas.hw_nas import HWNASParameters, SearchStrategy
from elasticai.explorer.hw_nas.search_space.quantization import (
    FixedPointInt8Scheme,
    parse_bytearray_to_fxp_tensor,
    parse_fxp_tensor_to_bytearray,
)

from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import (
    KnowledgeRepository,
)
from elasticai.explorer.generator.deployment.compiler import ENv5Compiler, VivadoParams
from elasticai.explorer.generator.deployment.device_communication import (
    ENv5Host,
    Host,
    SerialHost,
    SerialParams,
)
from elasticai.explorer.generator.deployment.hw_manager import (
    ENv5HWManager,
    HWManager,
    Metric,
)
from elasticai.explorer.generator.model_compiler.model_compiler import (
    CreatorModelCompiler,
)
from examples.example_helpers import (
    measure_on_device,
    setup_example_optimization_criteria,
    setup_knowledge_repository,
)

logging.config.fileConfig("logging.conf", disable_existing_loggers=False)

logger = logging.getLogger("explorer.main")


from pathlib import Path
from typing import Any, Callable

import torch
from elasticai.explorer.training.data import BaseDataset, DatasetSpecification
from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)

device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
BATCH_SIZE = 64
INPUT_DIM = 6


class SumDataset(BaseDataset):
    def __init__(
        self,
        root,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        thresholds=[-1.5, 0.0, 1.5],
        noise_std=0.0,
        *args,
        **kwargs,
    ):
        super().__init__(root, transform, target_transform, *args, **kwargs)
        self.data = torch.randn(BATCH_SIZE * 10, INPUT_DIM)
        summed = self.data.sum(dim=1)
        if noise_std > 0:
            summed = summed + noise_std * torch.randn_like(summed)

        self.targets = torch.empty(BATCH_SIZE * 10, dtype=torch.long)
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
        transform=lambda x: fxp_conf.as_rational(fxp_conf.cut_as_integer(x)),
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


def search_generate_measure_for_env5(
    explorer: Explorer,
    rpi_type: str,
    serial_params: SerialParams,
    compiler_params: VivadoParams,
    search_space_path: Path,
    retrain_epochs: int = 4,
    max_search_trials: int = 4,
    top_n_models: int = 2,
):

    quantization_scheme = FixedPointInt8Scheme()
    explorer.choose_target_hw(rpi_type, compiler_params, serial_params)
    explorer.generate_search_space(search_space_path)
    dataset_spec = create_example_dataset_spec(quantization_scheme)
    optimization_criteria = setup_example_optimization_criteria(
        dataset_spec, device, (1, 1, INPUT_DIM)
    )
    top_models = explorer.search(
        search_strategy=SearchStrategy.RANDOM_SEARCH,
        optimization_criteria=optimization_criteria,
        hw_nas_parameters=HWNASParameters(max_search_trials, top_n_models),
        model_builder=CreatorModelBuilder(),
    )

    metric_to_source = {Metric.ACCURACY: _run_accuracy_test}
    df = measure_on_device(
        explorer=explorer,
        top_models=top_models,
        metric_to_source=metric_to_source,
        retrain_epochs=retrain_epochs,
        device="cpu",
        dataset_spec=dataset_spec,
        model_suffix="",
        quantization_scheme=quantization_scheme,
    )
    logger.info("Models:\n %s", df)


if __name__ == "__main__":
    max_search_trials = 2
    top_n_models = 1
    retrain_epochs = 5
    hw_platform = "env5_s15"

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
    retrain_epochs = 3
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
