import logging.config
from pathlib import Path
import torch
from typing import Any, Callable

from elasticai.explorer.generator.generator import Generator
from elasticai.explorer.generator_registry import GeneratorRegistry
from elasticai.explorer.training.data import (
    BaseDataset,
    DatasetSpecification,
)
from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)


from elasticai.explorer.hw_nas.hw_nas import HWNASParameters, SearchStrategy
from elasticai.explorer.hw_nas.search_space.quantization import (
    CreatorFixedPointScheme,
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

from elasticai.implementations.creator_generator.deployment import CreatorModelTranslator, ENv5Compiler, ENv5HWManager, ENv5Host
from elasticai.implementations.creator_generator.model_builder import CreatorModelBuilder
from elasticai.implementations.creator_generator.quantization_utils import (
    parse_bytearray_to_fxp_tensor,
    parse_fxp_tensor_to_bytearray,
)
from examples.example_helpers import (
    measure_on_device,
    setup_example_optimization_criteria,
)
from settings import EXPERIMENTS_DIR

logging.config.fileConfig("logging.conf", disable_existing_loggers=False)

logger = logging.getLogger("explorer.main")


device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
BATCH_SIZE = 64
INPUT_DIM = 6


def setup_generator_registry():
    generator_registry = GeneratorRegistry()

    generator_registry.register_generator(
        Generator(
            "env5_s50",
            "Env5 with RP2040 and xc7s50ftgb196-2 FPGA",
            CreatorModelTranslator,
            ENv5HWManager,
            ENv5Host,
            ENv5Compiler,
            CreatorModelBuilder,
        )
    )
    generator_registry.register_generator(
        Generator(
            "env5_s15",
            "Env5 with RP2040 and xc7s15ftgb196-2 FPGA",
            CreatorModelTranslator,
            ENv5HWManager,
            ENv5Host,
            ENv5Compiler,
            CreatorModelBuilder,
        )
    )
    return generator_registry


class NotAndDataset(BaseDataset):
    def __init__(
        self,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.target_transform = target_transform
        self.transform = transform
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
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        thresholds=[-0.5, 0.0, 0.5],
        noise_std=0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data = torch.randn(BATCH_SIZE * 100, INPUT_DIM)
        summed = self.data.sum(dim=1)
        if noise_std > 0:
            summed = summed + noise_std * torch.randn_like(summed)
        self.target_transform = target_transform
        self.transform = transform
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
        dataset=SumDataset(
            transform=lambda x: fxp_conf.as_rational(fxp_conf.cut_as_integer(x)),
            target_transform=None,
        )
    )


def check_hw_manager(hw_manager: HWManager, host: Host):
    if not hw_manager.test_loader:
        raise TypeError("Testloader not defined.")
    if (
        not hw_manager.quantization_scheme
        or hw_manager.quantization_scheme is not CreatorFixedPointScheme
    ):
        raise TypeError("Quantization Scheme is not defined correctly.")
    if not isinstance(host, SerialHost):
        raise TypeError("Need Serialhost for this test.")


def _run_accuracy_test(host: Host, hw_manager: HWManager) -> dict[str, dict]:
    correct = 0
    total = 0
    num_bytes_outputs = 4
    if not hw_manager.test_loader:
        raise TypeError("Testloader not defined.")
    if (
        not hw_manager.quantization_scheme
        or type (hw_manager.quantization_scheme) is not CreatorFixedPointScheme
    ):
        raise TypeError("Quantization Scheme is not defined correctly.")
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
    fpga_type: str,
    serial_params: SerialParams,
    compiler_params: VivadoParams,
    search_space_path: Path,
    retrain_epochs: int = 4,
    max_search_trials: int = 4,
    top_n_models: int = 2,
):

    quantization_scheme = CreatorFixedPointScheme()
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
    max_search_trials = 2
    top_n_models = 2
    retrain_epochs = 2
    hw_platform = "env5_s50"

    compiler_params = VivadoParams(
        "/home/vivado/robin-build/", "65.108.38.237", "vivado", hw_platform
    )

    serial_params = SerialParams(
        device_path=Path("RPI-RP2"), serial_port="/dev/ttyACM0", baud_rate=9600
    )

    generator_registry = setup_generator_registry()
    explorer = Explorer(generator_registry, experiments_dir=EXPERIMENTS_DIR)
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
