import logging.config
from pathlib import Path
import torch

from elasticai.explorer.generator.generator import Generator
from elasticai.explorer.generator_registry import GeneratorRegistry
from elasticai.explorer.training.data import (
    DatasetSpecification,
)


from elasticai.explorer.hw_nas.hw_nas import HWNASParameters, SearchStrategy

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

from elasticai.creator.testing import run_cocotb_sim
from elasticai.explorer.utils.data_utils import load_json
from elasticai.explorer_impl.creator_generator.experimental_deployment.deployment import (
    CreatorEnv5ModelTranslator,
    CreatorBaseModelTranslator,
    ENv5Compiler,
    ENv5HWManager,
    ENv5Host,
)
from elasticai.explorer_impl.creator_generator.model_builder import (
    CreatorModelBuilder,
)
from elasticai.explorer_impl.creator_generator.quantization_utils import (
    parse_bytearray_to_fxp_tensor,
    parse_fxp_tensor_to_bytearray,
)


from elasticai.explorer_impl.creator_generator.simulation.dummy import (
    DummyCompiler,
    DummyHost,
)
from elasticai.explorer_impl.creator_generator.simulation.simulation_utils import (
    _prep_simulation,
)
from examples.example_helpers import (
    SumDataset,
    measure_on_device,
    setup_example_optimization_criteria,
)

from settings import EXPERIMENTS_DIR

logging.config.fileConfig("logging.conf", disable_existing_loggers=False)

logger = logging.getLogger("explorer.main")


device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
INPUT_DIM = 6
OUTPUT_DIM = 4


def setup_generator_registry():
    generator_registry = GeneratorRegistry()

    generator_registry.register_generator(
        Generator(
            "env5_s50",
            "Env5 with RP2040 and xc7s50ftgb196-2 FPGA",
            CreatorEnv5ModelTranslator,
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
            CreatorEnv5ModelTranslator,
            ENv5HWManager,
            ENv5Host,
            ENv5Compiler,
            CreatorModelBuilder,
        )
    )

    generator_registry.register_generator(
        Generator(
            "env5_simulation",
            "Cocotb/GHDL Simulation",
            CreatorBaseModelTranslator,
            ENv5HWManager,
            DummyHost,
            DummyCompiler,
            CreatorModelBuilder,
        )
    )
    return generator_registry


def _run_accuracy_simulation(host: Host, hw_manager: HWManager, path_to_model: Path):

    src_files, file_name, output_dir, result_file = _prep_simulation(
        host, hw_manager, path_to_model, INPUT_DIM, OUTPUT_DIM
    )
    results = {}
    try:
        run_cocotb_sim(
            src_files=src_files,
            top_module_name=file_name,
            cocotb_test_module="elasticai.explorer_plugins.creator_generator.simulation.simulation",
            waveform_save_dst=str(output_dir),
        )
        results = load_json(result_file)
    except:
        print(f"Simulation failed for {path_to_model}!")
        results["accuracy_percent"] = -1

    return {Metric.ACCURACY.value: {"value": results["accuracy_percent"], "unit": "%"}}


def _run_latency_simulation(host: Host, hw_manager: HWManager, path_to_model: Path):

    src_files, file_name, output_dir, result_file = _prep_simulation(
        host, hw_manager, path_to_model, INPUT_DIM, OUTPUT_DIM
    )
    results = {}
    try:
        run_cocotb_sim(
            src_files=src_files,
            top_module_name=file_name,
            cocotb_test_module="elasticai.explorer_plugins.creator_generator.simulation.simulation",
            waveform_save_dst=str(output_dir),
        )
        results = load_json(result_file)
    except:
        print(f"Simulation failed for {path_to_model}!")
        results["latency_ns"] = -1
    results = load_json(result_file)
    return {Metric.LATENCY.value: {"value": results["latency_ns"], "unit": "ns"}}


def _run_accuracy_deployed(
    host: Host, hw_manager: HWManager, path_to_model: Path
) -> dict[str, dict]:
    correct = 0
    total = 0
    num_bytes_outputs = OUTPUT_DIM
    if not hw_manager.test_loader:
        raise TypeError("Testloader not defined.")

    if (
        not hw_manager.quantization_scheme
        or not hw_manager.quantization_scheme.total_bits
        or not hw_manager.quantization_scheme.frac_bits
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
            print(result_bytes)
            batch_results_bytes.append(result_bytes)

        result = parse_bytearray_to_fxp_tensor(
            batch_results_bytes,
            hw_manager.quantization_scheme.total_bits,
            hw_manager.quantization_scheme.frac_bits,
            (64, num_bytes_outputs),
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

    explorer.choose_target_hw(fpga_type, compiler_params, serial_params)
    explorer.generate_search_space(search_space_path)
    dataset_spec = DatasetSpecification(
        dataset=SumDataset(kwargs={"input_dim": INPUT_DIM, "size": 12000})
    )
    optimization_criteria = setup_example_optimization_criteria(
        dataset_spec, device, (1, INPUT_DIM)
    )
    top_models, top_quantization_schemes = explorer.search(
        search_strategy=SearchStrategy.RANDOM_SEARCH,
        optimization_criteria=optimization_criteria,
        hw_nas_parameters=HWNASParameters(max_search_trials, top_n_models),
    )

    metric_to_source = {
        Metric.ACCURACY: _run_accuracy_simulation, # For an example of measuring directly on the env5 add _run_accuracy_deployed.
        Metric.LATENCY: _run_latency_simulation,
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
    retrain_epochs = 15
    hw_platform = "env5_simulation"

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
