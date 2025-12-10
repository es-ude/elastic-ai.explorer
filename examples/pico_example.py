import logging
import logging.config
from pathlib import Path

import torch
from torchvision.transforms import transforms

from elasticai.explorer.explorer import Explorer
from elasticai.explorer.hw_nas.hw_nas import HWNASParameters, SearchStrategy
from elasticai.explorer.generator.generator import Generator
from elasticai.explorer.knowledge_repository import (
    KnowledgeRepository,
)
from elasticai.explorer.generator.deployment.compiler import PicoCompiler
from elasticai.explorer.generator.deployment.device_communication import (
    PicoHost,
)
from elasticai.explorer.generator.deployment.hw_manager import (
    PicoHWManager,
    Metric,
)
from elasticai.explorer.generator.model_compiler.model_compiler import (
    TFliteModelCompiler,
)
from elasticai.explorer.generator.deployment.hw_manager import Metric
from elasticai.explorer.training.data import DatasetSpecification, MNISTWrapper
from elasticai.explorer.utils.data_utils import setup_mnist_for_cpp

from examples.example_helpers import (
    measure_on_device,
    setup_knowledge_repository,
    setup_example_optimization_criteria,
)

from settings import DOCKER_CONTEXT_DIR, ROOT_DIR

logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
logger = logging.getLogger("explorer.main")
device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def search_generate_measure_for_pico(
    explorer: Explorer,
    serial_params: SerialParams,
    compiler_params: CompilerParams,
    search_space: Path,
    retrain_epochs: int = 4,
    max_search_trials: int = 2,
    top_n_models: int = 2,
):
    explorer.choose_target_hw("pico", compiler_params, serial_params)
    explorer.generate_search_space(search_space)

    transf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    path_to_dataset = ROOT_DIR / Path("data/mnist")
    root_dir_cpp_mnist = ROOT_DIR / Path("data/cpp-mnist")
    setup_mnist_for_cpp(path_to_dataset, root_dir_cpp_mnist, transf)

    dataset_spec = DatasetSpecification(
        dataset_type=MNISTWrapper,
        dataset_location=path_to_dataset,
        deployable_dataset_path=root_dir_cpp_mnist,
        transform=transf,
    )
    criteria = setup_example_optimization_criteria(dataset_spec, device)

    top_models = explorer.search(
        search_strategy=SearchStrategy.EVOLUTIONARY_SEARCH,
        hw_nas_parameters=HWNASParameters(
            max_search_trials=max_search_trials, top_n_models=top_n_models
        ),
        optimization_criteria=criteria,
    )

    metric_to_source = {
        Metric.ACCURACY: Path("code/pico_crosscompiler/measure_accuracy"),
        Metric.LATENCY: Path("code/pico_crosscompiler/measure_latency"),
    }
    explorer.hw_setup_on_target(metric_to_source, dataset_spec)

    df = measure_on_device(
        explorer,
        top_models,
        metric_to_source,
        retrain_epochs,
        "cpu",
        dataset_spec,
        model_suffix=".tflite",
    )
    logger.info("Models:\n %s", df)


if __name__ == "__main__":
    ### Hyperparameters
    max_search_trials = 6
    top_n_models = 2
    retrain_epochs = 3

    serial_params = SerialParams(
        device_path=Path("/media/<username>/RPI-RP2")
    )  # <-- Set the device path and rest only if necessary.
    compiler_params = CompilerParams(
        library_path=Path("./code/pico_crosscompiler"),
        image_name="picobase",
        build_context=DOCKER_CONTEXT_DIR,
        path_to_dockerfile=ROOT_DIR / "docker/Dockerfile.picobase",
    )  # <-- Configure this only if necessary.

    knowledge_repo = setup_knowledge_repository()
    explorer = Explorer(knowledge_repo)
    search_space = Path("examples/search_space_examples/pico_search_space.yaml")
    search_generate_measure_for_pico(
        explorer,
        compiler_params=compiler_params,
        serial_params=serial_params,
        search_space=search_space,
        retrain_epochs=retrain_epochs,
        max_search_trials=max_search_trials,
        top_n_models=top_n_models,
    )
