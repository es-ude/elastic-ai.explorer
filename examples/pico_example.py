import logging.config
from pathlib import Path

import torch
from torchvision.transforms import transforms

from elasticai.explorer.explorer import Explorer
from elasticai.explorer.hw_nas.hw_nas import HWNASParameters, SearchStrategy

from elasticai.explorer.generator.deployment.compiler import CompilerParams
from elasticai.explorer.generator.deployment.device_communication import SerialParams

from elasticai.explorer.generator.deployment.hw_manager import Metric
from elasticai.explorer.training.data import DatasetSpecification, MNISTWrapper
from elasticai.explorer_impl.pico_generator.utils import setup_mnist_for_cpp

from examples.example_helpers import (
    measure_on_device,
    setup_generator_registry,
    setup_example_optimization_criteria,
)

from settings import DOCKER_CONTEXT_DIR, EXPERIMENTS_DIR, ROOT_DIR

logging.config.fileConfig(ROOT_DIR / "logging.conf", disable_existing_loggers=False)
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
    target: str = "pico",
):
    explorer.choose_target_hw(target, compiler_params, serial_params)
    explorer.generate_search_space(search_space)

    transf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    path_to_dataset = ROOT_DIR / Path("data/mnist")
    root_dir_cpp_mnist = ROOT_DIR / Path("data/cpp-mnist")
    setup_mnist_for_cpp(path_to_dataset, root_dir_cpp_mnist, transf)

    dataset_spec = DatasetSpecification(
        dataset=MNISTWrapper(root=path_to_dataset, transform=transf),
        deployable_dataset_path=root_dir_cpp_mnist,
    )
    criteria = setup_example_optimization_criteria(dataset_spec, device)

    top_models, top_quant_schemes = explorer.search(
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

    df = measure_on_device(
        explorer,
        top_models,
        metric_to_source,
        retrain_epochs,
        "cpu",
        dataset_spec,
        model_suffix=".tflite",
        top_quantization_schemes=top_quant_schemes,
    )
    logger.info("Models:\n %s", df)


if __name__ == "__main__":
    ### Hyperparameters
    max_search_trials = 6
    top_n_models = 2
    retrain_epochs = 3

    # additional device specifics, changes are necessary
    target_platform_name = "pico"  # <-- or pico2
    base_dockerfile = "docker/Dockerfile.picobase"
    cross_dockerfile = "docker/Dockerfile.picocross"
    usb_device_path = Path(
        "/media/<username>/RPI-RP2"
    )  # <-- add your username and for pico2 this should be "/media/<username>/RP2350" instead
    image_name = (
        "picobase"  # <-- for pico2 use "pico2base" to create a separate base image
    )
    serial_params = SerialParams(device_path=usb_device_path)
    compiler_params = CompilerParams(
        library_path=Path("./code/pico_crosscompiler"),
        cross_dockerfile_path=ROOT_DIR / cross_dockerfile,
        base_image_name=image_name,
        build_context=DOCKER_CONTEXT_DIR,
        base_dockerfile_path=ROOT_DIR / base_dockerfile,
        additional_params= {"platform_type": "rp2040"}, # <-- for pico2 set this to "rp2350"
    )

    knowledge_repo = setup_generator_registry()
    explorer = Explorer(knowledge_repo, EXPERIMENTS_DIR)
    search_space = Path("examples/search_space_examples/pico_search_space.yaml")
    search_generate_measure_for_pico(
        explorer,
        compiler_params=compiler_params,
        serial_params=serial_params,
        search_space=search_space,
        retrain_epochs=retrain_epochs,
        max_search_trials=max_search_trials,
        top_n_models=top_n_models,
        target=target_platform_name,
    )
