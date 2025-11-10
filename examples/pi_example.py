import logging.config
from pathlib import Path
import torch


from elasticai.explorer.hw_nas.hw_nas import HWNASParameters, SearchStrategy
from elasticai.explorer.explorer import Explorer

from elasticai.explorer.platforms.deployment.compiler import CompilerParams
from elasticai.explorer.platforms.deployment.device_communication import SSHParams

from elasticai.explorer.platforms.deployment.hw_manager import Metric

from examples.example_helpers import (
    measure_on_device,
    setup_knowledge_repository,
    setup_mnist,
    setup_example_optimization_criteria,
)
from settings import ROOT_DIR

logging.config.fileConfig(ROOT_DIR / "logging.conf", disable_existing_loggers=False)

logger = logging.getLogger("explorer.main")
device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def search_generate_measure_for_pi(
    explorer: Explorer,
    rpi_type: str,
    ssh_params: SSHParams,
    compiler_params: CompilerParams,
    search_space_path: Path,
    retrain_epochs: int = 4,
    max_search_trials: int = 4,
    top_n_models: int = 2,
):
    explorer.choose_target_hw(rpi_type, compiler_params, ssh_params)
    explorer.generate_search_space(search_space_path)

    path_to_test_data = ROOT_DIR / Path("data/mnist")
    dataset_spec = setup_mnist(path_to_test_data)
    criteria_reg = setup_example_optimization_criteria(dataset_spec, device)

    top_models = explorer.search(
        search_strategy=SearchStrategy.EVOlUTIONARY_SEARCH,
        optimization_criteria_registry=criteria_reg,
        hw_nas_parameters=HWNASParameters(max_search_trials, top_n_models),
    )
    metric_to_source = {
        Metric.ACCURACY: Path("code/measure_accuracy_mnist.cpp"),
        Metric.LATENCY: Path("code/measure_latency.cpp"),
    }
    explorer.hw_setup_on_target(metric_to_source, dataset_spec)

    df = measure_on_device(
        explorer, top_models, metric_to_source, retrain_epochs, device, dataset_spec
    )

    logger.info("Summary:\n %s", df)


if __name__ == "__main__":
    ### Hyperparameters
    max_search_trials = 6
    top_n_models = 2
    retrain_epochs = 3
    rpi_type = "rpi5"

    ssh_params = SSHParams(
        hostname="<hostname>", username="<username>"
    )  # <-- connection details for your RPi
    compiler_params = CompilerParams()  # <-- configure this only if necessary
    knowledge_repo = setup_knowledge_repository()
    explorer = Explorer(knowledge_repo)

    search_space = Path(
        ROOT_DIR / "examples/search_space_examples/pi_search_space.yaml"
    )

    search_generate_measure_for_pi(
        explorer=explorer,
        rpi_type=rpi_type,
        ssh_params=ssh_params,
        compiler_params=compiler_params,
        search_space_path=search_space,
        retrain_epochs=retrain_epochs,
        max_search_trials=max_search_trials,
        top_n_models=top_n_models,
    )
