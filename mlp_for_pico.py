import json
import logging
import logging.config
from pathlib import Path
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import nni

from torchvision.transforms import transforms
from elasticai.explorer import utils
from elasticai.explorer.config import DeploymentConfig, HWNASConfig, ModelConfig
from elasticai.explorer.data_to_csv import build_search_space_measurements_file
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.hw_nas.search_space.construct_sp import (
    CombinedSearchSpace,
    yml_to_dict,
)
from elasticai.explorer.knowledge_repository import (
    KnowledgeRepository,
    HWPlatform,
)
from elasticai.explorer.platforms.deployment.compiler import PicoCompiler
from elasticai.explorer.platforms.deployment.device_communication import (
    PicoHost,
)
from elasticai.explorer.platforms.deployment.manager import (
    CONTEXT_PATH,
    PicoHWManager,
    Metric,
)
from elasticai.explorer.platforms.generator.generator import (
    PicoGenerator,
)
from elasticai.explorer.trainer import MLPTrainer
from elasticai.explorer.utils import setup_mnist_for_cpp
from settings import ROOT_DIR

nni.enable_global_logging(False)
logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
logger = logging.getLogger("explorer.main")


def setup_knowledge_repository_pico() -> KnowledgeRepository:
    knowledge_repository = KnowledgeRepository()
    knowledge_repository.register_hw_platform(
        HWPlatform(
            "pico",
            "Pico with RP2040 MCU and 2MB control memory",
            PicoGenerator,
            PicoHWManager,
            PicoHost,
            PicoCompiler,
        )
    )

    return knowledge_repository


def find_generate_measure_for_pico(
    explorer: Explorer,
    deploy_cfg: DeploymentConfig,
    hwnas_cfg: HWNASConfig,
    search_space: CombinedSearchSpace,
):
    explorer.choose_target_hw(deploy_cfg)
    explorer.generate_search_space(search_space)
    top_models = explorer.search(hwnas_cfg)

    # Creating Train and Test set from MNIST #TODO build a generic dataclass/datawrapper
    transf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    path_to_dataset = ROOT_DIR / Path("data/mnist")
    trainloader: DataLoader = DataLoader(
        MNIST(path_to_dataset, download=True, transform=transf),
        batch_size=64,
        shuffle=True,
    )
    testloader: DataLoader = DataLoader(
        MNIST(path_to_dataset, download=True, train=False, transform=transf),
        batch_size=64,
    )
    root_dir_cpp_mnist = ROOT_DIR / Path("data/cpp-mnist")
    setup_mnist_for_cpp(str(path_to_dataset), str(root_dir_cpp_mnist))

    latency_measurements = []
    accuracy_measurements_on_device = []
    accuracy_after_retrain = []
    retrain_device = "cpu"
    for i, model in enumerate(top_models):
        mlp_trainer = MLPTrainer(
            device=retrain_device,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),  # type: ignore
        )
        mlp_trainer.train(model, trainloader=trainloader, epochs=4)
        accuracy_after_retrain_value = mlp_trainer.test(model, testloader=testloader)
        model_name = "ts_model_" + str(i) + ".tflite"
        explorer.generate_for_hw_platform(model, model_name)
        explorer.hw_setup_on_target(Path("data/cpp-mnist"))

        try:
            latency = explorer.run_measurement(Metric.LATENCY, model_name)
        except Exception as e:
            latency = json.loads('{ "Latency": { "value": -2, "unit": "microseconds"}}')
            print(f"An error occurred when measuring Latency on Pico: {e}")
        try:
            accuracy_on_device = explorer.run_measurement(Metric.ACCURACY, model_name)
        except Exception as e:
            accuracy_on_device = json.loads(
                '{"Accuracy": { "value":  -2, "unit": "percent"}}'
            )
            print(f"An error occurred when measuring accuracy on Pico: {e}")

        accuracy_after_retrain_dict = json.loads(
            '{"Accuracy after retrain": { "value":'
            + str(accuracy_after_retrain_value)
            + ' , "unit": "percent"}}'
        )
        latency_measurements.append(latency)
        accuracy_measurements_on_device.append(accuracy_on_device)
        accuracy_after_retrain.append(accuracy_after_retrain_dict)

    latencies = [latency["Latency"]["value"] for latency in latency_measurements]
    accuracies_on_device = [
        accuracy["Accuracy"]["value"] for accuracy in accuracy_measurements_on_device
    ]
    accuracy_after_retrain = [
        accuracy["Accuracy after retrain"]["value"]
        for accuracy in accuracy_after_retrain
    ]

    df = build_search_space_measurements_file(
        latencies,
        accuracies_on_device,
        accuracy_after_retrain,
        explorer.metric_dir / "metrics.json",
        explorer.model_dir / "models.json",
        explorer.experiment_dir / "experiment_data.csv",
    )
    logger.info("Models:\n %s", df)


if __name__ == "__main__":
    hwnas_cfg = HWNASConfig(config_path=Path("configs/pico/hwnas_config.yaml"))
    deploy_cfg = DeploymentConfig(
        config_path=Path("configs/pico/deployment_config.yaml")
    )
    model_cfg = ModelConfig(config_path=Path("configs/pico/model_config.yaml"))

    knowledge_repo = setup_knowledge_repository_pico()
    explorer = Explorer(knowledge_repo)
    explorer.set_model_cfg(model_cfg)
    search_space = yml_to_dict(
        Path("elasticai/explorer/hw_nas/search_space/search_space.yml")
    )
    search_space = CombinedSearchSpace(search_space)

    find_generate_measure_for_pico(
        explorer, deploy_cfg, hwnas_cfg, search_space  # type:ignore
    )
