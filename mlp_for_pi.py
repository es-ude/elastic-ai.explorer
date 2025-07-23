import json
import logging.config
import shutil
from pathlib import Path

import nni
import torch
from nni.nas.nn.pytorch import ModelSpace
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from elasticai_explorer.config import DeploymentConfig, HWNASConfig
from elasticai_explorer.data_to_csv import build_search_space_measurements_file
from elasticai_explorer.explorer import Explorer
from elasticai_explorer.hw_nas.search_space.construct_sp import (
    yml_to_dict,
    CombinedSearchSpace,
)
from elasticai_explorer.knowledge_repository import (
    KnowledgeRepository,
    Generator,
    Metrics,
)
from elasticai_explorer.platforms.deployment.compiler import RPICompiler
from elasticai_explorer.platforms.deployment.device_communication import RPIHost
from elasticai_explorer.platforms.deployment.manager import (
    PIHWManager,
    Metric,
)
from elasticai_explorer.platforms.generator.model_compiler import (
    TorchscriptCompiler,
)
from elasticai_explorer.trainer import MLPTrainer

nni.enable_global_logging(False)
logging.config.fileConfig("logging.conf", disable_existing_loggers=False)

logger = logging.getLogger("explorer.main")


def setup_knowledge_repository_pi() -> KnowledgeRepository:
    knowledge_repository = KnowledgeRepository()
    knowledge_repository.register_hw_platform(
        Generator(
            "rpi5",
            "Raspberry PI 5 with A76 processor and 8GB RAM",
            TorchscriptCompiler,
            PIHWManager,
            RPIHost,
            RPICompiler,
        )
    )

    knowledge_repository.register_hw_platform(
        Generator(
            "rpi4",
            "Raspberry PI 4 with A72 processor and 4GB RAM",
            TorchscriptCompiler,
            PIHWManager,
            RPIHost,
            RPICompiler,
        )
    )

    return knowledge_repository


def find_generate_measure_for_pi(
    explorer: Explorer,
    deploy_cfg: DeploymentConfig,
    hwnas_cfg: HWNASConfig,
    search_space: CombinedSearchSpace,
) -> Metrics:

    explorer.generate_search_space(search_space)
    top_models = explorer.search(hwnas_cfg)

    # Creating Train and Test set from MNIST #TODO build a generic dataclass/datawrapper
    transf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    path_to_dataset = Path("data/mnist")
    trainloader: DataLoader = DataLoader(
        MNIST(path_to_dataset, download=True, transform=transf),
        batch_size=64,
        shuffle=True,
    )
    testloader: DataLoader = DataLoader(
        MNIST(path_to_dataset, download=True, train=False, transform=transf),
        batch_size=64,
    )

    path_to_test_data = "docker/data/mnist"
    shutil.make_archive(path_to_test_data, "zip", "data/mnist/MNIST/raw")
    explorer.hw_setup_on_target(Path(path_to_test_data + ".zip"))

    latency_measurements = []
    accuracy_measurements = []
    accuracy_after_retrain = []

    retrain_device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    for i, model in enumerate(top_models):
        mlp_trainer = MLPTrainer(
            device=retrain_device,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),  # type: ignore
        )
        mlp_trainer.train(model, trainloader=trainloader, epochs=4)
        accuracy_after_retrain_value = mlp_trainer.test(model, testloader=testloader)
        model_name = "ts_model_" + str(i) + ".pt"
        explorer.generate_for_hw_platform(model, model_name)

        latency = explorer.run_measurement(Metric.LATENCY, model_name)
        latency_measurements.append(latency)
        accuracy_measurements.append(
            explorer.run_measurement(Metric.ACCURACY, model_name)
        )

        accuracy_after_retrain_dict = json.loads(
            '{"Accuracy after retrain": { "value":'
            + str(accuracy_after_retrain_value)
            + ' , "unit": "percent"}}'
        )
        accuracy_after_retrain.append(accuracy_after_retrain_dict)

    latencies = [latency["Latency"]["value"] for latency in latency_measurements]
    accuracies_on_device = [
        accuracy["Accuracy"]["value"] for accuracy in accuracy_measurements
    ]
    accuracies_after_retrain = [
        accuracy["Accuracy after retrain"]["value"]
        for accuracy in accuracy_after_retrain
    ]

    df = build_search_space_measurements_file(
        latencies,
        explorer.metric_dir / "metrics.json",
        explorer.model_dir / "models.json",
        explorer.experiment_dir / "experiment_data.csv",
    )
    logger.info("Models:\n %s", df)

    return Metrics(
        explorer.metric_dir / "metrics.json",
        explorer.model_dir / "models.json",
        accuracies_on_device,
        latencies,
    )


def search_models(explorer: Explorer, hwnas_cfg: HWNASConfig, search_space: ModelSpace):
    deploy_cfg = DeploymentConfig(config_path=Path("configs/deployment_config.yaml"))
    explorer.choose_target_hw(deploy_cfg)
    explorer.generate_search_space(search_space)
    top_models = explorer.search(hwnas_cfg)

    # Creating Train and Test set from MNIST #TODO build a generic dataclass/datawrapper
    transf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    path_to_dataset = Path("data/mnist")
    trainloader: DataLoader = DataLoader(
        MNIST(path_to_dataset, download=True, transform=transf),
        batch_size=64,
        shuffle=True,
    )
    testloader: DataLoader = DataLoader(
        MNIST(path_to_dataset, download=True, train=False, transform=transf),
        batch_size=64,
    )
    retrain_device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    for i, model in enumerate(top_models):
        print(f"found model {i}:  {model}")

        mlp_trainer = MLPTrainer(
            device=retrain_device,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),  # type: ignore
        )
        mlp_trainer.train(model, trainloader=trainloader, epochs=4)
        mlp_trainer.test(model, testloader=testloader)
        print("=================================================")
        model_name = "ts_model_" + str(i) + ".pt"

        explorer.generate_for_hw_platform(model, model_name)


if __name__ == "__main__":
    hwnas_cfg = HWNASConfig(config_path=Path("configs/hwnas_config.yaml"))
    deploy_cfg = DeploymentConfig(config_path=Path("configs/deployment_config.yaml"))
    knowledge_repo = setup_knowledge_repository_pi()
    explorer = Explorer(knowledge_repo)
    explorer.choose_target_hw(deploy_cfg)
    search_space = yml_to_dict(Path("hw_nas/search_space/search_space.yml"))
    search_space = CombinedSearchSpace(search_space)

    find_generate_measure_for_pi(explorer, deploy_cfg, hwnas_cfg, search_space)

    # search_space = yml_to_dict(
    #     Path("explorer/hw_nas/search_space/search_space.yml")
    # )
    # search_space = CombinedSearchSpace(search_space)
    # search_models(explorer, hwnas_cfg, search_space)
