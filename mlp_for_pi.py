import logging
import logging.config
from pathlib import Path
import shutil
import nni
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from elasticai.explorer.data_to_csv import build_search_space_measurements_file
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository.knowledge_repository import (
    KnowledgeRepository,
)

from elasticai.explorer.platforms.deployment.manager import Metric
from elasticai.explorer.trainer import MLPTrainer
from elasticai.explorer.config import DeploymentConfig, HWNASConfig, ModelConfig
from settings import ROOT_DIR

nni.enable_global_logging(False)
logging.config.fileConfig("logging.conf", disable_existing_loggers=False)

logger = logging.getLogger("explorer.main")


def find_generate_measure_for_pi(
    explorer: Explorer,
    deploy_cfg: DeploymentConfig,
    hwnas_cfg: HWNASConfig,
) -> None:
    explorer.choose_target_hw(deploy_cfg)
    explorer.generate_search_space()
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

    retrain_device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    for i, model in enumerate(top_models):
        mlp_trainer = MLPTrainer(
            device=retrain_device,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),  # type: ignore
        )
        mlp_trainer.train(model, trainloader=trainloader, epochs=3)
        mlp_trainer.test(model, testloader=testloader)
        model_name = "ts_model_" + str(i) + ".pt"
        explorer.generate_for_hw_platform(model, model_name)

        latency = explorer.run_measurement(Metric.LATENCY, model_name)
        latency_measurements.append(latency)
        accuracy_measurements.append(
            explorer.run_measurement(Metric.ACCURACY, model_name)
        )

    latencies = [latency["Latency"]["value"] for latency in latency_measurements]
    _accuracies = [accuracy["Accuracy"]["value"] for accuracy in accuracy_measurements]
    df = build_search_space_measurements_file(
        latencies,
        explorer.metric_dir / "metrics.json",
        explorer.model_dir / "models.json",
        explorer.experiment_dir / "experiment_data.csv",
    )
    logger.info("Models:\n %s", df)


if __name__ == "__main__":
    hwnas_cfg = HWNASConfig(config_path=Path("configs/hwnas_config.yaml"))
    deploy_cfg = DeploymentConfig(config_path=Path("configs/deployment_config.yaml"))
    model_cfg = ModelConfig(config_path=Path("configs/model_config.yaml"))

    knowledge_repository = KnowledgeRepository()
    knowledge_repository.load_hw_platforms()

    explorer = Explorer(knowledge_repository)
    explorer.set_model_cfg(model_cfg)
    find_generate_measure_for_pi(explorer, deploy_cfg, hwnas_cfg)
