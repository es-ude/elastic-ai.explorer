import logging
import os
from logging import config

from torchvision.transforms import transforms
from torchvision.datasets import MNIST
import nni
import torch
from torch.utils.data import DataLoader



from elasticai.explorer.data_to_csv import build_search_space_measurements_file
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import (
    KnowledgeRepository,
    HWPlatform,
    Metrics,
)
from elasticai.explorer.platforms.deployment.manager import PIHWManager
from elasticai.explorer.platforms.generator.generator import PIGenerator
from elasticai.explorer.trainer import MLPTrainer
from elasticai.explorer.visualizer import Visualizer
from elasticai.explorer.config import Config, ConnectionConfig, HWNASConfig, ModelConfig
from settings import ROOT_DIR

config = None

nni.enable_global_logging(False)
logging.config.fileConfig("logging.conf", disable_existing_loggers=False)

logger = logging.getLogger("explorer.main")


def setup_knowledge_repository() -> KnowledgeRepository:
    knowledge_repository = KnowledgeRepository()
    knowledge_repository.register_hw_platform(
        HWPlatform(
            "rpi5",
            "Raspberry PI 5 with A76 processor and 8GB RAM",
            PIGenerator,
            PIHWManager,
        )
    )
    return knowledge_repository


def find_for_pi(knowledge_repository: KnowledgeRepository, explorer: Explorer):

    explorer.choose_target_hw("rpi5")
    explorer.generate_search_space()
    top_models = explorer.search()


def find_generate_measure_for_pi(
        explorer: Explorer,
        connection_cfg: ConnectionConfig,
        hwnas_cfg: HWNASConfig
) -> Metrics:
    explorer.choose_target_hw("rpi5")
    explorer.generate_search_space()
    top_models = explorer.search(hwnas_cfg)

    explorer.hw_setup_on_target(connection_conf=connection_cfg)
    measurements_latency_mean = []
    measurements_accuracy = []



    #Creating Train and Test set from MNIST #TODO build a generic dataclass/datawrapper
    transf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trainloader = DataLoader(
        MNIST("data/mnist", download=True, transform=transf),
        batch_size=64,
        shuffle=True,
    )
    testloader = DataLoader(
        MNIST("data/mnist", download=True, train=False, transform=transf), batch_size=64
    )

    retrain_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for i, model in enumerate(top_models):
        mlp_trainer = MLPTrainer(device=retrain_device, optimizer= torch.optim.Adam(model.parameters(), lr=1e-3))
        mlp_trainer.train(model, trainloader=trainloader, epochs=3)
        mlp_trainer.test(model, testloader=testloader)
        model_name = "ts_model_" + str(i) + ".pt"
        data_path = str(ROOT_DIR) + "/data"
        explorer.generate_for_hw_platform(model, model_name)

        mean = explorer.run_latency_measurement(model_name)
        measurements_latency_mean.append(mean)
        measurements_accuracy.append(
            explorer.run_accuracy_measurement(model_name, data_path)
        )

    floats = [float(np_float) for np_float in measurements_latency_mean]
    df = build_search_space_measurements_file(floats, explorer.metric_dir / "metrics.json",
                                               explorer.model_dir / "models.json",
                                               explorer.experiment_dir / "experiment_data.csv")
    logger.info("Models:\n %s", df)

    return Metrics(
        explorer.metric_dir / "metrics.json",
        explorer.model_dir / "models.json",
        measurements_accuracy,
        measurements_latency_mean,
    )


def measure_latency(knowledge_repository: KnowledgeRepository, explorer: Explorer, model_name: str):

    explorer.choose_target_hw("rpi5")
    explorer.hw_setup_on_target()

    mean, std = explorer.run_latency_measurement(model_name=model_name)
    logger.info("Mean Latency: %.2f", mean)
    logger.info("Std Latency: %.2f", std)


def measure_accuracy(knowledge_repository: KnowledgeRepository, explorer: Explorer, model_name: str):
    explorer.choose_target_hw("rpi5")
    explorer.hw_setup_on_target()
    data_path = str(ROOT_DIR) + "/data"
    logger.info(
        "Accuracy: %.2f",
        explorer.run_accuracy_measurement(model_name, data_path),
    )


def prepare_pi():
    hw_manager = PIHWManager()
    hw_manager.compile_code()


if __name__ == "__main__":

    hwnas_cfg = HWNASConfig(config_path="configs/hwnas_config.yaml")
    connection_cfg = ConnectionConfig(config_path="configs/connection_config.yaml")
    model_cfg = ModelConfig(config_path="configs/model_config.yaml")

    knowledge_repo = setup_knowledge_repository()
    explorer = Explorer(knowledge_repo)
    explorer.set_model_cfg(model_cfg)

    metry = find_generate_measure_for_pi(explorer, connection_cfg, hwnas_cfg)
    visu = Visualizer(metry, explorer.plot_dir)
    visu.plot_all_results(filename="plot.png")


