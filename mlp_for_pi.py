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
from elasticai.explorer.platforms.deployment.manager import PIHWManager, ConnectionData
from elasticai.explorer.platforms.generator.generator import PIGenerator
from elasticai.explorer.trainer import MLPTrainer
from elasticai.explorer.visualizer import Visualizer
from settings import ROOT_DIR

nni.enable_global_logging(False)
logging.config.fileConfig("logging.conf", disable_existing_loggers=False)

logger = logging.getLogger("explorer.main")


def setup_knowledge_repository():
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


def find_for_pi(knowledge_repository, max_search_trials, top_k):
    explorer = Explorer(knowledge_repository)
    explorer.choose_target_hw("rpi5")
    explorer.generate_search_space()
    top_models = explorer.search(max_search_trials, top_k)


def find_generate_measure_for_pi(
        knowledge_repository, device_connection, max_search_trials, top_k
) -> Metrics:
    explorer = Explorer(knowledge_repository)
    explorer.choose_target_hw("rpi5")
    explorer.generate_search_space()
    top_models = explorer.search(max_search_trials, top_k)

    explorer.hw_setup_on_target(device_connection)
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

    for i, model in enumerate(top_models):
        mlp_trainer = MLPTrainer(device="cpu", optimizer= torch.optim.Adam(model.parameters(), lr=1e-3))
        mlp_trainer.train(model, trainloader=trainloader, epochs=3)
        mlp_trainer.test(model, testloader=testloader)
        model_path = str(ROOT_DIR) + "/models/ts_models/model_" + str(i) + ".pt"
        data_path = str(ROOT_DIR) + "/data"
        explorer.generate_for_hw_platform(model, model_path)

        mean, _ = explorer.run_latency_measurement(device_connection, model_path)
        measurements_latency_mean.append(mean)
        measurements_accuracy.append(
            explorer.run_accuracy_measurement(device_connection, model_path, data_path)
        )

    floats = [float(np_float) for np_float in measurements_latency_mean]
    df = build_search_space_measurements_file(floats)
    logger.info("Models:\n %s", df)

    return Metrics(
        "metrics/metrics.json",
        "models/models.json",
        measurements_accuracy,
        measurements_latency_mean,
    )


def measure_latency(knowledge_repository, connection_data):
    explorer = Explorer(knowledge_repository)
    explorer.choose_target_hw("rpi5")
    model_path = str(ROOT_DIR) + "/models/ts_models/model_0.pt"
    explorer.hw_setup_on_target(connection_data)

    mean, std = explorer.run_latency_measurement(connection_data, model_path)
    logger.info("Mean Latency: %.2f", mean)
    logger.info("Std Latency: %.2f", std)


def measure_accuracy(knowledge_repository, connection_data):
    explorer = Explorer(knowledge_repository)
    explorer.choose_target_hw("rpi5")
    explorer.hw_setup_on_target(connection_data)
    model_path = str(ROOT_DIR) + "/models/ts_models/model_0.pt"
    data_path = str(ROOT_DIR) + "/data"

    logger.info(
        "Accuracy: %.2f",
        explorer.run_accuracy_measurement(connection_data, model_path, data_path),
    )


def prepare_pi():
    hw_manager = PIHWManager()
    hw_manager.compile_code()


def make_dirs_if_not_exists():
    if not os.path.exists("plots"):
        os.makedirs("plots")
    if not os.path.exists("metrics"):
        os.makedirs("metrics")


if __name__ == "__main__":
    make_dirs_if_not_exists()

    host = "transfair.local"
    user = "robin"
    # 60 possible
    # ca hälfte über 90%
    # 1/3 der modelle unter 70
    # 1/4 der Modelle unter 50
    max_search_trials = 2
    top_k = 2
    knowledge_repo = setup_knowledge_repository()
    device_connection = ConnectionData(host, user)
    metry = find_generate_measure_for_pi(
        knowledge_repo, device_connection, max_search_trials, top_k
    )
    visu = Visualizer(metry)
    visu.plot_all_results(filename="plot")
