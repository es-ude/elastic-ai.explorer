import logging
import os
from logging import config

import nni
import torch
import yaml

from elasticai.explorer.data_to_csv import build_search_space_measurements_file
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import (
    KnowledgeRepository,
    HWPlatform,
    Metrics,
)
from elasticai.explorer.platforms.deployment.manager import PIHWManager, ConnectionData
from elasticai.explorer.platforms.generator.generator import PIGenerator
from elasticai.explorer.train_model import train, test
from elasticai.explorer.visualizer import Visualizer
from elasticai.explorer.config import Config, ExperimentConfig
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


def find_for_pi(knowledge_repository: KnowledgeRepository, max_search_trials: int, top_k: int):
    explorer = Explorer(knowledge_repository)
    explorer.choose_target_hw("rpi5")
    explorer.generate_search_space()
    top_models = explorer.search(max_search_trials, top_k)


def find_generate_measure_for_pi( 
        explorer: Explorer,
        device_connection: ConnectionData,
) -> Metrics:
    explorer.choose_target_hw("rpi5")
    explorer.generate_search_space()
    top_models = explorer.search()

    explorer.hw_setup_on_target(device_connection)
    measurements_latency_mean = []
    measurements_accuracy = []

    for i, model in enumerate(top_models):
        train(model, 3, device=host_device)
        test(model, device= host_device)
        model_path = str(ROOT_DIR) + "/models/ts_models/model_" + str(i) + ".pt"
        data_path = str(ROOT_DIR) + "/data"
        explorer.generate_for_hw_platform(model, model_path)

        mean = explorer.run_latency_measurement(device_connection, model_path)
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


def measure_latency(knowledge_repository: KnowledgeRepository, connection_data: ConnectionData):
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
    config = Config(config_path="config.yaml")

    #Changing config in code possible 
    config.experiment_conf.max_search_trials = 2

    knowledge_repo = setup_knowledge_repository()
    explorer = Explorer(knowledge_repo, config=config)
    device_connection = ConnectionData(explorer.connection_conf.target_host, explorer.connection_conf.target_user)
    metry = find_generate_measure_for_pi(
        explorer, device_connection
    )
    visu = Visualizer(metry)
    visu.plot_all_results(filename="plot")
