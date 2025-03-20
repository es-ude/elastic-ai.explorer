import logging
import os
from logging import config
from pathlib import Path

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
from elasticai.explorer.platforms.deployment.manager import PIHWManager
from elasticai.explorer.platforms.generator.generator import PIGenerator
from elasticai.explorer.train_model import train, test
from elasticai.explorer.visualizer import Visualizer
from elasticai.explorer.config import Config, ConnectionConfig, HWNASConfig, ModelConfig
from settings import ROOT_DIR
config = None

nni.enable_global_logging(False)
logging.config.fileConfig("logging.conf", disable_existing_loggers=False)


logger = logging.getLogger("explorer.main")


def setup_knowledge_repository_pi5() -> KnowledgeRepository:
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

def setup_knowledge_repository_pi4():
    knowledge_repository = KnowledgeRepository()
    knowledge_repository.register_hw_platform(
        HWPlatform(
            "rpi4",
            "Raspberry PI 4 with A72 processor and 4GB RAM",
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
        hwnas_cfg: HWNASConfig,
        host_path_to_libtorch="./code/libtorch"
) -> Metrics:
    explorer.choose_target_hw("rpi5")
    explorer.generate_search_space()
    top_models = explorer.search(hwnas_cfg)

    explorer.hw_setup_on_target(connection_conf=connection_cfg, host_path_to_libtorch=host_path_to_libtorch)
    measurements_latency_mean = []
    measurements_accuracy = []

    for i, model in enumerate(top_models):
        train(model, 3, device = hwnas_cfg.host_processor)
        test(model, device= hwnas_cfg.host_processor)
        model_name = "ts_model_" + str(i) + ".pt"
        data_path = ROOT_DIR / "data"
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


def measure_latency(knowledge_repository: KnowledgeRepository, explorer: Explorer, model_name: str, connection_cfg: ConnectionConfig, path_to_libtorch: Path="./code/libtorch",
    pi_type="rpi5"):
    
    explorer.choose_target_hw(pi_type)
    explorer.hw_setup_on_target(connection_conf=connection_cfg, host_path_to_libtorch=path_to_libtorch)

    mean, std = explorer.run_latency_measurement(model_name=model_name)
    logger.info("Mean Latency: %.2f", mean)
    logger.info("Std Latency: %.2f", std)


def measure_accuracy(knowledge_repository: KnowledgeRepository, explorer: Explorer, model_name:str, connection_cfg: ConnectionConfig, path_to_libtorch: Path ="./code/libtorch", pi_type: str="rpi5"):
    explorer = Explorer(knowledge_repository)
    explorer.choose_target_hw(pi_type)
    explorer.hw_setup_on_target(connection_cfg, path_to_libtorch)
    data_path = str(ROOT_DIR) + "/data"
    logger.info(
        "Accuracy: %.2f",
        explorer.run_accuracy_measurement(model_name, data_path),
    )

def prepare_pi5():
    hw_manager = PIHWManager()
    hw_manager.compile_code()


if __name__ == "__main__": 
    host_path_to_libtorch = "./code/libtorch"
    hwnas_cfg = HWNASConfig(config_path="configs/hwnas_config.yaml")
    connection_cfg = ConnectionConfig(config_path="configs/connection_config.yaml")
    model_cfg = ModelConfig(config_path="configs/model_config.yaml")

    knowledge_repo = setup_knowledge_repository_pi5()
    explorer = Explorer(knowledge_repo)
    explorer.set_model_cfg(model_cfg)

    metry = find_generate_measure_for_pi(explorer, connection_cfg, hwnas_cfg, host_path_to_libtorch=host_path_to_libtorch)
    visu = Visualizer(metry, explorer.plot_dir)
    visu.plot_all_results(filename="plot.png")


