import logging.config
import shutil
from pathlib import Path

import torch
from torch.optim.adam import Adam
from torchvision.transforms import transforms

from elasticai.explorer.config import DeploymentConfig, HWNASConfig
from elasticai.explorer.training.data import DatasetSpecification, MNISTWrapper
from elasticai.explorer.utils.data_to_csv import build_search_space_measurements_file
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import (
    KnowledgeRepository,
    HWPlatform,
)
from elasticai.explorer.platforms.deployment.compiler import Compiler
from elasticai.explorer.platforms.deployment.device_communication import Host
from elasticai.explorer.platforms.deployment.manager import PIHWManager, Metric
from elasticai.explorer.platforms.generator.generator import PIGenerator
from elasticai.explorer.training.trainer import SupervisedTrainer

logging.config.fileConfig("logging.conf", disable_existing_loggers=False)

logger = logging.getLogger("explorer.main")


def setup_knowledge_repository_pi() -> KnowledgeRepository:
    knowledge_repository = KnowledgeRepository()
    knowledge_repository.register_hw_platform(
        HWPlatform(
            "rpi5",
            "Raspberry PI 5 with A76 processor and 8GB RAM",
            PIGenerator,
            PIHWManager,
            Host,
            Compiler,
        )
    )

    knowledge_repository.register_hw_platform(
        HWPlatform(
            "rpi4",
            "Raspberry PI 4 with A72 processor and 4GB RAM",
            PIGenerator,
            PIHWManager,
            Host,
            Compiler,
        )
    )
    return knowledge_repository


def setup_mnist(path_to_test_data: Path):
    transf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    shutil.make_archive(
        str(path_to_test_data), "zip", f"{str(path_to_test_data)}/MNIST/raw"
    )
    dataset_spec = DatasetSpecification(MNISTWrapper, path_to_test_data, transf)
    return dataset_spec


def find_generate_measure_for_pi(
    explorer: Explorer,
    deploy_cfg: DeploymentConfig,
    hwnas_cfg: HWNASConfig,
    search_space_path: Path,
):
    explorer.choose_target_hw(deploy_cfg)
    explorer.generate_search_space(search_space_path)

    path_to_test_data = Path("data/mnist")
    dataset_spec = setup_mnist(path_to_test_data)

    top_models = explorer.search(hwnas_cfg, dataset_spec, SupervisedTrainer)
    explorer.hw_setup_on_target(Path(str(path_to_test_data) + ".zip"))
    latency_measurements = []
    accuracy_measurements = []

    retrain_device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    for i, model in enumerate(top_models):
        mlp_trainer = SupervisedTrainer(
            device=retrain_device,
            optimizer=Adam(model.parameters(), lr=1e-3),
            dataset_spec=dataset_spec,
        )
        mlp_trainer.train(model, epochs=6)
        mlp_trainer.test(model)
        model_name = "ts_model_" + str(i) + ".pt"
        explorer.generate_for_hw_platform(model, model_name)

        latency = explorer.run_measurement(Metric.LATENCY, model_name)
        latency_measurements.append(latency)
        accuracy_measurements.append(
            explorer.run_measurement(Metric.ACCURACY, model_name)
        )

    latencies = [latency["Latency"]["value"] for latency in latency_measurements]
    accuracies = [accuracy["Accuracy"]["value"] for accuracy in accuracy_measurements]
    df = build_search_space_measurements_file(
        latencies,
        accuracies,
        explorer.metric_dir / "metrics.json",
        explorer.model_dir / "models.json",
        explorer.experiment_dir / "experiment_data.csv",
    )
    logger.info("Models:\n %s", df)


def search_models(explorer: Explorer, hwnas_cfg: HWNASConfig, search_space):
    deploy_cfg = DeploymentConfig(config_path=Path("configs/deployment_config.yaml"))
    explorer.choose_target_hw(deploy_cfg)
    explorer.generate_search_space(search_space)
    path_to_test_data = Path("data/mnist")
    dataset_spec = setup_mnist(path_to_test_data)

    top_models = explorer.search(hwnas_cfg, dataset_spec, ClassificationTrainer)

    retrain_device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    for i, model in enumerate(top_models):
        print(f"found model {i}:  {model}")

        mlp_trainer = ClassificationTrainer(
            device=retrain_device,
            optimizer=Adam(model.parameters(), lr=1e-3),
            dataset_spec=dataset_spec,
        )
        mlp_trainer.train(model, epochs=3)
        mlp_trainer.test(model)
        print("=================================================")
        model_name = "ts_model_" + str(i) + ".pt"

        explorer.generate_for_hw_platform(model, model_name)


if __name__ == "__main__":
    hwnas_cfg = HWNASConfig(config_path=Path("configs/hwnas_config.yaml"))
    deploy_cfg = DeploymentConfig(config_path=Path("configs/deployment_config.yaml"))
    knowledge_repo = setup_knowledge_repository_pi()
    explorer = Explorer(knowledge_repo)

    search_space = Path("elasticai/explorer/hw_nas/search_space/search_space.yaml")

    find_generate_measure_for_pi(
        explorer=explorer,
        deploy_cfg=deploy_cfg,
        hwnas_cfg=hwnas_cfg,
        search_space_path=search_space,
    )
