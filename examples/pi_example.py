import json
import logging.config
from pathlib import Path

import torch
from torch import optim, nn
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
from elasticai.explorer.platforms.deployment.compiler import RPICompiler
from elasticai.explorer.platforms.deployment.device_communication import RPiHost
from elasticai.explorer.platforms.deployment.hw_manager import RPiHWManager, Metric
from elasticai.explorer.platforms.generator.generator import RPiGenerator
from elasticai.explorer.training.trainer import (
    SupervisedTrainer,
    accuracy_fn,
)
from settings import ROOT_DIR

logging.config.fileConfig(ROOT_DIR / "logging.conf", disable_existing_loggers=False)

logger = logging.getLogger("explorer.main")


def setup_knowledge_repository_pi() -> KnowledgeRepository:
    knowledge_repository = KnowledgeRepository()
    knowledge_repository.register_hw_platform(
        HWPlatform(
            "rpi5",
            "Raspberry PI 5 with A76 processor and 8GB RAM",
            RPiGenerator,
            RPiHWManager,
            RPiHost,
            RPICompiler,
        )
    )

    knowledge_repository.register_hw_platform(
        HWPlatform(
            "rpi4",
            "Raspberry PI 4 with A72 processor and 4GB RAM",
            RPiGenerator,
            RPiHWManager,
            RPiHost,
            RPICompiler,
        )
    )
    return knowledge_repository


def setup_mnist(path_to_test_data: Path):
    transf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset_spec = DatasetSpecification(
        dataset=MNISTWrapper(root=path_to_test_data, transform=transf),
        deployable_dataset_path=path_to_test_data,
    )
    return dataset_spec


def find_generate_measure_for_pi(
    explorer: Explorer,
    deploy_cfg: DeploymentConfig,
    hwnas_cfg: HWNASConfig,
    search_space_path: Path,
    retrain_epochs: int = 4,
):
    explorer.choose_target_hw(deploy_cfg)
    explorer.generate_search_space(search_space_path)

    path_to_test_data = ROOT_DIR / Path("data/mnist")
    dataset_spec = setup_mnist(path_to_test_data)

    trainer = SupervisedTrainer(
        hwnas_cfg.host_processor,
        dataset_spec=dataset_spec,
        loss_fn=nn.CrossEntropyLoss(),
        batch_size=65,
        extra_metrics={"accuracy": accuracy_fn},
    )
    top_models = explorer.search(hwnas_cfg, trainer)
    metric_to_source = {
        Metric.ACCURACY: Path("code/measure_accuracy_mnist.cpp"),
        Metric.LATENCY: Path("code/measure_latency.cpp"),
    }
    explorer.hw_setup_on_target(metric_to_source, dataset_spec)
    latency_measurements = []
    accuracy_measurements = []
    accuracy_after_retrain = []

    retrain_device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    for i, model in enumerate(top_models):
        mlp_trainer = SupervisedTrainer(
            device=retrain_device,
            dataset_spec=dataset_spec,
            loss_fn=nn.CrossEntropyLoss(),
            batch_size=64,
            extra_metrics={"accuracy": accuracy_fn},
        )
        mlp_trainer.configure_optimizer(optimizer=Adam(model.parameters(), lr=1e-3))
        mlp_trainer.train(model, epochs=retrain_epochs)
        test_metrics, _ = mlp_trainer.test(model)
        model_name = "ts_model_" + str(i) + ".pt"
        explorer.generate_for_hw_platform(model, model_name, dataset_spec)

        latency = explorer.run_measurement(Metric.LATENCY, model_name)
        latency_measurements.append(latency)
        accuracy_measurements.append(
            explorer.run_measurement(Metric.ACCURACY, model_name)
        )
        accuracy_after_retrain_dict = json.loads(
            '{"Accuracy after retrain": { "value":'
            + (str(test_metrics["accuracy"]))
            + ' , "unit": "percent"}}'
        )

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
        accuracies_on_device,
        accuracies_after_retrain,
        explorer.metric_dir / "metrics.json",
        explorer.model_dir / "models.json",
        explorer.experiment_dir / "experiment_data.csv",
    )
    logger.info("Models:\n %s", df)


def search_models(explorer: Explorer, hwnas_cfg: HWNASConfig, search_space):
    retrain_device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    deploy_cfg = DeploymentConfig(config_path=Path("configs/deployment_config.yaml"))
    explorer.choose_target_hw(deploy_cfg)
    explorer.generate_search_space(search_space)
    path_to_test_data = ROOT_DIR / Path("data/mnist")
    dataset_spec = setup_mnist(path_to_test_data)
    top_models = explorer.search(
        hwnas_cfg,
        SupervisedTrainer(
            device=retrain_device,
            dataset_spec=dataset_spec,
            extra_metrics={"accuracy": accuracy_fn},
        ),
    )


    for i, model in enumerate(top_models):
        print(f"found model {i}:  {model}")

        mlp_trainer = SupervisedTrainer(
            device=retrain_device,
            dataset_spec=dataset_spec,
            extra_metrics={"accuracy": accuracy_fn},
        )
        mlp_trainer.configure_optimizer(optimizer=Adam(model.parameters(), lr=1e-3))
        mlp_trainer.train(model, epochs=3)
        mlp_trainer.test(model)
        print("=================================================")
        model_name = "ts_model_" + str(i) + ".pt"

        explorer.generate_for_hw_platform(model, model_name, dataset_spec)


if __name__ == "__main__":
    hwnas_cfg = HWNASConfig(config_path=Path(ROOT_DIR / "configs/hwnas_config.yaml"))
    deploy_cfg = DeploymentConfig(
        config_path=ROOT_DIR / Path("configs/deployment_config.yaml")
    )
    knowledge_repo = setup_knowledge_repository_pi()
    explorer = Explorer(knowledge_repo)

    search_space = Path(
        ROOT_DIR / "elasticai/explorer/hw_nas/search_space/search_space.yaml"
    )

    find_generate_measure_for_pi(
        explorer=explorer,
        deploy_cfg=deploy_cfg,
        hwnas_cfg=hwnas_cfg,
        search_space_path=search_space,
        retrain_epochs=3,
    )
