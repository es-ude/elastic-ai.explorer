import json
import logging.config
from math import log10
from pathlib import Path
import torch
from torch.optim.adam import Adam
from torchvision.transforms import transforms

from elasticai.explorer.hw_nas.constraints import ConstraintRegistry
from elasticai.explorer.hw_nas.estimators import AccuracyEstimator, FLOPsEstimator
from elasticai.explorer.hw_nas.hw_nas import HWNASParameters
from elasticai.explorer.training.data import DatasetSpecification, MNISTWrapper
from elasticai.explorer.utils.data_to_csv import build_search_space_measurements_file
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import (
    KnowledgeRepository,
    HWPlatform,
)
from elasticai.explorer.platforms.deployment.compiler import DockerParams, RPICompiler
from elasticai.explorer.platforms.deployment.device_communication import (
    RPiHost,
    SSHParams,
)
from elasticai.explorer.platforms.deployment.hw_manager import RPiHWManager, Metric
from elasticai.explorer.platforms.generator.generator import RPiGenerator
from elasticai.explorer.training.trainer import MLPTrainer
from settings import ROOT_DIR

logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
logger = logging.getLogger("explorer.main")
device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


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
        dataset_type=MNISTWrapper,
        dataset_location=path_to_test_data,
        deployable_dataset_path=path_to_test_data,
        transform=transf,
    )
    return dataset_spec


def setup_constraints(dataset_spec) -> ConstraintRegistry:
    constr_reg = ConstraintRegistry()

    accuracy_estimator = AccuracyEstimator(MLPTrainer, dataset_spec, 3, device=device)

    data_sample = torch.randn((1, 1, 28, 28), dtype=torch.float32, device=device)

    constr_reg.register_soft_constraint(estimator=accuracy_estimator, is_reward=True)

    constr_reg.register_soft_constraint(
        estimator=FLOPsEstimator(data_sample), estimate_transform=log10, weight=2.0
    )
    return constr_reg


def find_generate_measure_for_pi(
    explorer: Explorer,
    ssh_params: SSHParams,
    docker_params: DockerParams,
    search_space_path: Path,
    retrain_epochs: int = 4,
    max_search_trials: int = 4,
    top_n_models: int = 2,
):
    explorer.choose_target_hw("rpi5", docker_params, ssh_params)
    explorer.generate_search_space(search_space_path)

    path_to_test_data = ROOT_DIR / Path("data/mnist")
    dataset_spec = setup_mnist(path_to_test_data)
    constr_reg = setup_constraints(dataset_spec=dataset_spec)
    top_models = explorer.search(
        constraint_registry=constr_reg,
        hw_nas_parameters=HWNASParameters(
            max_search_trials=max_search_trials, top_n_models=top_n_models
        ),
    )
    metric_to_source = {
        Metric.ACCURACY: Path("code/measure_accuracy_mnist.cpp"),
        Metric.LATENCY: Path("code/measure_latency.cpp"),
    }
    explorer.hw_setup_on_target(metric_to_source, dataset_spec)
    latency_measurements = []
    accuracy_measurements = []
    accuracy_after_retrain = []

    for i, model in enumerate(top_models):
        mlp_trainer = MLPTrainer(
            device=device,
            optimizer=Adam(model.parameters(), lr=1e-3),
            dataset_spec=dataset_spec,
        )
        mlp_trainer.train(model, epochs=retrain_epochs)
        accuracy_after_retrain_value, _ = mlp_trainer.test(model)
        model_name = "ts_model_" + str(i) + ".pt"
        explorer.generate_for_hw_platform(model, model_name, dataset_spec)

        latency = explorer.run_measurement(Metric.LATENCY, model_name)
        latency_measurements.append(latency)
        accuracy_measurements.append(
            explorer.run_measurement(Metric.ACCURACY, model_name)
        )

        accuracy_after_retrain_dict = json.loads(
            '{"Accuracy after retrain": { "value":'
            + (str(accuracy_after_retrain_value))
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
        accuracies_on_device,
        accuracies_after_retrain,
        explorer.metric_dir / "metrics.json",
        explorer.model_dir / "models.json",
        explorer.experiment_dir / "experiment_data.csv",
    )
    logger.info("Models:\n %s", df)


def search_models(
    explorer: Explorer, ssh_params: SSHParams, docker_params: DockerParams, search_space
):
    explorer.choose_target_hw(
        "rpi5", communication_params=ssh_params, docker_params=docker_params
    )
    explorer.generate_search_space(search_space)
    path_to_test_data = ROOT_DIR / Path("data/mnist")
    dataset_spec = setup_mnist(path_to_test_data)

    top_models = explorer.search()

    retrain_device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    for i, model in enumerate(top_models):
        print(f"found model {i}:  {model}")

        mlp_trainer = MLPTrainer(
            device=retrain_device,
            optimizer=Adam(model.parameters(), lr=1e-3),
            dataset_spec=dataset_spec,
        )
        mlp_trainer.train(model, epochs=3)
        mlp_trainer.test(model)
        print("=================================================")
        model_name = "ts_model_" + str(i) + ".pt"

        explorer.generate_for_hw_platform(model, model_name, dataset_spec)


if __name__ == "__main__":
    ssh_params = SSHParams(
        hostname="<hostname>", username="<username>"
    )  # <-- Setup for your RPi
    docker_params = DockerParams()  # <-- configure this only if necessary
    knowledge_repo = setup_knowledge_repository_pi()
    explorer = Explorer(knowledge_repo)

    search_space = Path("elasticai/explorer/hw_nas/search_space/search_space.yaml")

    find_generate_measure_for_pi(
        explorer=explorer,
        ssh_params=ssh_params,
        docker_params=docker_params,
        search_space_path=search_space,
        retrain_epochs=3,
        max_search_trials=4,
        top_n_models=2,
    )
