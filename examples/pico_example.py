import json
import logging
import logging.config
from pathlib import Path

import torch
from torchvision.transforms import transforms

from elasticai.explorer.config import DeploymentConfig, HWNASConfig
from elasticai.explorer.explorer import Explorer

from elasticai.explorer.knowledge_repository import (
    KnowledgeRepository,
    HWPlatform,
)
from elasticai.explorer.platforms.deployment.compiler import PicoCompiler
from elasticai.explorer.platforms.deployment.device_communication import (
    PicoHost,
)
from elasticai.explorer.platforms.deployment.hw_manager import (
    PicoHWManager,
    Metric,
)
from elasticai.explorer.platforms.generator.generator import (
    PicoGenerator,
)

from elasticai.explorer.training.data import DatasetSpecification, MNISTWrapper
from elasticai.explorer.training.trainer import MLPTrainer
from elasticai.explorer.utils.data_to_csv import build_search_space_measurements_file
from elasticai.explorer.utils.data_utils import setup_mnist_for_cpp
from settings import ROOT_DIR

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
    search_space: Path,
    retrain_epochs: int = 4,
):
    explorer.choose_target_hw(deploy_cfg)
    explorer.generate_search_space(search_space)

    transf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    path_to_dataset = ROOT_DIR / Path("data/mnist")
    root_dir_cpp_mnist = ROOT_DIR / Path("data/cpp-mnist")
    setup_mnist_for_cpp(path_to_dataset, root_dir_cpp_mnist, transf)
    dataset_spec = DatasetSpecification(
        dataset=MNISTWrapper(root=path_to_dataset, transform=transf),
        deployable_dataset_path=root_dir_cpp_mnist,
    )
    top_models = explorer.search(hwnas_cfg, dataset_spec, MLPTrainer)

    latency_measurements = []
    accuracy_measurements_on_device = []
    accuracy_after_retrain = []
    retrain_device = "cpu"
    for i, model in enumerate(top_models):
        mlp_trainer = MLPTrainer(
            device=retrain_device,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            dataset_spec=dataset_spec,
        )
        mlp_trainer.train(model, epochs=retrain_epochs)
        accuracy_after_retrain_value, _ = mlp_trainer.test(model)
        model_name = "ts_model_" + str(i) + ".tflite"
        explorer.generate_for_hw_platform(model, model_name, dataset_spec)

        metric_to_source = {
            Metric.ACCURACY: Path("code/pico_crosscompiler/measure_accuracy"),
            Metric.LATENCY: Path("code/pico_crosscompiler/measure_latency"),
        }
        explorer.hw_setup_on_target(metric_to_source, dataset_spec)

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
        accuracy_after_retrain,
        accuracies_on_device,
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

    knowledge_repo = setup_knowledge_repository_pico()
    explorer = Explorer(knowledge_repo)
    search_space = Path("elasticai/explorer/hw_nas/search_space/search_space.yaml")
    retrain_epochs = 3
    find_generate_measure_for_pico(
        explorer, deploy_cfg, hwnas_cfg, search_space, retrain_epochs
    )
