import json
import logging.config
from pathlib import Path

import torch


from elasticai.explorer.config import DeploymentConfig, HWNASConfig
from elasticai.explorer.generator.generator import Generator
from elasticai.explorer.generator.model_builder.model_builder import CreatorModelBuilder
from elasticai.explorer.hw_nas.search_space.quantization import (
    FixedPointInt8Scheme,
)

from elasticai.explorer.utils.data_to_csv import build_search_space_measurements_file
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import (
    KnowledgeRepository,
)
from elasticai.explorer.generator.deployment.compiler import ENv5Compiler
from elasticai.explorer.generator.deployment.device_communication import ENv5Host
from elasticai.explorer.generator.deployment.hw_manager import ENv5HWManager, Metric
from elasticai.explorer.generator.model_compiler.model_compiler import (
    CreatorModelCompiler,
)

from elasticai.explorer.training.trainer import MLPTrainer
from tests.system_tests.test_env5_measurements import create_example_dataset_spec

logging.config.fileConfig("logging.conf", disable_existing_loggers=False)

logger = logging.getLogger("explorer.main")





def setup_knowledge_repository_env5() -> KnowledgeRepository:
    knowledge_repository = KnowledgeRepository()
    knowledge_repository.register_hw_platform(
        Generator(
            "env5",
            "Env5 with RP2040 and xc7s50ftgb196-2 FPGA",
            CreatorModelCompiler,
            ENv5HWManager,
            ENv5Host,
            ENv5Compiler,
        )
    )
    return knowledge_repository


def find_generate_measure_for_env5(
    explorer: Explorer,
    deploy_cfg: DeploymentConfig,
    hwnas_cfg: HWNASConfig,
    search_space: Path,
    retrain_epochs: int = 4,
):
    explorer.choose_target_hw(deploy_cfg)
    explorer.generate_search_space(search_space)

    quantization_scheme = FixedPointInt8Scheme()
    dataset_spec = create_example_dataset_spec(quantization_scheme)
    top_models = explorer.search(
        hwnas_cfg, dataset_spec, MLPTrainer, CreatorModelBuilder()
    )

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
        model_name = "creator_model_" + str(i)
        explorer.generate_for_hw_platform(model, model_name, dataset_spec)

        metric_to_source = {Metric.ACCURACY: explorer.model_dir / model_name}
        explorer.hw_setup_on_target(metric_to_source, dataset_spec, quantization_scheme)

        accuracy_on_device = explorer.run_measurement(
            Metric.ACCURACY, model_name
        )

        accuracy_after_retrain_dict = json.loads(
            '{"Accuracy after retrain": { "value":'
            + str(accuracy_after_retrain_value)
            + ' , "unit": "percent"}}'
        )
        accuracy_measurements_on_device.append(accuracy_on_device)
        accuracy_after_retrain.append(accuracy_after_retrain_dict)
    accuracies_on_device = [
        accuracy["Accuracy"]["value"] for accuracy in accuracy_measurements_on_device
    ]
    accuracy_after_retrain = [
        accuracy["Accuracy after retrain"]["value"]
        for accuracy in accuracy_after_retrain
    ]

    df = build_search_space_measurements_file(
        [i for i in range(0, len(accuracies_on_device), 1)],
        accuracy_after_retrain,
        accuracies_on_device,
        explorer.metric_dir / "metrics.json",
        explorer.model_dir / "models.json",
        explorer.experiment_dir / "experiment_data.csv",
    )
    logger.info("Models:\n %s", df)


if __name__ == "__main__":
    hwnas_cfg = HWNASConfig(config_path=Path("configs/env5/hwnas_config.yaml"))
    deploy_cfg = DeploymentConfig(
        config_path=Path("configs/env5/deployment_config.yaml")
    )

    knowledge_repo = setup_knowledge_repository_env5()
    explorer = Explorer(knowledge_repo)
    search_space = Path("configs/env5/search_space.yaml")
    retrain_epochs = 3
    find_generate_measure_for_env5(
        explorer, deploy_cfg, hwnas_cfg, search_space, retrain_epochs
    )
