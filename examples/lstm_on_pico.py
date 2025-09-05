
import logging
import logging.config
from pathlib import Path

import torch
from elasticai.explorer.config import DeploymentConfig
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import HWPlatform, KnowledgeRepository
from elasticai.explorer.platforms.deployment.compiler import PicoCompiler
from elasticai.explorer.platforms.deployment.device_communication import PicoHost
from elasticai.explorer.platforms.deployment.hw_manager import Metric, PicoHWManager
from elasticai.explorer.platforms.generator.generator import PicoGenerator
import torch.nn as nn

from elasticai.explorer.training.data import BaseDataset, DatasetSpecification
from settings import ROOT_DIR

logging.config.fileConfig("logging.conf", disable_existing_loggers=False)

logger = logging.getLogger("explorer.lstm_on_pico")

# LSTM Autoencoder
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, *args, **kwargs):
        super(LSTMAutoencoder, self).__init__(*args, **kwargs)
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):

        _, (hidden, _) = self.encoder(x)
        hidden = hidden.repeat(x.size(1), 1, 1).permute(
            1, 0, 2
        )  # Repeat for each timestep
        reconstructed, _ = self.decoder(hidden)
        return reconstructed


class TensorOnesDataset(BaseDataset):
    def __init__(
        self, root: str = "", transform=None, target_transform=None, *args, **kwargs
    ):
        super().__init__(root, transform, target_transform, *args, **kwargs)

    def __len__(self):
        return 128

    def __getitem__(self, idx):
        return torch.ones((1,5,1)), 1


def setup_for_pico():
    knowledge_repository = KnowledgeRepository()
    knowledge_repository.register_hw_platform(
        HWPlatform(
            "pico",
            "Pico mit RP2040",
            PicoGenerator,
            PicoHWManager,
            PicoHost,
            PicoCompiler,
        )
    )
    return knowledge_repository


if __name__ == "__main__":
    deploy_cfg = DeploymentConfig(
        config_path=ROOT_DIR / Path("configs/pico_lstm/deployment_config.yaml")
    )

    knowledge_repository = setup_for_pico()
    explorer = Explorer(knowledge_repository, "lstm_on_pico")
    model = LSTMAutoencoder(input_dim=1)

    explorer.choose_target_hw(deploy_cfg)
    model_name = "lstm_model.tflite"

    dataset_spec = DatasetSpecification(
        dataset_type=TensorOnesDataset,
        dataset_location=Path("data/empty_dataset"),
        deployable_dataset_path=Path("data/empty_dataset"),
        transform=None,
    )
    explorer.generate_for_hw_platform(
        model=model, model_name=model_name, dataset_spec=dataset_spec
    )

    metric_to_source = {
            Metric.ACCURACY: Path("code/pico_crosscompiler/lstm_on_pico"),
        }
    explorer.hw_setup_on_target(metric_to_source, dataset_spec)

    explorer.run_measurement(Metric.ACCURACY, model_name)
