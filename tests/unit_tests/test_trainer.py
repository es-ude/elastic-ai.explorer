from pathlib import Path
from typing import Any, Callable
import torch.nn as nn
import torch

from elasticai.explorer.training.data import BaseDataset, DatasetSpecification
from elasticai.explorer.training.trainer import ReconstructionAutoencoderTrainer

INPUT_DIM = 10
SEQ_LENGTH = 3
BATCH_SIZE = 64


class DatasetExample(BaseDataset):

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.test_data = torch.randn(BATCH_SIZE, SEQ_LENGTH, INPUT_DIM)

    def __getitem__(self, idx) -> Any:
        return self.test_data[idx]

    def __len__(self) -> int:
        return len(self.test_data)


# LSTM Autoencoder
class SimpleLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)
        self.info = "LSTMAutoencoder"

    def forward(self, x):

        _, (hidden, _) = self.encoder(x)
        hidden = hidden.repeat(x.size(1), 1, 1).permute(
            1, 0, 2
        )  # Repeat for each timestep
        reconstructed, _ = self.decoder(hidden)
        return reconstructed


class TestAutoencoderTrainer:
    def setup_class(self):
        self.dataset_spec = DatasetSpecification(
            dataset=DatasetExample(),
        )
        self.autoencoder = SimpleLSTMAutoencoder(INPUT_DIM)

    def test_autoencoder_trainer(self):

        autoencoder_trainer = ReconstructionAutoencoderTrainer(
            "cpu",
            dataset_spec=self.dataset_spec,
        )
        autoencoder_trainer.configure_optimizer(torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3))  # type: ignore
        autoencoder_trainer.train(self.autoencoder, 20)
        assert autoencoder_trainer.validate(self.autoencoder)[1] > 0
        assert autoencoder_trainer.test(self.autoencoder)[1] > 0
