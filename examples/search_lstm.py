from pathlib import Path
from typing import Union, Optional, Callable

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Lambda, Compose
from torchvision.transforms import v2

from elasticai.explorer.config import HWNASConfig
from elasticai.explorer.hw_nas import hw_nas
from elasticai.explorer.hw_nas.search_space import architecture_components
from elasticai.explorer.hw_nas.search_space.architecture_components import LinearOne
from elasticai.explorer.hw_nas.search_space.utils import yaml_to_dict
from elasticai.explorer.training.data import DatasetSpecification, BaseDataset
from elasticai.explorer.training.trainer import SupervisedTrainer
from settings import ROOT_DIR


class SineDataset(BaseDataset):
    def __init__(
        self,
        root,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seq_length=50,
        total_samples=1000,
        *args,
        **kwargs,
    ):
        super().__init__("", transform, target_transform, *args, **kwargs)
        x = np.linspace(0, 100, total_samples + seq_length)
        noise_level = 0.1
        self.data = (
            np.sin(x)
            + 0.5 * np.sin(3 * x)
            + 0.2 * np.sin(7 * x)
            + np.random.normal(scale=noise_level, size=x.shape)
        )
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        seq = self.data[idx : idx + self.seq_length]
        target = self.data[idx + self.seq_length]
        return torch.tensor(seq, dtype=torch.float32).unsqueeze(-1), torch.tensor(
            target, dtype=torch.float32
        ).unsqueeze(-1)


class LSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        out, _ = self.lstm(x)
        # take last output only
        out = out[:, -1, :]  # [batch, hidden_size]
        out = self.fc(out)  # [batch, 1]
        return out.squeeze(-1)  # [batch]


class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, bidirectional=True, batch_first=True
        )
        self.linear = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.linear(last_output).squeeze(-1)


def validate(model, test_loader):
    model.eval()
    preds = []
    targets = []
    total_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for seqs, target in test_loader:
            output = model(seqs)

            for i, out in enumerate(output):
                preds.append(output[i].item())
                targets.append(target[i].item())

            loss = criterion(output, target)
            total_loss += loss.item()
    print(preds)
    print(targets)
    print(len(test_loader))
    print(total_loss / len(test_loader))
    # Plot
    plt.plot(targets, label="True")
    plt.plot(preds, label="Predicted")
    plt.legend()
    plt.title("Sine Wave Prediction")
    plt.show()


def run_training():
    seq_length = 50
    batch_size = 32
    # SineDataset(root="", seq_length=seq_length)
    data_spec = DatasetSpecification(
        dataset_type=SineDataset,
        dataset_location=Path(""),
        transform=None,
        target_transform=None,  # v2.Compose([v2.Lambda(lambda y: y.unsqueeze(1))]),
        train_test_val_ratio=[0.7, 0.1, 0.2],
        shuffle=False,
        split_seed=42,
    )
    trainer = SupervisedTrainer(
        "cpu",
        dataset_spec=data_spec,
        batch_size=batch_size,
        loss_fn=nn.MSELoss(),
        extra_metrics={},
    )
    model = SimpleLSTM(input_size=1, hidden_size=50)
    trainer.configure_optimizer(torch.optim.Adam(model.parameters(), lr=0.01))
    trainer.train(model, epochs=50, early_stopping=True)
    metrics, loss = trainer.test(model)
    print(metrics)
    print(loss)
    print("----------------")

    validate(model, trainer.test_loader)


def run_lstm_search():
    search_space = Path(
        ROOT_DIR / "elasticai/explorer/hw_nas/search_space/search_space.yaml"
    )

    batch_size = 32
    data_spec = DatasetSpecification(
        dataset_type=SineDataset,
        dataset_location=Path(""),
        transform=None,
        target_transform=None,  # v2.Compose([v2.Lambda(lambda y: y.unsqueeze(1))]),
        train_test_val_ratio=[0.7, 0.1, 0.2],
        shuffle=False,
        split_seed=42,
    )
    trainer = SupervisedTrainer(
        "cpu",
        dataset_spec=data_spec,
        batch_size=batch_size,
        loss_fn=nn.MSELoss(),
        extra_metrics={},
    )
    search_space_cfg = yaml_to_dict(search_space)
    top_models, _, _ = hw_nas.search(
        search_space_cfg,
        HWNASConfig(Path(ROOT_DIR / "configs/hwnas_config.yaml")),
        trainer=trainer,
    )
    model = top_models[0]
    print(model)
    trainer = SupervisedTrainer(
        "cpu",
        dataset_spec=data_spec,
        batch_size=batch_size,
        loss_fn=nn.MSELoss(),
        extra_metrics={},
    )
    trainer.configure_optimizer(torch.optim.Adam(model.parameters(), lr=0.01))
    trainer.train(model, epochs=50, early_stopping=True)
    validate(model, trainer.test_loader)


def trythedata():
    batch_size = 32
    data_spec = DatasetSpecification(
        dataset_type=SineDataset,
        dataset_location=Path(""),
        transform=None,
        target_transform=None,  # v2.Compose([v2.Lambda(lambda y: y.unsqueeze(1))]),
        train_test_val_ratio=[0.7, 0.1, 0.2],
        shuffle=False,
        split_seed=42,
    )
    trainer = SupervisedTrainer(
        "cpu",
        dataset_spec=data_spec,
        batch_size=batch_size,
        loss_fn=nn.MSELoss(),
        extra_metrics={},
    )

    model = nn.Sequential(
        architecture_components.SimpleLSTM(
            input_size=1,
            hidden_size=20,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        ),
        LinearOne(20, 1),
    )
    trainer.configure_optimizer(torch.optim.Adam(model.parameters(), lr=0.01))
    trainer.train(model, epochs=1, early_stopping=True)


# Objekte: SearchspaceComponents
# implementieren interface : kriegt predecessor-> hat infos über: output dimensionalität
# wie input formatiert sein muss, -> eventuelle schritte zum umformatieren
# evtl fehlermeldungen wenn ein predecessor keinen sinn macht (zb linear vor conv)
# für jede layer eins, anzahl kann von blöcken gemanaged werden

if __name__ == "__main__":

    # d = DataLoader(dataset, 32)
    run_lstm_search()
# trythedata()
# run_training()
# seq_length = 50
# batch_size = 32
# dataset = SineDataset(seq_length=seq_length)
# train_size = int(0.8 * len(dataset))
# train_dataset, test_dataset = torch.utils.data.random_split(
#     dataset, [train_size, len(dataset) - train_size]
# )
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=1)
# model = SimpleLSTM()
# train(model, train_loader)
# validate(model, test_loader)
# hwnas_cfg = HWNASConfig(config_path=Path("configs/hwnas_config.yaml"))
# deploy_cfg = DeploymentConfig(config_path=Path("configs/deployment_config.yaml"))
# knowledge_repo = setup_knowledge_repository_pi()
# explorer = Explorer(knowledge_repo)
#
# search_space = Path("examples/search_space_example.yaml")
#
# search_for_timeseries(explorer, hwnas_cfg, search_space)
