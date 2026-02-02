from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from elasticai.explorer.hw_nas import hw_nas

from elasticai.explorer.hw_nas.estimators import TrainMetricsEstimator
from elasticai.explorer.hw_nas.optimization_criteria import OptimizationCriteria
from elasticai.explorer.hw_nas.sampler import RandomSamplerWrapper
from elasticai.explorer.hw_nas.search_space.utils import yaml_to_dict
from elasticai.explorer.platforms.generator.generator import RPiGenerator

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


def run_lstm_search():
    search_space = Path(
        ROOT_DIR / "examples/search_space_examples/lstm_search_space.yaml"
    )
    device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    max_search_trials, top_n_models = 2, 2
    batch_size = 32
    data_spec = DatasetSpecification(
        dataset_type=SineDataset,
        dataset_location=Path(""),
        transform=None,
        target_transform=None,
        train_val_test_ratio=[0.7, 0.1, 0.2],
        shuffle=False,
        split_seed=42,
    )
    trainer = SupervisedTrainer(
        device,
        dataset_spec=data_spec,
        batch_size=batch_size,
        loss_fn=nn.MSELoss(),
        extra_metrics={},
    )

    criteria_reg = OptimizationCriteria()

    accuracy_estimator = TrainMetricsEstimator(trainer, n_estimation_epochs=3)
    criteria_reg.register_objective(estimator=accuracy_estimator)

    search_space_cfg = yaml_to_dict(search_space)
    top_models, _, _ = hw_nas.search(
        search_space_cfg,
        RandomSamplerWrapper(),
        criteria_reg,
        hw_nas_parameters=hw_nas.HWNASParameters(max_search_trials, top_n_models),
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
    generator = RPiGenerator()
    generator.generate(model, ROOT_DIR / "experiments/lstm_model")


if __name__ == "__main__":
    run_lstm_search()
