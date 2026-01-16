from typing import Optional, Callable, Union
from functools import partial

import torch
import pandas as pd
from matplotlib import pyplot as plt
from torch import nn
from torchvision.transforms import Compose

from elasticai.explorer.config import HWNASConfig
from elasticai.explorer.hw_nas import hw_nas

from elasticai.explorer.hw_nas.search_space.utils import yaml_to_dict
from elasticai.explorer.platforms.generator.generator import RPiGenerator

from elasticai.explorer.training.data import DatasetSpecification, MultivariateTimeseriesDataset
from elasticai.explorer.training.trainer import SupervisedTrainer

from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

settings_path = Path(__file__).resolve().parents[1] / "settings.py"
spec = spec_from_file_location("settings", settings_path)
settings = module_from_spec(spec)
spec.loader.exec_module(settings)

ROOT_DIR = settings.ROOT_DIR

class KuntzeDataset(MultivariateTimeseriesDataset):
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
        data_columns = ["pH", "Temp", "Cl2", "DIS-Control1", "Event_No_Water", "Event_Dosage_Check"]
        self.df = pd.read_csv(root, usecols=data_columns, skiprows=4)
        self.num_features = len(data_columns) - 1   # Exclude target column
        self.system_id = int(Path(root).stem.split("_")[2])
        self.lag_time_minutes = 12.67 if self.system_id == 570 \
                    else 9.87 if self.system_id == 785 \
                    else 6.31 if self.system_id == 1215 \
                    else 0
        self.lag_time_samples = round(self.lag_time_minutes * 6)  # Assuming data is sampled every 10 seconds

        super().__init__(root, transform, target_transform, *args, **kwargs)

class KuntzeRegressionDataset(KuntzeDataset):
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
        super().__init__(root, transform, target_transform, *args, **kwargs)

    def _setup_data(self):
        return self.df.copy(deep=False).drop(columns=["Event_No_Water", "Event_Dosage_Check"])[:-self.lag_time_samples].astype(float)

    def _setup_targets(self):
        # TODO: Use Event_No_Water(t) to mask prediction targets
        return self.df.copy(deep=False)["Cl2"][self.lag_time_samples:].astype(float)


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
    plt.title("Cl2 Prediction")
    plt.show()


def run_lstm_search():
    search_space = Path(ROOT_DIR / "examples/lstm_search_space_kuntze.yaml")

    batch_size = 32
    data_spec = DatasetSpecification(
        dataset_type=KuntzeRegressionDataset,
        dataset_location=Path("data/kuntze/raw_data/exported_data_570_2024-10-01 00-00-00_to_2024-10-31 00-00-00.csv"),
        transform=Compose([partial(torch.tensor, dtype=torch.float32)]),
        target_transform=Compose([partial(torch.tensor, dtype=torch.float32)]),
        train_val_test_ratio=[0.7, 0.1, 0.2],
        shuffle=False,
        split_seed=42,
    )
    trainer = SupervisedTrainer(
        "cpu",
        dataset_spec=data_spec,
        batch_size=batch_size,
        loss_fn=nn.L1Loss(),
        extra_metrics={},
    )
    search_space_cfg = yaml_to_dict(search_space)
    top_models, _, _ = hw_nas.search(
        search_space_cfg,
        HWNASConfig(),
        trainer=trainer,
    )
    model = top_models[0]
    print(model)
    trainer = SupervisedTrainer(
        "cpu",
        dataset_spec=data_spec,
        batch_size=batch_size,
        loss_fn=nn.L1Loss(),
        extra_metrics={},
    )
    trainer.configure_optimizer(torch.optim.Adam(model.parameters(), lr=0.01))
    trainer.train(model, epochs=50, early_stopping=True)
    validate(model, trainer.test_loader)
    generator = RPiGenerator()
    generator.generate(model, ROOT_DIR / "experiments/lstm_model")


if __name__ == "__main__":
    run_lstm_search()
