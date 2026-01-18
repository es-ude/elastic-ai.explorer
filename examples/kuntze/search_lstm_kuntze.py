from typing import Optional, Callable, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch

from elasticai.explorer.config import HWNASConfig
from elasticai.explorer.hw_nas import hw_nas

from elasticai.explorer.hw_nas.search_space.utils import yaml_to_dict
from elasticai.explorer.platforms.generator.generator import RPiGenerator

from elasticai.explorer.training.data import (
    DatasetSpecification,
    MultivariateTimeseriesDataset,
)
from elasticai.explorer.training.trainer import SupervisedTrainer

from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

settings_path = Path(__file__).resolve().parents[2] / "settings.py"
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
        window_size: int = 90,  # FIXME: Hardcoded. Could be hyperparameter.
        *args,
        **kwargs,
    ):
        data_columns = [
            "pH",
            "Temp",
            "Cl2",
            "DIS-Control1",
            "Event_No_Water",
            "Event_Dosage_Check",
        ]
        self.df = pd.read_csv(root, usecols=data_columns, skiprows=4)
        self.num_features = len(data_columns) - 1  # Exclude target column
        self.system_id = int(Path(root).stem.split("_")[2])
        self.lag_time_minutes = (
            12.67
            if self.system_id == 570
            else (
                9.87 if self.system_id == 785 else 6.31 if self.system_id == 1215 else 0
            )
        ) / 2  # FIXME: Hardcoded, halve lag time for prediction, ensure that window_size > lag_time_samples. Could be hyperparameter.
        self.lag_time_samples = 1 + round(
            self.lag_time_minutes * 6
        )  # Assuming data is sampled every 10 seconds, FIXME: hardcoded

        super().__init__(
            root, transform, target_transform, window_size=window_size, *args, **kwargs
        )


class KuntzeRegressionDataset(KuntzeDataset):
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        window_size: int = 90,
        *args,
        **kwargs,
    ):
        super().__init__(
            root, transform, target_transform, window_size=window_size, *args, **kwargs
        )

    def _setup_data(self):
        return (
            self.df.copy(deep=False)
            .drop(columns=["Event_No_Water", "Event_Dosage_Check"])[
                : -self.lag_time_samples
            ]
            .astype(float)
        )

    def _setup_targets(self):
        # TODO: Use Event_No_Water(t) to mask prediction targets
        return self.df.copy(deep=False)["Cl2"][self.lag_time_samples :].astype(float)


def validate(model, test_loader, window_size: int, lag_time_samples: int):
    model.eval()
    preds = []
    targets = []
    time_indices = []
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    global_sample_idx = 0  # zählt Dataset-Samples (nicht CSV-Zeitpunkte!)

    with torch.no_grad():
        for seqs, target in test_loader:
            output = model(seqs)

            batch_size = target.shape[0]

            # Verlust korrekt über Batch
            loss = criterion(output, target)
            total_loss += loss.item()

            for i in range(batch_size):
                preds.append(output[i].item())
                targets.append(target[i].item())

                # Rekonstruktion des Original-Zeitindex
                # y(idx) gehört zu:
                # CSV-Zeit = idx + window_size + lag_time_samples
                time_idx = global_sample_idx + window_size + lag_time_samples
                time_indices.append(time_idx)

                global_sample_idx += 1

    avg_loss = total_loss / len(test_loader)
    print("Total loss:", avg_loss)

    # Sortieren (robust gegen zukünftige Änderungen wie shuffle=True)
    time_indices = np.array(time_indices)
    preds = np.array(preds)
    targets = np.array(targets)

    order = np.argsort(time_indices)
    time_indices = time_indices[order]
    preds = preds[order]
    targets = targets[order]

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(time_indices, targets, label="True", linewidth=0.7, alpha=0.7)
    plt.plot(time_indices, preds, label="Predicted", linewidth=0.7, alpha=0.7)
    plt.plot(
        time_indices,
        targets - preds,
        label="Difference",
        linewidth=0.6,
        alpha=0.5,
    )
    plt.legend()
    plt.xlabel("Time index (CSV samples)")
    plt.ylabel("Cl2")
    plt.title("Cl2 Prediction")
    plt.tight_layout()
    plt.savefig(
        ROOT_DIR / "examples/kuntze/results/lstm_model.svg",
        format="svg",
    )
    plt.close()



def run_lstm_search():
    search_space = Path(ROOT_DIR / "examples/kuntze/config/lstm_search_space_kuntze.yaml")

    batch_size = 32
    data_spec = DatasetSpecification(
        dataset_type=KuntzeRegressionDataset,
        dataset_location=Path(
            ROOT_DIR
            / "data/kuntze/raw_data/exported_data_570_2024-10-01 00-00-00_to_2024-10-31 00-00-00.csv"
        ),
        train_val_test_ratio=[0.7, 0.1, 0.2],
        shuffle=False,
        split_seed=42,
    )
    trainer = SupervisedTrainer(
        "cuda",
        dataset_spec=data_spec,
        batch_size=batch_size,
        loss_fn=torch.nn.MSELoss(),
        extra_metrics={},
    )
    search_space_cfg = yaml_to_dict(search_space)
    top_models, _, _ = hw_nas.search(
        search_space_cfg,
        HWNASConfig(ROOT_DIR / "examples/kuntze/config/hwnas_config.yaml"),
        trainer=trainer,
    )

    for n, model in enumerate(top_models):
        with open(ROOT_DIR / f"examples/kuntze/results/top_model_{n}.txt", "w") as f:
            print(top_models[n], file=f)
    
    model = top_models[0]
    
    trainer = SupervisedTrainer(
        "cuda",
        dataset_spec=data_spec,
        batch_size=batch_size,
        loss_fn=torch.nn.MSELoss(),
        extra_metrics={},
    )
    trainer.configure_optimizer(torch.optim.Adam(model.parameters(), lr=0.01))
    trainer.train(model, epochs=50, early_stopping=True)

    model.to("cpu")
    validate(model, trainer.test_loader, window_size=trainer.dataset.window_size, lag_time_samples=trainer.dataset.lag_time_samples)

    torch.save(
        model.state_dict(),
        ROOT_DIR / "examples/kuntze/results/lstm_model_0.pt",
    )

    generator = RPiGenerator()
    generator.generate(model, ROOT_DIR / "examples/kuntze/results/lstm_model_0_rpi")


if __name__ == "__main__":
    run_lstm_search()
