from pathlib import Path
from typing import Optional, Callable, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import (
    Module,
    Sequential,
    LSTM,
    Linear,
    LayerNorm,
    Linear,
    Tanh,
)

from elasticai.explorer.training.data import (
    DatasetSpecification,
    MultivariateTimeseriesDataset,
)
from elasticai.explorer.training.trainer import ReconstructionAutoencoderTrainer


class GaussianDropout(Module):
    def __init__(self, p=0.5):
        super(GaussianDropout, self).__init__()
        if p < 0 or p >= 1:
            raise Exception("p value should accomplish 0 <= p < 1")
        self.p = p

    def forward(self, x):
        if self.training:
            stddev = (self.p / (1.0 - self.p)) ** 0.5
            epsilon = torch.randn_like(x) * stddev
            return x * epsilon
        else:
            return x


class LambdaLayer(Module):
    def __init__(self, func):
        super(LambdaLayer, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class TimeDistributed(Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(
            -1, x.size(-1)
        )  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        y = y.contiguous().view(
            x.size(0), -1, y.size(-1)
        )  # (samples, timesteps, output_size)

        if not self.batch_first:
            y = y.transpose(
                0, 1
            ).contiguous()  # transpose to (timesteps, samples, output_size)

        return y


class KuntzeDataset(MultivariateTimeseriesDataset):
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        window_size: Optional[int] = 1,
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
        )
        self.lag_time_samples = round(
            self.lag_time_minutes * 6
        )  # Assuming data is sampled every 10 seconds

        super().__init__(
            root, transform, target_transform, window_size=window_size, *args, **kwargs
        )


class KuntzeRegressionDataset(KuntzeDataset):
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        window_size: Optional[int] = 1,
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


class KuntzeClassificationDataset(KuntzeDataset):
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
        return (
            self.df.copy(deep=False)
            .drop(columns=["Event_No_Water", "Event_Dosage_Check"])[
                : -self.lag_time_samples
            ]
            .astype(float)
        )

    def _setup_targets(self):
        # TODO: Use Event_No_Water(t) to mask prediction targets
        return self.df.copy(deep=False)["Event_Dosage_Check"][
            self.lag_time_samples :
        ].astype(float)


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
    plt.title("Kuntze Prediction")
    plt.show()


def get_autoencoder_model(window_size: int = 90, num_features: int = 4):
    # * Define hyperparameters
    n_lstm_layers = 3
    gaussian_dropout_rate = 0.1

    model = Sequential()
    model.append(
        LayerNorm(normalized_shape=[window_size, num_features])
    )  # input normalization

    # ** encoder
    model.append(
        LSTM(
            input_size=num_features,
            hidden_size=15,
            num_layers=n_lstm_layers - 1,
            bidirectional=True,
        )
    )
    model.append(LSTM(input_size=15, hidden_size=15, num_layers=1, bidirectional=True))
    model.append(
        LambdaLayer(lambda x: torch.Tensor.reshape(x, x[:, -1, :]))
    )  # "return_sequences=False" in Keras

    # ** latent space
    model.append(GaussianDropout(gaussian_dropout_rate))
    model.append(LambdaLayer(lambda x: torch.Tensor.repeat(x, window_size)))

    # ** decoder
    model.append(
        LSTM(
            input_size=window_size,
            hidden_size=15,
            num_layers=n_lstm_layers,
            bidirectional=True,
        )
    )
    model.append(TimeDistributed(Linear(in_features=15, out_features=num_features)))
    model.append(Tanh())
    model.append(LambdaLayer(lambda x: 3.0 * x))

    return model


def train_autoencoder():
    batch_size = 32
    data_spec = DatasetSpecification(
        dataset_type=KuntzeRegressionDataset,
        dataset_location=Path(
            "data/kuntze/raw_data/exported_data_570_2024-10-01 00-00-00_to_2024-10-31 00-00-00.csv"
        ),
        transform=None,
        target_transform=None,
        train_val_test_ratio=[0.7, 0.1, 0.2],
        shuffle=False,
        split_seed=42,
        window_size=90,
    )

    model = get_autoencoder_model()
    print(model)

    trainer = ReconstructionAutoencoderTrainer(
        "cuda" if torch.cuda.is_available() else "cpu",
        dataset_spec=data_spec,
        batch_size=batch_size,
        loss_fn=nn.L1Loss(),
    )
    trainer.configure_optimizer(torch.optim.Adam(model.parameters(), lr=0.01))
    trainer.train(model, epochs=5, early_stopping=True)
    validate(model, trainer.test_loader)


if __name__ == "__main__":
    train_autoencoder()
