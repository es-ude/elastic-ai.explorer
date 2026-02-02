from typing import Optional, Callable, Union
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch import jit, nn
from torch.utils.data import Subset, DataLoader

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
        window_size: int = 90,
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


def validate(model, test_loader):
    model.eval()
    preds = []
    targets = []
    total_loss = 0
    criterion = torch.nn.L1Loss()

    with torch.no_grad():
        for seqs, target in test_loader:
            output = model(seqs)

            for i, out in enumerate(output):
                preds.append(output[i].item())
                targets.append(target[i].item())

            loss = criterion(output, target)
            total_loss += loss.item()
    print("Total loss:", total_loss / len(test_loader))
    # Plot
    plt.plot(targets, label="True", linewidth=0.5, alpha=0.7)
    plt.plot(preds, label="Predicted", linewidth=0.5, alpha=0.7)
    plt.plot(
        list((t - p) for t, p in zip(targets, preds)),
        label="Difference",
        linewidth=0.5,
        alpha=0.5,
    )
    plt.legend()
    plt.title("Cl2 Prediction")
    plt.savefig(ROOT_DIR / "examples/kuntze/experiments/lstm_model.svg", format="svg")


def run_lstm_search():
    search_space = Path(
        ROOT_DIR / "examples/kuntze/config/lstm_search_space_kuntze.yaml"
    )

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
        "mps",
        dataset_spec=data_spec,
        batch_size=batch_size,
        loss_fn=torch.nn.L1Loss(),
        extra_metrics={},
    )
    search_space_cfg = yaml_to_dict(search_space)
    top_models, _, _ = hw_nas.search(
        search_space_cfg,
        HWNASConfig(ROOT_DIR / "examples/kuntze/config/hwnas_config.yaml"),
        trainer=trainer,
    )

    for n, model in enumerate(top_models):
        with open(
            ROOT_DIR / f"examples/kuntze/experiments/top_model_{n}.txt", "w"
        ) as f:
            print(top_models[n], file=f)

    model = top_models[0]

    trainer = SupervisedTrainer(
        "mps",
        dataset_spec=data_spec,
        batch_size=batch_size,
        loss_fn=torch.nn.L1Loss(),
        extra_metrics={},
    )
    trainer.configure_optimizer(torch.optim.Adam(model.parameters(), lr=0.01))
    trainer.train(model, epochs=50, early_stopping=True)

    model.to("cpu")
    validate(model, trainer.test_loader)

    torch.save(
        model.state_dict(),
        ROOT_DIR / "examples/kuntze/experiments/lstm_model_0.pt",
    )

    generator = RPiGenerator()
    generator.generate(model, ROOT_DIR / "examples/kuntze/experiments/lstm_model_0_rpi")


def time_slice_mask(timestamps, start=None, end=None):
    """
    Build a boolean mask for timestamps between start and end (inclusive).

    timestamps: pd.Series of dtype datetime64[ns]
    start, end: str, pd.Timestamp, or None
    """

    ts = pd.to_datetime(timestamps)  # ensure datetime

    mask = pd.Series(True, index=ts.index)

    if start is not None:
        start = pd.to_datetime(start)
        mask &= ts >= start

    if end is not None:
        end = pd.to_datetime(end)
        mask &= ts <= end

    return mask.values  # return as NumPy array for indexing


def event_intervals(timestamps, event_flags):
    """
    Convert a binary event array into contiguous (start, end) timestamp intervals.
    """
    event_flags = np.asarray(event_flags).astype(bool)
    idx = np.where(event_flags)[0]

    if len(idx) == 0:
        return []

    intervals = []
    start = idx[0]

    for i in range(1, len(idx)):
        if idx[i] != idx[i - 1] + 1:
            intervals.append((timestamps.iloc[start], timestamps.iloc[idx[i - 1]]))
            start = idx[i]

    intervals.append((timestamps.iloc[start], timestamps.iloc[idx[-1]]))
    return intervals


def validate_events(
    model,
    test_loader,
    dosage_loader,
    nowater_loader,
    feature_index=3,  # index of feature to visualize
    window_size=90,
    start_time=None,
    end_time=None,
):
    model.eval()

    preds, targets = [], []
    feature_values = []
    dosage_events, nowater_events = [], []
    cl2 = []
    criterion = torch.nn.L1Loss()
    total_loss = 0.0

    timestamps = get_test_timestamps(window_size=window_size)

    with torch.no_grad():
        for (seqs, target), d_evt, nw_evt in zip(
            test_loader, dosage_loader, nowater_loader
        ):
            output = model(seqs)

            preds.extend(output.cpu().numpy().ravel())
            targets.extend(target.cpu().numpy().ravel())

            # use last timestep in window
            feature_batch = seqs[:, -1, feature_index].cpu().numpy()
            feature_values.extend(feature_batch)
            cl2_batch = seqs[:, -1, 2].cpu().numpy()
            cl2.extend(cl2_batch)
            dosage_events.extend(d_evt.cpu().numpy().ravel())
            nowater_events.extend(nw_evt.cpu().numpy().ravel())

            total_loss += criterion(output, target).item()

    print("Total loss:", total_loss / len(test_loader))
    # dosage_intervals = event_intervals(timestamps, dosage_events)
    # nowater_intervals = event_intervals(timestamps, nowater_events)

    # --- convert to numpy ---
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    cl2 = np.asarray(cl2)
    feature_values = np.asarray(feature_values)
    diff = targets - preds

    # dosage_idx = np.where(np.asarray(dosage_events) == 1)[0]
    # nowater_idx = np.where(np.asarray(nowater_events) == 1)[0]
    mask = time_slice_mask(timestamps, start_time, end_time)

    timestamps = timestamps[mask].reset_index(drop=True)
    preds = preds[mask]
    targets = targets[mask]
    cl2 = cl2[mask]
    feature_values = feature_values[mask]
    diff = diff[mask]

    dosage_events = np.asarray(dosage_events)[mask]
    nowater_events = np.asarray(nowater_events)[mask]
    dosage_intervals = event_intervals(timestamps, dosage_events)
    nowater_intervals = event_intervals(timestamps, nowater_events)
    print(timestamps.min(), timestamps.max(), len(timestamps))

    # --- figure with stacked axes ---
    fig, (ax_pred, ax_diff, ax_feat) = plt.subplots(
        3,
        1,
        figsize=(14, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1, 1]},
    )

    # ======================
    # Prediction subplot
    # ======================
    ax_pred.plot(
        timestamps,
        targets,
        label="True CL2 at x + Lagtime",
        linewidth=1.5,
        alpha=1,
    )
    ax_pred.plot(
        timestamps,
        cl2,
        label="CL2 at Timestep x",
        linewidth=1.5,
        alpha=1,
    )
    ax_pred.plot(
        timestamps,
        preds,
        label="Predicted CL2 at x + Lagtime",
        linewidth=1.5,
        alpha=1,
    )
    ax_diff.plot(
        timestamps,
        diff,
        label="Difference",
        linewidth=1.5,
        alpha=1,
    )
    # ax_pred.set_ylim(1, 3)

    ax_diff.set_ylabel("Difference")
    ax_diff.grid(alpha=0.2)
    ax_diff.set_axisbelow(True)
    ax_pred.set_ylabel("CL2 Prediction")
    ax_pred.legend()
    ax_pred.grid(alpha=0.2)
    ax_pred.set_axisbelow(True)

    ymin, ymax = ax_pred.get_ylim()
    band_height = 0.03 * (ymax - ymin)

    # Event markers (subtle, non-dominating)
    for start, end in dosage_intervals:
        ax_pred.axvspan(
            start,
            end,
            color="red",
            alpha=0.12,
            label="Event_Dosage_Check",
        )
        ax_diff.axvspan(
            start,
            end,
            color="red",
            alpha=0.12,
            label="Event_Dosage_Check",
        )
        ax_feat.axvspan(
            start,
            end,
            color="red",
            alpha=0.12,
            label="Event_Dosage_Check",
        )

    for start, end in nowater_intervals:
        ax_pred.axvspan(
            start,
            end,
            color="blue",
            alpha=0.12,
            label="Event_No_Water",
        )
        ax_diff.axvspan(
            start,
            end,
            color="blue",
            alpha=0.12,
            label="Event_No_Water",
        )
        ax_feat.axvspan(
            start,
            end,
            color="blue",
            alpha=0.12,
            label="Event_No_Water",
        )

    handles, labels = ax_pred.get_legend_handles_labels()
    ax_pred.legend(
        dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys()
    )

    # ======================
    # Feature subplot
    # ======================
    ax_feat.plot(
        timestamps,
        feature_values,
        color="red",
        linewidth=1.5,
        alpha=1,
        label="Dis-Control",
    )

    ax_feat.set_ylabel("Dis-Control")
    ax_feat.set_xlabel("Time")
    ax_feat.grid(alpha=0.2)
    ax_feat.set_axisbelow(True)

    # --- datetime formatting ---
    import matplotlib.dates as mdates

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax_feat.xaxis.set_major_locator(locator)
    ax_feat.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()

    # --- title & save ---
    fig.suptitle("CL2 Prediction with Events and Dis-Control Feature", y=0.98)

    plt.tight_layout()
    plt.savefig(
        ROOT_DIR / "examples/kuntze/experiments/lstm_model_dis_control_stacked.svg",
        format="svg",
    )
    plt.show(block=True)


def get_event_windows(window_size=90, batch_size=32):
    lag_time_minutes = 12.67
    lag_time_samples = round(lag_time_minutes * 6)

    dataset_location = Path(
        ROOT_DIR
        / "data/kuntze/raw_data/exported_data_570_2024-10-01 00-00-00_to_2024-10-31 00-00-00.csv"
    )

    df = pd.read_csv(
        dataset_location,
        usecols=("Event_Dosage_Check", "Event_No_Water", "device_ts"),
        skiprows=4,
    )
    # --- raw binary series ---
    dosage_raw = df["Event_Dosage_Check"].astype(int)
    nowater_raw = df["Event_No_Water"].astype(int)

    # --- apply lag ---
    dosage_lagged = dosage_raw.iloc[lag_time_samples:].reset_index(drop=True)
    nowater_lagged = nowater_raw.iloc[lag_time_samples:].reset_index(drop=True)

    # --- window → one value per prediction ---
    dosage_windowed = np.array(
        [
            int(dosage_lagged.iloc[i : i + window_size].any())
            for i in range(len(dosage_lagged) - window_size)
        ],
        dtype=np.int64,
    )

    nowater_windowed = np.array(
        [
            int(nowater_lagged.iloc[i : i + window_size].any())
            for i in range(len(nowater_lagged) - window_size)
        ],
        dtype=np.int64,
    )

    # --- same split logic as dataset ---
    ratio = [0.7, 0.1, 0.2]
    N = len(dosage_windowed)
    val_end = int((ratio[0] + ratio[1]) * N)

    dosage_test = dosage_windowed[val_end:]
    nowater_test = nowater_windowed[val_end:]

    dosage_loader = DataLoader(dosage_test, batch_size=batch_size, shuffle=False)
    nowater_loader = DataLoader(nowater_test, batch_size=batch_size, shuffle=False)

    return dosage_loader, nowater_loader


# def get_test_timestamps(window_size=90):
#     lag_time_minutes = 12.67
#     lag_time_samples = round(lag_time_minutes * 6)
#
#     dataset_location = Path(
#         ROOT_DIR
#         / "data/kuntze/raw_data/exported_data_570_2024-10-01 00-00-00_to_2024-10-31 00-00-00.csv"
#     )
#
#     df = pd.read_csv(
#         dataset_location,
#         usecols=("device_ts",),
#         skiprows=4,
#     )
#     print(df.head())
#     print(df.dtypes)
#     ts = pd.to_datetime(df["device_ts"])
#
#     # timestamps aligned with window targets
#     ts_aligned = ts.iloc[
#         lag_time_samples + window_size : lag_time_samples + window_size + len(ts)
#     ].reset_index(drop=True)
#
#     ratio = [0.7, 0.1, 0.2]
#     N = len(ts_aligned)
#     val_end = int((ratio[0] + ratio[1]) * N)
#
#     return ts_aligned.iloc[val_end:]


def get_test_timestamps(
    window_size=90,
    dataset_location=Path(
        ROOT_DIR
        / "data/kuntze/raw_data/exported_data_570_2024-10-01 00-00-00_to_2024-10-31 00-00-00.csv"
    ),
):
    lag_time_minutes = 12.67
    lag_time_samples = round(lag_time_minutes * 6)

    df = pd.read_csv(dataset_location, usecols=("device_ts",), skiprows=4)  # important

    ts = pd.to_datetime(df["device_ts"], errors="raise")

    ts_aligned = ts.iloc[lag_time_samples + window_size :].reset_index(drop=True)

    # same split logic as dataset
    ratio = [0.7, 0.1, 0.2]
    N = len(ts_aligned)
    val_end = int((ratio[0] + ratio[1]) * N)

    return ts_aligned.iloc[val_end:]


def validate_model(model, window_size=90):

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
        "mps",
        dataset_spec=data_spec,
        batch_size=batch_size,
        loss_fn=nn.MSELoss(),
        extra_metrics={},
    )
    dosage_loader, nowater_loader = get_event_windows(window_size=window_size)
    count = 0
    total = 0

    for batch in nowater_loader:
        batch = batch.cpu().numpy().ravel()
        count += (batch == 1).sum()
        total += len(batch)

    print(f"Dosage_Check == 1: {count} / {total} " f"({100 * count / total:.2f}%)")
    validate_events(
        model,
        trainer.test_loader,
        dosage_loader,
        nowater_loader,
        window_size=window_size,
        start_time="2024-10-26 12:00:00",
        end_time="2024-10-26 13:59:59",
    )


if __name__ == "__main__":
    #   run_lstm_search()
    window_size = 90
    model = jit.load(
        ROOT_DIR
        / "examples/kuntze/experiments/schonmal_funktionierend/lstm_model_0_rpi.pt"
    )
    validate_model(window_size=window_size, model=model)
