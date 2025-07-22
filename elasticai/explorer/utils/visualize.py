import logging
import os
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

from elasticai.explorer.utils import data_utils
from elasticai.explorer.utils.stats import compute_kendall

logger = logging.getLogger(__name__)


def plot_parallel_coordinates(df: pd.DataFrame):
    fig = px.parallel_coordinates(
        df,
        color="default",
        color_continuous_scale=px.colors.diverging.Tealrose,
    )
    fig.show()


class Metrics:
    def __init__(
        self,
        path_to_metrics: Path,
        path_to_samples: Path,
        accuracy_list: list,
        latency_list: list,
    ):
        self.raw_measured_accuracies: list[float] = accuracy_list
        self.raw_measured_latencies: list[int] = latency_list
        self.metric_list = data_utils.load_json(path_to_metrics)
        self.sample_list = data_utils.load_json(path_to_samples)
        self._structure()

    def _structure(self):

        number_of_models = len(self.sample_list)
        self.structured_est_metrics: List[List[List[float]]] = np.reshape(
            np.arange(0, 3 * 2 * number_of_models, 1, dtype=float),
            [3, 2, number_of_models],
        )  # type: ignore
        self.structured_samples: List[str] = []
        self.structured_est_flops: List[float] = []
        self.structured_est_accuracies: List[float] = []
        self.structured_est_combined: List[float] = []

        # first dimension accuracy, Latency, Combined
        # second dimension estimation, measured
        # third dimension sample number
        for n, metric in enumerate(self.metric_list):
            self.structured_est_metrics[0][0][n] = float(metric["Accuracy"])
            self.structured_est_metrics[1][0][n] = float(metric["flops log10"])
            self.structured_est_metrics[2][0][n] = float(metric["default"])

            self.structured_est_flops.append(metric["flops log10"])
            self.structured_est_accuracies.append(metric["Accuracy"])
            self.structured_est_combined.append(metric["default"])

        for sample in self.sample_list:
            self.structured_samples.append(str(sample))

        # Accuracy in %
        for n, accuracy in enumerate(self.raw_measured_accuracies):
            self.structured_est_metrics[0][1][n] = float(accuracy) * 100
            self.structured_est_metrics[2][1][n] = float(accuracy) * 100

        # Latency in milliseconds
        for n, latency in enumerate(self.raw_measured_latencies):
            self.structured_est_metrics[1][1][n] = float(latency) / 1000


class BarPlotVisualizer:

    def __init__(self, metrics: Metrics, plot_dir: Path):
        self.data: List[List[List[float]]] = metrics.structured_est_metrics
        self.labels: List[str] = metrics.structured_samples
        self.metrics: Metrics = metrics
        self.plot_dir: Path = plot_dir

    def plot_all_results(self, figure_size: list[int] = [15, 20], filename: str = ""):
        plt.figure()
        fig = plt.figure(num=1, clear=True)
        fig.set_size_inches(figure_size[0], figure_size[1], forward=True)
        axes = []
        axes.append(fig.add_subplot(311))
        axes.append(fig.add_subplot(312))
        axes.append(fig.add_subplot(313))
        bar_width = 0.2

        # get Kendall for estimated and measured Accuracies
        kendall_coef_accuracy = compute_kendall(
            self.metrics.structured_est_accuracies, self.metrics.raw_measured_accuracies
        )
        # get Kendall for estimated FLOPs and measured latencies
        kendall_coef_latencies = compute_kendall(
            self.metrics.structured_est_flops, self.metrics.raw_measured_latencies
        )
        # get Kendall for Combined Metric and measured Accuracies
        kendall_coef_combined = compute_kendall(
            self.metrics.structured_est_combined, self.metrics.raw_measured_latencies
        )

        indices = np.arange(0, len(self.data[0][0][:]), 1)

        # Accuracy Estimate vs Accuracy on Pi
        axes[0].bar(
            x=indices,
            height=self.data[0][0][:],
            width=bar_width,
            label="Estimated Accuracy in %",
        )
        axes[0].bar(
            x=indices + bar_width,
            height=self.data[0][1][:],
            width=bar_width,
            label="Measured Accuracy in %",
        )
        axes[0].set_xlabel("Sample")
        axes[0].set_ylabel("Accuracy in %")
        axes[0].set_title(
            f"Accuracy Estimation vs Measured Accuracy: Kendall's tau = {kendall_coef_accuracy}"
        )

        # FLOPS Proxy vs Latency on Pi
        axes[1].bar(
            x=indices,
            height=self.data[1][0][:],
            width=bar_width,
            label="FLOPs Estimation in Log10",
        )
        axes[1].bar(
            x=indices + bar_width,
            height=self.data[1][1][:],
            width=bar_width,
            label="Latency in Milliseconds",
        )
        axes[1].set_xlabel("Sample")
        axes[1].set_ylabel("Number FLOPs log10 and Latency in ms")
        axes[1].set_title(
            f"FLOPs Proxy-Estimation vs Measured Latency: Kendall's tau = {kendall_coef_latencies}"
        )

        # Combined Metric, accuracy-estimation and flops estimation
        axes[2].bar(
            x=indices,
            height=self.data[2][0][:],
            width=bar_width,
            label="Combined Metric",
        )
        axes[2].bar(
            x=indices + bar_width,
            height=self.data[2][1][:],
            width=bar_width,
            label="Accuracy",
        )
        axes[2].set_xlabel("Sample")
        axes[2].set_ylabel("Accuracy and FLOPs")
        axes[2].set_title(
            f"Combined Estimation vs Measured Accuracy: Kendall's tau = {kendall_coef_combined}"
        )

        for ax in axes:
            ax.legend(loc="upper left")
            ax.set_xticks(indices + bar_width / 2)
            ax.set_xticklabels(self.labels)

        os.makedirs(self.plot_dir, exist_ok=True)
        if filename:
            plt.savefig(self.plot_dir / (filename))
        else:
            plt.show()
