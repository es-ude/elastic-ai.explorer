import json
from dataclasses import dataclass
from typing import Type

import numpy as np

from elasticai.explorer.platforms.deployment.manager import HWManager
from elasticai.explorer.platforms.generator.generator import Generator


@dataclass
class HWPlatform:
    name: str
    info: str
    model_generator: Type[Generator]
    platform_manager: Type[HWManager]


class KnowledgeRepository:
    def __init__(self):
        self.supported_hw_platforms = {}

    def register_hw_platform(self, platform: HWPlatform):
        self.supported_hw_platforms[platform.name] = platform

    def fetch_hw_info(self, name: str) -> HWPlatform:
        return self.supported_hw_platforms[name]


class Metrics:
    def __init__(self, path_to_metrics: str, path_to_samples: str, accuracy_list: list, latency_list: list):
        self.raw_metrics = path_to_metrics
        self.raw_measured_accuracies = accuracy_list
        self.raw_measured_latencies = latency_list

        with open(path_to_metrics, "r") as f:
            self.metric_list = json.load(f)

        with open(path_to_samples, "r") as f:
            self.sample_list = json.load(f)

        self._structure()

    def _structure(self):

        number_of_models = len(self.sample_list)
        self.structured_est_metrics = np.reshape(np.arange(0, 3 * 2 * number_of_models, 1, dtype=float),
                                                 [3, 2, number_of_models])
        self.structured_samples = []
        self.structured_est_flops = []
        self.structured_est_accuracies = []
        self.structured_est_combined = []

        # first dimension accuracy, Latency, Combined
        # second dimension estimation, measured
        # third dimension sample number
        for n, metric in enumerate(self.metric_list):
            self.structured_est_metrics[0][0][n] = metric["accuracy"]
            self.structured_est_metrics[1][0][n] = metric["flops log10"]
            self.structured_est_metrics[2][0][n] = metric["default"]

            self.structured_est_flops.append(metric["flops log10"])
            self.structured_est_accuracies.append(metric["accuracy"])
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
