import logging
import random
import threading
from typing import Any

import optuna
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.samplers import BaseSampler
from optuna.trial import FrozenTrial, TrialState

logger = logging.getLogger("explorer.nas")


class _Node:
    __slots__ = ("options", "children", "done")

    def __init__(self, options: list):
        self.options = options
        self.children: dict[Any, "_Node"] = {}
        self.done: set = set()

    def open_options(self) -> list:
        return [o for o in self.options if o not in self.done]

    @property
    def exhausted(self) -> bool:
        return len(self.done) == len(self.options)


def _options(dist: BaseDistribution) -> list:
    if isinstance(dist, CategoricalDistribution):
        return list(range(len(dist.choices)))  # Indizes
    if isinstance(dist, IntDistribution):
        return list(range(dist.low, dist.high + 1, dist.step))
    if isinstance(dist, FloatDistribution):
        if dist.step is None:
            raise ValueError(
                "Float-Parameter ohne `step` können nicht gegridet werden "
                f"(Verteilung: {dist}). Gib im YAML eine Liste oder einen step an."
            )
        n = int(round((dist.high - dist.low) / dist.step))
        return [dist.low + i * dist.step for i in range(n + 1)]
    raise NotImplementedError(f"Nicht unterstützte Verteilung: {dist}")


def _decode(dist: BaseDistribution, option: Any) -> Any:
    if isinstance(dist, CategoricalDistribution):
        return dist.choices[option]
    return option


class ExhaustiveGridSampler(BaseSampler):
    def __init__(self, seed: int | None = None, shuffle: bool = True):
        self._rng = random.Random(seed) if shuffle else None
        self._root: _Node | None = None
        self._paths: dict[int, list[tuple[_Node, Any]]] = {}
        self._lock = threading.Lock()

    def infer_relative_search_space(self, study, trial):
        return {}

    def sample_relative(self, study, trial, search_space):
        return {}

    def sample_independent(
        self,
        study: optuna.Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        with self._lock:
            path = self._paths.setdefault(trial.number, [])

            if not path:
                if self._root is None:
                    self._root = _Node(_options(param_distribution))
                node = self._root
            else:
                parent, parent_option = path[-1]
                node = parent.children.get(parent_option)
                if node is None:
                    node = _Node(_options(param_distribution))
                    parent.children[parent_option] = node

            open_options = node.open_options()
            if not open_options:
                logger.info("Grid is already searched completely")
                raise optuna.TrialPruned()

            option = self._rng.choice(open_options) if self._rng else open_options[0]
            path.append((node, option))

        return _decode(param_distribution, option)

    def after_trial(
        self,
        study: optuna.Study,
        trial: FrozenTrial,
        state: TrialState,
        values,
    ) -> None:
        with self._lock:
            path = self._paths.pop(trial.number, [])
            for node, option in reversed(path):
                node.done.add(option)
                if not node.exhausted:
                    break

            if self._root is not None and self._root.exhausted:
                logger.info("Grid Search completed: all configurations used.")
                study.stop()
