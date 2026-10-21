from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import optuna
from optuna.samplers import BaseSampler

StudyDirection: TypeAlias = Literal["minimize", "maximize"]
SearchSpaceConfig: TypeAlias = dict[str, Any]
CreateSamplerFn: TypeAlias = Callable[[int], BaseSampler]
OptimizationObjective: TypeAlias = Callable[
    [optuna.Trial, SearchSpaceConfig, str], int | float | Sequence[int | float]
]


@dataclass(frozen=True, kw_only=True)
class OptunaSearchConfig:
    search_space_cfg: SearchSpaceConfig
    create_sampler_fn: CreateSamplerFn
    optimization_objective: OptimizationObjective
    study_name: str
    directions: tuple[StudyDirection, ...]
    max_search_trials: int | None = None
    count_only_completed_trials: bool = False


@dataclass(frozen=True, kw_only=True)
class MultiprocessingConfig:
    journal_file: str
    devices: list[str]
    n_workers: int = 2
    sampler_checkpoint_dir: Path | None = None
