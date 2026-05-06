import logging
import multiprocessing as mp
import os
import pickle
import re
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import optuna
from optuna.samplers import BaseSampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from elasticai.explorer.parallel.config import (
    CreateSamplerFn,
    MultiprocessingConfig,
    OptimizationObjective,
    OptunaSearchConfig,
    SearchSpaceConfig,
    StudyDirection,
)

_logger = logging.getLogger("explorer.nas")
_DEVICE_RE = re.compile(r"^(cpu|mps|cuda:\d+)$")


def _run_optuna_multiprocessing_search(
    optuna_search_config: OptunaSearchConfig,
    multiprocessing_config: MultiprocessingConfig,
) -> optuna.Study:
    _validate_directions(
        direction=optuna_search_config.direction,
        directions=optuna_search_config.directions,
    )

    if multiprocessing_config.sampler_checkpoint_dir is not None:
        multiprocessing_config.sampler_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # multiprocessing works only with journal file or postgres backend
    study = optuna.create_study(
        direction=optuna_search_config.direction,
        directions=optuna_search_config.directions,
        study_name=optuna_search_config.study_name,
        storage=_create_journal_storage(multiprocessing_config.journal_file),
        sampler=None,
        load_if_exists=True,
    )

    _fail_stale_running_trials(
        study=study,
        study_name=optuna_search_config.study_name,
    )

    if multiprocessing_config.n_workers > 1:
        callbacks = _create_max_trials_callbacks(
            max_search_trials=optuna_search_config.max_search_trials,
            count_only_completed_trials=optuna_search_config.count_only_completed_trials,
        )

        _run_worker_pool(
            optuna_search_config=optuna_search_config,
            multiprocessing_config=multiprocessing_config,
            callbacks=callbacks,
        )
    else:
        raise ValueError("Set n_workers > 1 with devices for multiprocessing")
    return study


def _validate_directions(
    direction: StudyDirection | None,
    directions: Sequence[StudyDirection] | None,
) -> None:
    if direction is None and directions is None:
        raise ValueError("Either direction or directions must be set!")

    if direction is not None and directions is not None:
        raise ValueError("Greedy! Only set either direction or directions!")


def _create_max_trials_callbacks(
    max_search_trials: int | None,
    count_only_completed_trials: bool,
) -> list:
    if max_search_trials is not None:
        states = (TrialState.COMPLETE,) if count_only_completed_trials else None
        callbacks = [MaxTrialsCallback(max_search_trials, states=states)]
    else:
        callbacks = []

    return callbacks


def _fail_stale_running_trials(
    study: optuna.Study,
    study_name: str,
) -> None:
    # remove aborted trials (meant to clean up aborted trials at amplitUDE timeout)
    n_completed = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))

    stale = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))

    for trial in stale:
        study.tell(trial.number, state=TrialState.FAIL)

    n_stale_failed = len(stale)

    _logger.info(
        f"Loaded study '{study_name}': {n_completed} completed trials; "
        f"marked {n_stale_failed} stale RUNNING trials as FAILED."
    )


def _run_worker_pool(
    optuna_search_config: OptunaSearchConfig,
    multiprocessing_config: MultiprocessingConfig,
    callbacks: list[Callable],
) -> None:
    _ensure_pickable(
        obj=optuna_search_config.optimization_objective, label="optimization objective"
    )
    _ensure_pickable(
        obj=optuna_search_config.create_sampler_fn, label="sampler builder"
    )

    assigned_devices = _assign_workers_to_devices(
        n_workers=multiprocessing_config.n_workers,
        devices=multiprocessing_config.devices,
    )
    # necessary for using cuda with multiprocessing
    ctx = mp.get_context("spawn")

    with ctx.Pool(processes=multiprocessing_config.n_workers) as pool:
        pool.starmap(
            partial(
                _optimize_objective,
                optuna_search_config=optuna_search_config,
                multiprocessing_config=multiprocessing_config,
                callbacks=callbacks,
            ),
            enumerate(assigned_devices),
        )


def _optimize_objective(
    worker_idx: int,
    device: str,
    optuna_search_config: OptunaSearchConfig,
    multiprocessing_config: MultiprocessingConfig,
    callbacks: list[Callable],
):
    sampler = _load_or_build_sampler(
        create_sampler_fn=optuna_search_config.create_sampler_fn,
        worker_idx=worker_idx,
        checkpoint_dir=multiprocessing_config.sampler_checkpoint_dir,
    )

    study = optuna.create_study(
        direction=optuna_search_config.direction,
        directions=optuna_search_config.directions,
        study_name=optuna_search_config.study_name,
        storage=_create_journal_storage(multiprocessing_config.journal_file),
        sampler=sampler,
        load_if_exists=True,
    )

    if multiprocessing_config.sampler_checkpoint_dir is not None:
        callbacks.append(
            _save_sampler_callback(
                _sampler_checkpoint(
                    checkpoint_dir=multiprocessing_config.sampler_checkpoint_dir,
                    worker_idx=worker_idx,
                )
            )
        )

    study.optimize(
        partial(
            _parallel_objective,
            optimization_objective=optuna_search_config.optimization_objective,
            search_space_cfg=optuna_search_config.search_space_cfg,
            device=device,
        ),
        callbacks=callbacks,
        show_progress_bar=True,
        gc_after_trial=True,
    )


def _parallel_objective(
    trial: optuna.Trial,
    optimization_objective: OptimizationObjective,
    search_space_cfg: SearchSpaceConfig,
    device: str,
):
    trial.set_user_attr("worker_pid", os.getpid())
    return optimization_objective(trial, search_space_cfg, device)


def _sampler_checkpoint(
    checkpoint_dir: Path,
    worker_idx: int,
) -> Path:
    return checkpoint_dir / f"sampler_worker_{worker_idx}.pkl"


def _load_or_build_sampler(
    create_sampler_fn: CreateSamplerFn,
    worker_idx: int,
    checkpoint_dir: Path | None,
) -> BaseSampler:
    if checkpoint_dir is None:
        return create_sampler_fn(worker_idx)

    sampler_checkpoint = _sampler_checkpoint(
        checkpoint_dir=checkpoint_dir,
        worker_idx=worker_idx,
    )

    # loaded sampler state from pickle file
    if sampler_checkpoint.exists():
        with open(sampler_checkpoint, "rb") as f:
            _logger.info(
                f"Worker {worker_idx} resuming sampler from {sampler_checkpoint}"
            )
            return pickle.load(f)

    return create_sampler_fn(worker_idx)


def _save_sampler_callback(checkpoint_path: Path) -> Callable:
    def callback(study, trial):
        with open(checkpoint_path, "wb") as f:
            pickle.dump(study.sampler, f)

    return callback


def _ensure_pickable(
    obj: Any,
    label: str,
) -> None:
    try:
        pickle.dumps(obj)
    except Exception as e:
        raise TypeError(
            f"{label} is not picklable and will fail in worker process. "
            f"Common cause: lambda functions. Original message: {e}"
        ) from e


def _assign_workers_to_devices(
    n_workers: int,
    devices: list[str],
) -> list[str]:
    if n_workers <= 0:
        raise ValueError("n_workers must be > 0.")
    if not devices:
        raise ValueError("devices can't be empty.")

    # remove duplicates and normalize
    unique_devices = list(dict.fromkeys(d.strip().lower() for d in devices))

    invalid = [d for d in unique_devices if not _DEVICE_RE.fullmatch(d)]
    if invalid:
        raise ValueError(f"Invalid devices: {invalid}")

    n_devices = len(unique_devices)

    assigned = [unique_devices[i % n_devices] for i in range(n_workers)]

    return assigned


def _create_journal_storage(file: str) -> JournalStorage:
    return JournalStorage(JournalFileBackend(file_path=file))
