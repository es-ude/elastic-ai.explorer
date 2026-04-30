import logging
import multiprocessing as mp
import os
import pickle
import re
from functools import partial
from pathlib import Path
from typing import Any, Callable

import optuna
from optuna.samplers import BaseSampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

logger = logging.getLogger("explorer.nas")
_DEVICE_RE = re.compile(r"^(cpu|mps|cuda:\d+)$")


def _sampler_checkpoint(
    checkpoint_dir: Path,
    worker_idx: int,
) -> Path:
    return checkpoint_dir / f"sampler_worker_{worker_idx}.pkl"


def _load_or_build_sampler(
    sampler_builder: Callable[[int], BaseSampler],
    worker_idx: int,
    checkpoint_dir: Path | None,
) -> BaseSampler:
    if checkpoint_dir is None:
        return sampler_builder(worker_idx)

    sampler_checkpoint = _sampler_checkpoint(
        checkpoint_dir=checkpoint_dir,
        worker_idx=worker_idx,
    )

    # loaded sampler state from pickle file
    if sampler_checkpoint.exists():
        with open(sampler_checkpoint, "rb") as f:
            logger.info(
                f"Worker {worker_idx} resuming sampler from {sampler_checkpoint}"
            )
            return pickle.load(f)

    return sampler_builder(worker_idx)


def _save_sampler_callback(checkpoint_path: Path) -> Callable:
    def callback(study, trial):
        with open(checkpoint_path, "wb") as f:
            pickle.dump(study.sampler, f)

    return callback


def _fail_stale_running_trials(study: optuna.Study) -> int:
    stale = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))

    for trial in stale:
        study.tell(trial.number, state=TrialState.FAIL)

    return len(stale)


def _validate_pickable(
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


def assign_workers_to_devices(
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


def is_duplicated(trial: optuna.Trial) -> bool:
    states_to_consider = (
        TrialState.RUNNING,
        TrialState.COMPLETE,
    )
    trials_to_consider = trial.study.get_trials(
        deepcopy=False, states=states_to_consider
    )

    for t in trials_to_consider:
        if t.number == trial.number:
            continue
        if t.params == trial.params:
            return True

    return False


def parallel_objective(
    trial: optuna.Trial,
    optimization_objective: Callable[[optuna.Trial, dict[str, Any], str], Any],
    search_space_cfg: dict,
    device: str,
):
    trial.set_user_attr("worker_pid", os.getpid())
    return optimization_objective(trial, search_space_cfg, device)


def _create_journal_storage(file: str) -> JournalStorage:
    return JournalStorage(JournalFileBackend(file_path=file))


def _optimize_objective(
    worker_idx: int,
    device: str,
    search_space_cfg: dict,
    optimization_objective: Callable[[optuna.Trial, dict[str, Any], str], Any],
    study_name: str,
    journal_file: str,
    sampler_builder: Callable[[int], BaseSampler],
    callbacks: list,
    sampler_checkpoint_dir: Path | None,
    direction: str | None = None,
    directions: list[str] | None = None,
):
    sampler = _load_or_build_sampler(
        sampler_builder=sampler_builder,
        worker_idx=worker_idx,
        checkpoint_dir=sampler_checkpoint_dir,
    )

    study = optuna.create_study(
        direction=direction,
        directions=directions,
        study_name=study_name,
        storage=_create_journal_storage(journal_file),
        sampler=sampler,
        load_if_exists=True,
    )

    effective_callbacks = list(callbacks)
    if sampler_checkpoint_dir is not None:
        effective_callbacks.append(
            _save_sampler_callback(
                _sampler_checkpoint(
                    checkpoint_dir=sampler_checkpoint_dir,
                    worker_idx=worker_idx,
                )
            )
        )

    study.optimize(
        partial(
            parallel_objective,
            optimization_objective=optimization_objective,
            search_space_cfg=search_space_cfg,
            device=device,
        ),
        callbacks=effective_callbacks,
        show_progress_bar=True,
        gc_after_trial=True,
    )


def _run_multiprocessing_search(
    study_name: str,
    sampler_builder: Callable[[int], BaseSampler],
    optimization_objective: Callable[[optuna.Trial, dict[str, Any], str], Any],
    journal_file: str,
    devices: list[str],
    search_space_cfg: dict,
    n_workers: int,
    callbacks: list[Callable],
    sampler_checkpoint_dir: Path | None,
    direction: str | None = None,
    directions: list[str] | None = None,
) -> None:
    _validate_pickable(obj=optimization_objective, label="optimization objective")
    _validate_pickable(obj=sampler_builder, label="sampler builder")

    assigned_devices = assign_workers_to_devices(
        n_workers=n_workers,
        devices=devices,
    )
    # necessary for using cuda with multiprocessing
    ctx = mp.get_context("spawn")

    with ctx.Pool(processes=n_workers) as pool:
        pool.starmap(
            partial(
                _optimize_objective,
                search_space_cfg=search_space_cfg,
                optimization_objective=optimization_objective,
                study_name=study_name,
                journal_file=journal_file,
                sampler_builder=sampler_builder,
                callbacks=callbacks,
                sampler_checkpoint_dir=sampler_checkpoint_dir,
                directions=directions,
                direction=direction,
            ),
            enumerate(assigned_devices),
        )


def run_parallel_optuna_search(
    search_space_cfg: dict,
    sampler_builder: Callable[[int], BaseSampler],
    optimization_objective: Callable[[optuna.Trial, dict[str, Any], str], Any],
    study_name: str,
    journal_file: str,
    direction: str | None = None,
    directions: list[str] | None = None,
    n_workers: int = 1,
    devices: list[str] | None = None,
    max_search_trials: int | None = None,
    count_only_completed_trials: bool = False,
    sampler_checkpoint_dir: Path | None = None,
) -> optuna.Study:
    if direction is None and directions is None:
        raise ValueError("Either direction or directions must be set!")

    if direction is not None and directions is not None:
        raise ValueError("Greedy! Only set either direction or directions!")

    if devices is None:
        logger.warning(
            "Warning! devices is not specified and is set to cpu as default!"
        )
        devices = ["cpu"]

    if sampler_checkpoint_dir is not None:
        sampler_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # multiprocessing works only with journal file or postgres backend
    study = optuna.create_study(
        direction=direction,
        directions=directions,
        study_name=study_name,
        storage=_create_journal_storage(journal_file),
        sampler=None,
        load_if_exists=True,
    )

    n_completed = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
    # remove aborted trials (meant to clean up aborted trials at amplitUDE timeout)
    n_stale_failed = _fail_stale_running_trials(study=study)
    logger.info(
        f"Loaded study '{study_name}': {n_completed} completed trials; "
        f"marked {n_stale_failed} stale RUNNING trials as FAILED."
    )

    if n_workers > 1:
        if max_search_trials is not None:
            states = (TrialState.COMPLETE,) if count_only_completed_trials else None
            callbacks = [MaxTrialsCallback(max_search_trials, states=states)]
        else:
            callbacks = []

        _run_multiprocessing_search(
            study_name=study_name,
            sampler_builder=sampler_builder,
            optimization_objective=optimization_objective,
            direction=direction,
            directions=directions,
            journal_file=journal_file,
            devices=devices,
            search_space_cfg=search_space_cfg,
            n_workers=n_workers,
            callbacks=callbacks,
            sampler_checkpoint_dir=sampler_checkpoint_dir,
        )
    else:
        raise ValueError("Set n_workers > 1 with devices for multiprocessing")
    return study
