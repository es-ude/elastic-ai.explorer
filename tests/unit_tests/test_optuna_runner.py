import pickle
from types import SimpleNamespace

import pytest
from optuna.samplers import RandomSampler

from elasticai.explorer.parallel.optuna_runner import (
    _load_or_build_sampler,
    _sampler_checkpoint,
    _save_sampler_callback,
    assign_workers_to_devices,
)


def _builder(worker_idx: int) -> RandomSampler:
    return RandomSampler(seed=42 + worker_idx)


def test_load_or_build_sampler_no_checkpoint_dir():
    sampler = _load_or_build_sampler(
        sampler_builder=_builder,
        worker_idx=3,
        checkpoint_dir=None,
    )
    assert isinstance(sampler, RandomSampler)


def test_load_or_build_sampler_loads_existing(tmp_path):
    saved = RandomSampler(seed=42)
    sampler_checkpoint_path = _sampler_checkpoint(
        checkpoint_dir=tmp_path,
        worker_idx=3,
    )
    with open(sampler_checkpoint_path, "wb") as f:
        pickle.dump(saved, f)

    loaded = _load_or_build_sampler(
        sampler_builder=_builder,
        worker_idx=3,
        checkpoint_dir=tmp_path,
    )

    assert isinstance(loaded, RandomSampler)
    assert pickle.dumps(loaded) == pickle.dumps(saved)


def test_save_sampler_callback_writes_files(tmp_path):
    checkpoint_path = tmp_path / "sampler.pkl"
    sampler = RandomSampler(seed=42)

    fake_study = SimpleNamespace(sampler=sampler)
    callback = _save_sampler_callback(checkpoint_path=checkpoint_path)
    callback(study=fake_study, trial=None)

    assert checkpoint_path.exists()

    with open(checkpoint_path, "rb") as f:
        loaded = pickle.load(f)

    assert isinstance(loaded, RandomSampler)


@pytest.mark.parametrize(
    "n_workers, devices, expected",
    [
        (1, ["cpu"], ["cpu"]),
        (5, ["cpu"], ["cpu", "cpu", "cpu", "cpu", "cpu"]),
        (3, ["cuda:0", "cuda:1", "cuda:2"], ["cuda:0", "cuda:1", "cuda:2"]),
        (4, ["CPU", "cpu", "MPS"], ["cpu", "mps", "cpu", "mps"]),
        (5, ["cpu", "cuda:0", "cuda:1"], ["cpu", "cuda:0", "cuda:1", "cpu", "cuda:0"]),
    ],
)
def test_assign_workers_to_devices_valid(
    n_workers: int,
    devices: list[str],
    expected: list[str],
):
    assert assign_workers_to_devices(n_workers=n_workers, devices=devices) == expected


@pytest.mark.parametrize(
    "n_workers, devices",
    [
        (0, ["cpu"]),
        (-1, ["cpu"]),
        (2, []),
        (2, ["gpu:0"]),
        (2, ["cuda:0", "invalid"]),
        (2, ["cuda"]),
    ],
)
def test_assign_workers_to_devices_invalid(
    n_workers: int,
    devices: list[str],
):
    with pytest.raises(ValueError):
        assign_workers_to_devices(n_workers=n_workers, devices=devices)
