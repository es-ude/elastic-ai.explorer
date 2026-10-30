import optuna
import pytest
from optuna.trial import TrialState

from elasticai.explorer.hw_nas.search_space.exhaustive_sampler import ExhaustiveGridSampler

optuna.logging.set_verbosity(optuna.logging.ERROR)


def conditional_space(trial: optuna.Trial) -> tuple:
    depth1 = trial.suggest_categorical("b1_depth", [1, 2, 3])
    kernel = trial.suggest_categorical("b1_kernel", [3, 5])
    channels = trial.suggest_categorical("b1_channels", [6, 16, 32])

    depth2 = trial.suggest_categorical("b2_depth", [1, 2])
    layers = tuple(
        (
            trial.suggest_categorical(f"b2_{i}_width", [32, 64, 128]),
            trial.suggest_categorical(f"b2_{i}_act", ["relu", "sigmoid"]),
        )
        for i in range(depth2)
    )
    return depth1, kernel, channels, depth2, layers


EXPECTED_CONDITIONAL = 756


def run_study(objective, n_trials=None, **sampler_kwargs):
    study = optuna.create_study(
        sampler=ExhaustiveGridSampler(**sampler_kwargs), direction="maximize"
    )
    study.optimize(objective, n_trials=n_trials)
    return study


def collect(space_fn, n_trials=None, **sampler_kwargs):
    seen = []

    def objective(trial):
        seen.append(space_fn(trial))
        return float(len(seen))

    study = run_study(objective, n_trials=n_trials, **sampler_kwargs)
    return study, seen


@pytest.mark.parametrize("shuffle", [True, False])
def test_visits_every_configuration_exactly_once(shuffle):
    study, seen = collect(conditional_space, seed=0, shuffle=shuffle)

    assert len(study.trials) == EXPECTED_CONDITIONAL
    assert len(seen) == EXPECTED_CONDITIONAL
    assert len(set(seen)) == EXPECTED_CONDITIONAL


def test_study_stops_automatically_when_grid_is_exhausted():
    study, _ = collect(conditional_space, n_trials=10_000, seed=0)
    assert len(study.trials) == EXPECTED_CONDITIONAL


def test_runs_without_n_trials_and_terminates():
    study, _ = collect(conditional_space, n_trials=None, seed=0)
    assert len(study.trials) == EXPECTED_CONDITIONAL


def test_flat_space_is_cartesian_product():
    def space(trial):
        return (
            trial.suggest_categorical("a", [1, 2, 3]),
            trial.suggest_categorical("b", ["x", "y"]),
        )

    study, seen = collect(space, seed=0)
    assert len(study.trials) == 6
    assert set(seen) == {(a, b) for a in (1, 2, 3) for b in ("x", "y")}


def test_limited_trials_have_no_duplicates():
    study, seen = collect(conditional_space, n_trials=100, seed=1)

    assert len(study.trials) == 100
    assert len(set(seen)) == 100


def test_limited_trials_are_subset_of_full_grid():
    _, full = collect(conditional_space, seed=0)
    _, subset = collect(conditional_space, n_trials=50, seed=5)

    assert set(subset) <= set(full)


def test_shuffle_false_is_deterministic_depth_first():
    _, first = collect(conditional_space, n_trials=20, shuffle=False)
    _, second = collect(conditional_space, n_trials=20, shuffle=False)

    assert first == second
    assert first[0] == (1, 3, 6, 1, ((32, "relu"),))


def test_same_seed_gives_same_order():
    _, a = collect(conditional_space, n_trials=50, seed=42)
    _, b = collect(conditional_space, n_trials=50, seed=42)
    assert a == b


def test_different_seed_gives_different_order():
    _, a = collect(conditional_space, n_trials=50, seed=1)
    _, b = collect(conditional_space, n_trials=50, seed=2)
    assert a != b

def test_categorical_returns_choice_values_not_indices():
    def space(trial):
        return trial.suggest_categorical("act", ["relu", "sigmoid", "tanh"])

    _, seen = collect(space, seed=0)
    assert set(seen) == {"relu", "sigmoid", "tanh"}


def test_int_range_with_step_is_enumerated():
    def space(trial):
        return trial.suggest_int("width", 10, 50, step=10)

    _, seen = collect(space, seed=0)
    assert sorted(seen) == [10, 20, 30, 40, 50]


def test_float_with_step_is_enumerated():
    def space(trial):
        return trial.suggest_float("p", 0.0, 1.0, step=0.25)

    _, seen = collect(space, seed=0)
    assert sorted(seen) == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_float_without_step_raises_clear_error():
    def objective(trial):
        trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        return 0.0

    study = optuna.create_study(sampler=ExhaustiveGridSampler(seed=0))
    with pytest.raises(ValueError, match="step"):
        study.optimize(objective, n_trials=1)


def test_single_value_parameter_does_not_multiply_trials():
    def space(trial):
        trial.suggest_categorical("fixed", [1])  # nur eine Option
        return trial.suggest_categorical("a", [1, 2, 3])

    study, seen = collect(space, seed=0)
    assert len(study.trials) == 3
    assert sorted(seen) == [1, 2, 3]


def test_conditional_parameters_only_exist_when_sampled():
    study, _ = collect(conditional_space, seed=0)

    for trial in study.trials:
        depth2 = trial.params["b2_depth"]
        assert ("b2_1_width" in trial.params) == (depth2 == 2)
        assert ("b2_1_act" in trial.params) == (depth2 == 2)


def test_pruned_trials_count_as_visited_and_are_not_repeated():
    seen = []

    def objective(trial):
        a = trial.suggest_categorical("a", [1, 2, 3])
        b = trial.suggest_int("b", 1, 3)
        seen.append((a, b))
        if a == 2 and b == 3:
            raise optuna.TrialPruned()
        return float(a + b)

    study = run_study(objective, seed=0)

    assert len(study.trials) == 9
    assert len(set(seen)) == 9
    states = [t.state for t in study.trials]
    assert states.count(TrialState.PRUNED) == 1
    assert states.count(TrialState.COMPLETE) == 8


def test_prune_after_complete_sampling_does_not_hide_siblings():
    def objective(trial):
        depth = trial.suggest_categorical("depth", [1, 2, 3])
        widths = [trial.suggest_categorical(f"w{i}", [8, 16]) for i in range(depth)]
        if depth == 3 and widths[0] == 8:
            raise optuna.TrialPruned()
        return 1.0

    study = run_study(objective, seed=0)

    assert len(study.trials) == 2 + 4 + 8
    assert sum(t.state == TrialState.PRUNED for t in study.trials) == 4


def test_frozen_trial_replay_reproduces_configuration():
    seen = {}

    def objective(trial):
        config = conditional_space(trial)
        seen[trial.number] = config
        return float(trial.number)

    study = run_study(objective, n_trials=30, seed=3)

    best = study.best_trial
    assert conditional_space(best) == seen[best.number]


def test_no_leftover_trial_paths_after_run():
    sampler = ExhaustiveGridSampler(seed=0)
    study = optuna.create_study(sampler=sampler)
    study.optimize(lambda t: float(len(conditional_space(t))), n_trials=25)

    assert sampler._paths == {}
