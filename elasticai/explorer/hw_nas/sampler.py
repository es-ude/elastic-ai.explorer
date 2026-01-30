from abc import ABC, abstractmethod
from typing import (
    Optional,
    Sequence,
    Dict,
    Any,
    Callable,
    List,
)

import numpy as np

from optuna.samplers import (
    BaseSampler,
    RandomSampler,
    BruteForceSampler,
    GPSampler,
    TPESampler,
    CmaEsSampler,
    NSGAIISampler,
    NSGAIIISampler,
    QMCSampler,
)
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState
from optuna.distributions import BaseDistribution
from optuna.samplers.nsgaii import BaseCrossover


class Sampler(ABC):
    """Abstract sampler wrapper."""

    @abstractmethod
    def build(self) -> Any:
        pass


class RandomSamplerWrapper(Sampler):
    def __init__(self, seed: Optional[int] = None):
        self.seed: Optional[int] = seed

    def build(self) -> RandomSampler:
        return RandomSampler(seed=self.seed)


# Default functions from Optuna.
def default_gamma(x: int) -> int:
    return min(int(np.ceil(0.1 * x)), 25)


def default_weights(x: int) -> np.ndarray:
    if x == 0:
        return np.asarray([])
    elif x < 25:
        return np.ones(x)
    else:
        ramp = np.linspace(1.0 / x, 1.0, num=x - 25)
        flat = np.ones(25)
        return np.concatenate([ramp, flat], axis=0)


class TPESamplerWrapper(Sampler):

    def __init__(
        self,
        *,
        consider_prior: bool = True,
        prior_weight: float = 1.0,
        consider_magic_clip: bool = True,
        consider_endpoints: bool = False,
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
        gamma: Callable[[int], int] | None = default_gamma,
        weights: Callable[[int], np.ndarray] | None = default_weights,
        seed: Optional[int] = None,
        multivariate: bool = False,
        group: bool = False,
        warn_independent_sampling: bool = True,
        constant_liar: bool = False,
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
        categorical_distance_func: Dict[str, Callable[[Any, Any], float]] | None = None,
    ):
        self.consider_prior = consider_prior
        self.prior_weight = prior_weight
        self.consider_magic_clip = consider_magic_clip
        self.consider_endpoints = consider_endpoints
        self.n_startup_trials = n_startup_trials
        self.n_ei_candidates = n_ei_candidates
        self.gamma = gamma
        self.weights = weights
        self.seed = seed
        self.multivariate = multivariate
        self.group = group
        self.warn_independent_sampling = warn_independent_sampling
        self.constant_liar = constant_liar
        self.constraints_func = constraints_func
        self.categorical_distance_func = categorical_distance_func

    def build(self) -> TPESampler:
        return TPESampler(
            consider_prior=self.consider_prior,
            prior_weight=self.prior_weight,
            consider_magic_clip=self.consider_magic_clip,
            consider_endpoints=self.consider_endpoints,
            n_startup_trials=self.n_startup_trials,
            n_ei_candidates=self.n_ei_candidates,
            gamma=self.gamma,  # type: ignore
            weights=self.weights,  # type: ignore
            seed=self.seed,
            multivariate=self.multivariate,
            group=self.group,
            warn_independent_sampling=self.warn_independent_sampling,
            constant_liar=self.constant_liar,
            constraints_func=self.constraints_func,
            categorical_distance_func=self.categorical_distance_func,
        )


class CmaEsSamplerWrapper(Sampler):
    def __init__(
        self,
        x0: Dict[str, Any] | None = None,
        sigma0: float | None = None,
        n_startup_trials: int = 1,
        independent_sampler: BaseSampler | None = None,
        warn_independent_sampling: bool = True,
        seed: Optional[int] = None,
        *,
        consider_pruned_trials: bool = False,
        restart_strategy: str | None = None,
        popsize: int | None = None,
        inc_popsize: int = -1,
        use_separable_cma: bool = False,
        with_margin: bool = False,
        lr_adapt: bool = False,
        source_trials: List[FrozenTrial] | None = None,
    ):
        self.x0 = x0
        self.sigma0 = sigma0
        self.n_startup_trials = n_startup_trials
        self.independent_sampler = independent_sampler
        self.warn_independent_sampling = warn_independent_sampling
        self.seed = seed
        self.consider_pruned_trials = consider_pruned_trials
        self.restart_strategy = restart_strategy
        self.popsize = popsize
        self.inc_popsize = inc_popsize
        self.use_separable_cma = use_separable_cma
        self.with_margin = with_margin
        self.lr_adapt = lr_adapt
        self.source_trials = source_trials

    def build(self) -> CmaEsSampler:
        return CmaEsSampler(
            x0=self.x0,
            sigma0=self.sigma0,
            n_startup_trials=self.n_startup_trials,
            independent_sampler=self.independent_sampler,
            warn_independent_sampling=self.warn_independent_sampling,
            seed=self.seed,
            consider_pruned_trials=self.consider_pruned_trials,
            restart_strategy=self.restart_strategy,
            popsize=self.popsize,
            inc_popsize=self.inc_popsize,
            use_separable_cma=self.use_separable_cma,
            with_margin=self.with_margin,
            lr_adapt=self.lr_adapt,
            source_trials=self.source_trials,
        )


class NSGAIISamplerWrapper(Sampler):
    def __init__(
        self,
        *,
        population_size: int = 50,
        mutation_prob: float | None = None,
        crossover: BaseCrossover | None = None,
        crossover_prob: float = 0.9,
        swapping_prob: float = 0.5,
        seed: Optional[int] = None,
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
        elite_population_selection_strategy: (
            Callable[[Study, List[FrozenTrial]], List[FrozenTrial]] | None
        ) = None,
        child_generation_strategy: (
            Callable[
                [Study, Dict[str, BaseDistribution], List[FrozenTrial]], Dict[str, Any]
            ]
            | None
        ) = None,
        after_trial_strategy: (
            Callable[[Study, FrozenTrial, TrialState, Sequence[float] | None], None]
            | None
        ) = None,
    ):
        self.population_size = population_size
        self.mutation_prob = mutation_prob
        self.crossover = crossover
        self.crossover_prob = crossover_prob
        self.swapping_prob = swapping_prob
        self.seed = seed
        self.constraints_func = constraints_func
        self.elite_population_selection_strategy = elite_population_selection_strategy
        self.child_generation_strategy = child_generation_strategy
        self.after_trial_strategy = after_trial_strategy

    def build(self) -> NSGAIISampler:
        return NSGAIISampler(
            population_size=self.population_size,
            mutation_prob=self.mutation_prob,
            crossover=self.crossover,
            crossover_prob=self.crossover_prob,
            swapping_prob=self.swapping_prob,
            seed=self.seed,
            constraints_func=self.constraints_func,
            elite_population_selection_strategy=self.elite_population_selection_strategy,
            child_generation_strategy=self.child_generation_strategy,
            after_trial_strategy=self.after_trial_strategy,
        )


class NSGAIIISamplerWrapper(Sampler):
    def __init__(
        self,
        *,
        population_size: int = 50,
        mutation_prob: float | None = None,
        crossover: BaseCrossover | None = None,
        crossover_prob: float = 0.9,
        swapping_prob: float = 0.5,
        seed: Optional[int] = None,
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
        reference_points: np.ndarray | None = None,
        dividing_parameter: int = 3,
        elite_population_selection_strategy: (
            Callable[[Study, List[FrozenTrial]], List[FrozenTrial]] | None
        ) = None,
        child_generation_strategy: (
            Callable[
                [Study, Dict[str, BaseDistribution], List[FrozenTrial]], Dict[str, Any]
            ]
            | None
        ) = None,
        after_trial_strategy: (
            Callable[[Study, FrozenTrial, TrialState, Sequence[float] | None], None]
            | None
        ) = None,
    ):
        self.population_size = population_size
        self.mutation_prob = mutation_prob
        self.crossover = crossover
        self.crossover_prob = crossover_prob
        self.swapping_prob = swapping_prob
        self.seed = seed
        self.constraints_func = constraints_func
        self.reference_points = reference_points
        self.dividing_parameter = dividing_parameter
        self.elite_population_selection_strategy = elite_population_selection_strategy
        self.child_generation_strategy = child_generation_strategy
        self.after_trial_strategy = after_trial_strategy

    def build(self) -> NSGAIIISampler:
        return NSGAIIISampler(
            population_size=self.population_size,
            mutation_prob=self.mutation_prob,
            crossover=self.crossover,
            crossover_prob=self.crossover_prob,
            swapping_prob=self.swapping_prob,
            seed=self.seed,
            constraints_func=self.constraints_func,
            reference_points=self.reference_points,
            dividing_parameter=self.dividing_parameter,
            elite_population_selection_strategy=self.elite_population_selection_strategy,
            child_generation_strategy=self.child_generation_strategy,
            after_trial_strategy=self.after_trial_strategy,
        )


class QMCSamplerWrapper(Sampler):
    def __init__(
        self,
        *,
        qmc_type: str = "sobol",
        scramble: bool = False,
        seed: Optional[int] = None,
        independent_sampler: BaseSampler | None = None,
        warn_asynchronous_seeding: bool = True,
        warn_independent_sampling: bool = True,
    ):
        self.qmc_type = qmc_type
        self.scramble = scramble
        self.seed = seed
        self.independent_sampler = independent_sampler
        self.warn_asynchronous_seeding = warn_asynchronous_seeding
        self.warn_independent_sampling = warn_independent_sampling

    def build(self) -> QMCSampler:
        return QMCSampler(
            qmc_type=self.qmc_type,
            scramble=self.scramble,
            seed=self.seed,
            independent_sampler=self.independent_sampler,
            warn_asynchronous_seeding=self.warn_asynchronous_seeding,
            warn_independent_sampling=self.warn_independent_sampling,
        )


class BruteForceSamplerWrapper(Sampler):
    def __init__(
        self,
        seed: int | None = None,
        avoid_premature_stop: bool = False,
    ):
        self.seed = seed
        self.avoid_premature_stop = avoid_premature_stop

    def build(self):
        return BruteForceSampler(
            seed=self.seed,
            avoid_premature_stop=self.avoid_premature_stop,
        )


class GPSamplerWrapper(Sampler):
    def __init__(
        self,
        *,
        seed: int | None = None,
        independent_sampler: BaseSampler | None = None,
        n_startup_trials: int = 10,
        deterministic_objective: bool = False,
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
    ):
        self.seed = seed
        self.independent_sampler = independent_sampler
        self.n_startup_trials = n_startup_trials
        self.deterministic_objective = deterministic_objective
        self.constraints_func = constraints_func

    def build(self):
        return GPSampler(
            seed=self.seed,
            independent_sampler=self.independent_sampler,
            n_startup_trials=self.n_startup_trials,
            deterministic_objective=self.deterministic_objective,
            constraints_func=self.constraints_func,
        )
