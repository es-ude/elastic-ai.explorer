import logging

from elasticai.explorer.hw_nas.estimators import Estimator
from dataclasses import dataclass
from typing import Callable, Dict, List

logger = logging.getLogger("explorer.constraints")
Comparator = Callable[[float, float], bool]
Transform = Callable[[float], float]


@dataclass
class _Objective:
    transform: Transform | None
    weight: float = 1.0


@dataclass
class _HardConstraint:
    comparator: Comparator
    constraint_value: float


@dataclass
class _SoftConstraint:
    comparator: Comparator
    constraint_value: float
    penalty_fn: Callable[[float, float], float]
    penalty_estimate_transform: Callable[[float], float]
    penalty_weight: float = 1.0


class OptimizationCriteriaRegistry:
    def __init__(self):
        self._optimization_criteria: Dict[
            Estimator, List[_Objective | _HardConstraint | _SoftConstraint]
        ] = {}

    def get_hard_constraints(self, estimator: Estimator) -> List[_HardConstraint]:
        return [
            (c)
            for c in self._optimization_criteria[estimator]
            if isinstance(c, _HardConstraint)
        ]

    def get_soft_constraints(self, estimator: Estimator) -> List[_SoftConstraint]:
        return [
            (c)
            for c in self._optimization_criteria[estimator]
            if isinstance(c, _SoftConstraint)
        ]

    def get_objectives(self, estimator: Estimator) -> List[_Objective]:
        return [
            (c)
            for c in self._optimization_criteria[estimator]
            if isinstance(c, _Objective)
        ]

    def get_estimators(self) -> List[Estimator]:
        return list(self._optimization_criteria.keys())

    def get_criteria(
        self, estimator: Estimator
    ) -> List[_Objective | _HardConstraint | _SoftConstraint]:
        return self._optimization_criteria[estimator]

    def register_objective(
        self,
        estimator: Estimator,
        transform: Transform | None = None,
        weight: float = 1.0,
    ):
        if not (estimator in self._optimization_criteria):
            self._optimization_criteria[estimator] = []
        self._optimization_criteria[estimator].append(_Objective(transform, weight))

    def register_hard_constraint(
        self, estimator: Estimator, operator: Comparator, value: float
    ):
        if not (estimator in self._optimization_criteria):
            self._optimization_criteria[estimator] = []

        self._optimization_criteria[estimator].append(_HardConstraint(operator, value))

    def register_soft_constraint(
        self,
        estimator: Estimator,
        comparator: Comparator,
        constraint_value: float = 0,
        penalty_fn: Callable[[float, float], float] = (lambda x, y: x),
        penalty_estimate_transform: Callable[[float], float] = lambda x: x,
        penalty_weight: float = 1.0,
    ):
        """
        Soft constraints will be evaluated like this:
        if(comparator(estimate, boundary_value)):
            penalty_value += weight * penalty_fn(transform(estimate), boundary_value)
        """
        if not (estimator in self._optimization_criteria):
            self._optimization_criteria[estimator] = []

        self._optimization_criteria[estimator].append(
            _SoftConstraint(
                comparator=comparator,
                constraint_value=constraint_value,
                penalty_fn=penalty_fn,
                penalty_estimate_transform=penalty_estimate_transform,
                penalty_weight=penalty_weight,
            )
        )

    def __iter__(self):
        return iter(self._optimization_criteria)

    def __len__(self):
        return len(self._optimization_criteria)
