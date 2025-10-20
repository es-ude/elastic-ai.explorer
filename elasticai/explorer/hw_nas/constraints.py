import logging
from elasticai.explorer.hw_nas.estimators import Estimator
from dataclasses import dataclass
from typing import Callable, Dict, List

logger = logging.getLogger("explorer.constraints")
Comparator = Callable[[float, float], bool]


@dataclass
class _HardConstraint:
    comparison_operator: Comparator
    constraint_value: float


@dataclass
class _SoftConstraint:
    penalty_fn: Callable[[float, float | None], float]
    constraint_value: float | None
    estimate_transform: Callable[[float], float]
    is_reward: bool = False
    weight: float = 1.0


class ConstraintRegistry:
    def __init__(self):
        self._constraints: Dict[Estimator, List[_HardConstraint | _SoftConstraint]] = {}

    def get_hard_constraints(self, estimator: Estimator) -> List[_HardConstraint]:
        return [
            (c) for c in self._constraints[estimator] if isinstance(c, _HardConstraint)
        ]

    def get_soft_constraints(self, estimator: Estimator) -> List[_SoftConstraint]:
        return [
            (c) for c in self._constraints[estimator] if isinstance(c, _SoftConstraint)
        ]

    def get_estimators(self) -> List[Estimator]:
        return list(self._constraints.keys())

    def register_hard_constraint(
        self, estimator: Estimator, operator: Comparator, value: float
    ):
        if not (estimator in self._constraints):
            self._constraints[estimator] = []

        self._constraints[estimator].append(_HardConstraint(operator, value))

    def register_soft_constraint(
        self,
        estimator: Estimator,
        penalty_fn: Callable[[float, float | None], float] = (lambda x, y: x),
        constraint_value: float | None = None,
        estimate_transform: Callable[[float], float] = lambda x: x,
        weight: float = 1.0,
        is_reward: bool = False,
    ):
        """
        Soft constraints are evaluated like this:
        penalty_value = penalty_fn(transform(estimate), constraint_value)
        """
        if not (estimator in self._constraints):
            self._constraints[estimator] = []
        if not estimate_transform:
            estimate_transform = lambda x: x

        self._constraints[estimator].append(
            _SoftConstraint(
                penalty_fn=penalty_fn,
                constraint_value=constraint_value,
                weight=weight,
                estimate_transform=estimate_transform,
                is_reward=is_reward,
            )
        )

    def __iter__(self):
        return iter(self._constraints)

    def __len__(self):
        return len(self._constraints)
