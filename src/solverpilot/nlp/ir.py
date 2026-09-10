from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from solverpilot.model.expression import ExprNode


@dataclass(frozen=True, slots=True)
class NLPConstraintBlock:
    source_id: str
    node: ExprNode
    lower: np.ndarray
    upper: np.ndarray
    shape: tuple[int, ...]
    parameter_dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        lo = np.asarray(self.lower, dtype=float).reshape(-1).copy()
        hi = np.asarray(self.upper, dtype=float).reshape(-1).copy()
        if lo.shape != hi.shape:
            raise ValueError("constraint lower/upper shape mismatch")
        lo.flags.writeable = False; hi.flags.writeable = False
        object.__setattr__(self, "lower", lo); object.__setattr__(self, "upper", hi)


@dataclass(frozen=True, slots=True)
class NLPProblem:
    n_variables: int
    variable_lower: np.ndarray
    variable_upper: np.ndarray
    variable_layout: Mapping[str, tuple[int, tuple[int, ...]]]
    parameter_values: Mapping[str, np.ndarray]
    objective_node: ExprNode
    constraints: tuple[NLPConstraintBlock, ...]
    objective_sense: str
    name: str | None = None
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        vl = np.asarray(self.variable_lower, dtype=float).reshape(-1).copy()
        vu = np.asarray(self.variable_upper, dtype=float).reshape(-1).copy()
        if vl.shape != (self.n_variables,) or vu.shape != (self.n_variables,):
            raise ValueError("variable bound shape mismatch")
        if np.any(vl > vu):
            raise ValueError("variable lower bound exceeds upper bound")
        vl.flags.writeable = False; vu.flags.writeable = False
        object.__setattr__(self, "variable_lower", vl); object.__setattr__(self, "variable_upper", vu)
        object.__setattr__(self, "variable_layout", MappingProxyType(dict(self.variable_layout)))
        pv = {k: np.asarray(v, dtype=float).copy() for k, v in self.parameter_values.items()}
        for v in pv.values(): v.flags.writeable = False
        object.__setattr__(self, "parameter_values", MappingProxyType(pv))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def n_constraints(self) -> int:
        return int(sum(c.lower.size for c in self.constraints))

    @property
    def constraint_lower(self) -> np.ndarray:
        if not self.constraints: return np.empty(0, dtype=float)
        return np.concatenate([c.lower for c in self.constraints])

    @property
    def constraint_upper(self) -> np.ndarray:
        if not self.constraints: return np.empty(0, dtype=float)
        return np.concatenate([c.upper for c in self.constraints])
