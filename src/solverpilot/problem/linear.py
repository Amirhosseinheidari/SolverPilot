from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse

from .enums import ObjectiveSense, VariableDomain
from .hashing import data_hash_linear, structural_hash_linear


def _readonly_f64(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    arr = np.array(values, dtype=np.float64, copy=True)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.isnan(arr).any():
        raise ValueError(f"{name} must not contain NaN")
    arr.flags.writeable = False
    return arr


def _require_finite(values: np.ndarray, *, name: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")


def _require_bound_directions(lower: np.ndarray, upper: np.ndarray, *, label: str) -> None:
    if np.isposinf(lower).any():
        idx = int(np.flatnonzero(np.isposinf(lower))[0])
        raise ValueError(f"{label} lower bound at index {idx} cannot be +infinity")
    if np.isneginf(upper).any():
        idx = int(np.flatnonzero(np.isneginf(upper))[0])
        raise ValueError(f"{label} upper bound at index {idx} cannot be -infinity")


def _normalize_domains(domains: Iterable[VariableDomain | str], n: int) -> np.ndarray:
    vals = []
    for value in domains:
        try:
            vals.append(VariableDomain(value).value)
        except ValueError as exc:
            raise ValueError(f"unknown variable domain: {value!r}") from exc
    if len(vals) != n:
        raise ValueError(f"domains must have length {n}, got {len(vals)}")
    arr = np.asarray(vals, dtype="U16")
    arr.flags.writeable = False
    return arr


def _readonly_csr(A: sparse.spmatrix | np.ndarray) -> sparse.csr_matrix:
    matrix = sparse.csr_matrix(A, dtype=np.float64, copy=True)
    if matrix.ndim != 2:
        raise ValueError("A must be two-dimensional")
    if np.isnan(matrix.data).any() or np.isinf(matrix.data).any():
        raise ValueError("A coefficients must be finite")
    matrix.sum_duplicates()
    if not np.isfinite(matrix.data).all():
        raise ValueError("A coefficients must be finite after summing duplicates")
    matrix.eliminate_zeros()
    matrix.sort_indices()
    matrix.data.flags.writeable = False
    matrix.indices.flags.writeable = False
    matrix.indptr.flags.writeable = False
    return matrix


@dataclass(frozen=True, slots=True)
class LinearProblem:
    A: sparse.csr_matrix
    c: np.ndarray
    variable_lower: np.ndarray
    variable_upper: np.ndarray
    constraint_lower: np.ndarray
    constraint_upper: np.ndarray
    domains: np.ndarray
    objective_sense: ObjectiveSense = ObjectiveSense.MINIMIZE
    objective_offset: float = 0.0
    name: str | None = None
    metadata: dict[str, object] = field(default_factory=dict, compare=False, repr=False)
    structural_hash: str = field(init=False)
    data_hash: str = field(init=False)

    def __post_init__(self) -> None:
        A = _readonly_csr(self.A)
        c = _readonly_f64(self.c, name="c")
        vl = _readonly_f64(self.variable_lower, name="variable_lower")
        vu = _readonly_f64(self.variable_upper, name="variable_upper")
        cl = _readonly_f64(self.constraint_lower, name="constraint_lower")
        cu = _readonly_f64(self.constraint_upper, name="constraint_upper")
        sense = ObjectiveSense(self.objective_sense)

        m, n = A.shape
        if c.shape != (n,):
            raise ValueError(f"c must have shape ({n},), got {c.shape}")
        _require_finite(c, name="c")
        if vl.shape != (n,) or vu.shape != (n,):
            raise ValueError("variable bounds must match number of variables")
        if cl.shape != (m,) or cu.shape != (m,):
            raise ValueError("constraint bounds must match number of constraints")
        _require_bound_directions(vl, vu, label="variable")
        _require_bound_directions(cl, cu, label="constraint")
        if np.any(vl > vu):
            idx = int(np.flatnonzero(vl > vu)[0])
            raise ValueError(f"inconsistent variable bounds at index {idx}: lower > upper")
        if np.any(cl > cu):
            idx = int(np.flatnonzero(cl > cu)[0])
            raise ValueError(f"inconsistent constraint bounds at index {idx}: lower > upper")

        domains = _normalize_domains(self.domains, n)
        for j, domain in enumerate(domains):
            if domain == VariableDomain.BINARY.value:
                if vl[j] < 0.0 or vu[j] > 1.0:
                    raise ValueError(
                        f"binary variable {j} must have bounds inside [0, 1], "
                        f"got [{vl[j]}, {vu[j]}]"
                    )

        if not np.isfinite(float(self.objective_offset)):
            raise ValueError("objective_offset must be finite")

        object.__setattr__(self, "A", A)
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "variable_lower", vl)
        object.__setattr__(self, "variable_upper", vu)
        object.__setattr__(self, "constraint_lower", cl)
        object.__setattr__(self, "constraint_upper", cu)
        object.__setattr__(self, "domains", domains)
        object.__setattr__(self, "objective_sense", sense)
        object.__setattr__(self, "objective_offset", float(self.objective_offset))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "structural_hash", structural_hash_linear(A, domains))
        object.__setattr__(
            self,
            "data_hash",
            data_hash_linear(
                A,
                c,
                vl,
                vu,
                cl,
                cu,
                domains,
                sense.value,
                float(self.objective_offset),
            ),
        )

    @property
    def n_variables(self) -> int:
        return self.A.shape[1]

    @property
    def n_constraints(self) -> int:
        return self.A.shape[0]

    @property
    def nnz(self) -> int:
        return int(self.A.nnz)

    @property
    def has_integer_variables(self) -> bool:
        return bool(np.any(self.domains != VariableDomain.CONTINUOUS.value))

    @classmethod
    def from_data(
        cls,
        *,
        A: sparse.spmatrix | np.ndarray,
        c: Sequence[float] | np.ndarray,
        variable_lower: Sequence[float] | np.ndarray,
        variable_upper: Sequence[float] | np.ndarray,
        constraint_lower: Sequence[float] | np.ndarray,
        constraint_upper: Sequence[float] | np.ndarray,
        domains: Iterable[VariableDomain | str] | None = None,
        objective_sense: ObjectiveSense | str = ObjectiveSense.MINIMIZE,
        objective_offset: float = 0.0,
        name: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> "LinearProblem":
        n = sparse.csr_matrix(A).shape[1]
        if domains is None:
            domains = [VariableDomain.CONTINUOUS] * n
        return cls(
            A=A,
            c=np.asarray(c),
            variable_lower=np.asarray(variable_lower),
            variable_upper=np.asarray(variable_upper),
            constraint_lower=np.asarray(constraint_lower),
            constraint_upper=np.asarray(constraint_upper),
            domains=list(domains),
            objective_sense=ObjectiveSense(objective_sense),
            objective_offset=objective_offset,
            name=name,
            metadata={} if metadata is None else metadata,
        )
