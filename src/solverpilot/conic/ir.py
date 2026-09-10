from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from scipy import sparse

from solverpilot.problem import ConvexityStatus, ObjectiveSense, QuadraticProblem, VariableDomain
from solverpilot.problem.linear import _readonly_csr, _readonly_f64, _require_bound_directions, _require_finite
from solverpilot.problem.quadratic import _canonicalize_hessian, _check_convexity
from solverpilot.problem.hashing import _hash_parts, hash_sparse_matrix


class ConeKind(str, Enum):
    SECOND_ORDER = "second_order"
    ROTATED_SECOND_ORDER = "rotated_second_order"
    POSITIVE_SEMIDEFINITE = "positive_semidefinite"


def _readonly_shape(value: tuple[int, ...]) -> tuple[int, ...]:
    out = tuple(int(x) for x in value)
    if not out or any(x <= 0 for x in out):
        raise ValueError("cone output_shape must contain positive dimensions")
    return out


def _hash_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _json_float(value: float) -> float | str:
    v = float(value)
    if np.isposinf(v): return "+inf"
    if np.isneginf(v): return "-inf"
    if np.isnan(v): raise ValueError("NaN cannot be serialized")
    return v


def _csr_dict(matrix: sparse.csr_matrix) -> dict[str, Any]:
    m = sparse.csr_matrix(matrix)
    return {
        "shape": [int(m.shape[0]), int(m.shape[1])],
        "indptr": [int(x) for x in m.indptr],
        "indices": [int(x) for x in m.indices],
        "data": [float(x) for x in m.data],
    }


@dataclass(frozen=True, slots=True)
class ConeAffineBlock:
    """Affine cone membership block ``F x + g in K``.

    For vector cones, ``output_shape`` is ``(dim,)``. For PSD cones it is
    ``(n, n)`` and the flattened affine map is row-major. The block itself is
    solver-independent; backend-specific vectorization/scaling is performed by
    the adapter.
    """

    kind: ConeKind
    F: sparse.csr_matrix
    g: np.ndarray
    output_shape: tuple[int, ...]
    source_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)
    structural_hash: str = field(init=False)
    data_hash: str = field(init=False)

    def __post_init__(self) -> None:
        kind = ConeKind(self.kind)
        F = _readonly_csr(self.F)
        g = _readonly_f64(self.g, name="cone offset")
        _require_finite(g, name="cone offset")
        shape = _readonly_shape(self.output_shape)
        expected = int(np.prod(shape, dtype=np.int64))
        if F.shape[0] != expected:
            raise ValueError(f"cone F has {F.shape[0]} rows but output_shape requires {expected}")
        if g.shape != (expected,):
            raise ValueError(f"cone g must have shape ({expected},), got {g.shape}")
        if kind is ConeKind.SECOND_ORDER:
            if len(shape) != 1 or shape[0] < 2:
                raise ValueError("second-order cone requires vector dimension >= 2")
        elif kind is ConeKind.ROTATED_SECOND_ORDER:
            if len(shape) != 1 or shape[0] < 3:
                raise ValueError("rotated second-order cone requires vector dimension >= 3")
        elif kind is ConeKind.POSITIVE_SEMIDEFINITE:
            if len(shape) != 2 or shape[0] != shape[1]:
                raise ValueError("PSD cone requires a square matrix output_shape")
        else:  # pragma: no cover - Enum guards this
            raise ValueError(kind)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "F", F)
        object.__setattr__(self, "g", g)
        object.__setattr__(self, "output_shape", shape)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        structural = _hash_parts([
            kind.value.encode(),
            json.dumps(shape).encode(),
            hash_sparse_matrix(F.astype(bool)).encode(),
        ])
        data = _hash_parts([
            structural.encode(),
            hash_sparse_matrix(F).encode(),
            _hash_array(g).encode(),
        ])
        object.__setattr__(self, "structural_hash", structural)
        object.__setattr__(self, "data_hash", data)

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "F": _csr_dict(self.F),
            "g": [float(x) for x in self.g],
            "output_shape": list(self.output_shape),
            "source_id": self.source_id,
            "metadata": dict(self.metadata),
            "structural_hash": self.structural_hash,
            "data_hash": self.data_hash,
        }

    @property
    def n_variables(self) -> int:
        return int(self.F.shape[1])

    @property
    def dimension(self) -> int:
        return int(self.F.shape[0])

    def value(self, x: np.ndarray) -> np.ndarray:
        vec = np.asarray(self.F @ np.asarray(x, dtype=np.float64).reshape(-1), dtype=np.float64).reshape(-1) + self.g
        return vec.reshape(self.output_shape)


@dataclass(frozen=True, slots=True)
class ConicProblem:
    """Convex quadratic objective + linear rows + affine cone memberships."""

    P: sparse.csr_matrix
    q: np.ndarray
    A: sparse.csr_matrix
    variable_lower: np.ndarray
    variable_upper: np.ndarray
    constraint_lower: np.ndarray
    constraint_upper: np.ndarray
    cones: tuple[ConeAffineBlock, ...]
    objective_sense: ObjectiveSense = ObjectiveSense.MINIMIZE
    objective_offset: float = 0.0
    name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)
    convexity_status: ConvexityStatus = ConvexityStatus.UNKNOWN
    structural_hash: str = field(init=False)
    data_hash: str = field(init=False)

    def __post_init__(self) -> None:
        P_raw = _readonly_csr(self.P)
        q = _readonly_f64(self.q, name="q")
        A = _readonly_csr(self.A)
        vl = _readonly_f64(self.variable_lower, name="variable_lower")
        vu = _readonly_f64(self.variable_upper, name="variable_upper")
        cl = _readonly_f64(self.constraint_lower, name="constraint_lower")
        cu = _readonly_f64(self.constraint_upper, name="constraint_upper")
        sense = ObjectiveSense(self.objective_sense)
        if sense is not ObjectiveSense.MINIMIZE:
            raise ValueError("P6 ConicProblem supports convex minimization only")
        m, n = A.shape
        if P_raw.shape != (n, n):
            raise ValueError(f"P must have shape ({n}, {n})")
        P = _canonicalize_hessian(P_raw)
        if q.shape != (n,) or vl.shape != (n,) or vu.shape != (n,):
            raise ValueError("q/variable bounds must match n_variables")
        _require_finite(q, name="q")
        if cl.shape != (m,) or cu.shape != (m,):
            raise ValueError("linear constraint bounds must match A rows")
        if np.any(vl > vu) or np.any(cl > cu):
            raise ValueError("inconsistent bounds")
        _require_bound_directions(vl, vu, label="variable")
        _require_bound_directions(cl, cu, label="constraint")
        if not np.isfinite(float(self.objective_offset)):
            raise ValueError("objective_offset must be finite")
        cones = tuple(self.cones)
        if not cones:
            raise ValueError("ConicProblem requires at least one cone block")
        for block in cones:
            if block.n_variables != n:
                raise ValueError("cone block variable dimension mismatch")
        status = ConvexityStatus(self.convexity_status)
        if status is ConvexityStatus.UNKNOWN:
            status = _check_convexity(P)
        elif status is ConvexityStatus.CONFIRMED and _check_convexity(P) is not ConvexityStatus.CONFIRMED:
            raise ValueError("conic quadratic objective is not independently confirmed convex")
        if status is ConvexityStatus.REJECTED:
            raise ValueError("conic quadratic objective is not convex")
        object.__setattr__(self, "P", P)
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "variable_lower", vl)
        object.__setattr__(self, "variable_upper", vu)
        object.__setattr__(self, "constraint_lower", cl)
        object.__setattr__(self, "constraint_upper", cu)
        object.__setattr__(self, "cones", cones)
        object.__setattr__(self, "objective_sense", sense)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "convexity_status", status)
        structural = _hash_parts([
            b"solverpilot-conic-ir-v1",
            hash_sparse_matrix(P.astype(bool)).encode(),
            hash_sparse_matrix(A.astype(bool)).encode(),
            *(c.structural_hash.encode() for c in cones),
        ])
        data = _hash_parts([
            structural.encode(),
            hash_sparse_matrix(P).encode(),
            _hash_array(q).encode(),
            hash_sparse_matrix(A).encode(),
            _hash_array(vl).encode(), _hash_array(vu).encode(),
            _hash_array(cl).encode(), _hash_array(cu).encode(),
            *(c.data_hash.encode() for c in cones),
            sense.value.encode(), float(self.objective_offset).hex().encode(),
        ])
        object.__setattr__(self, "structural_hash", structural)
        object.__setattr__(self, "data_hash", data)

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "solverpilot.conic-ir.v1",
            "P": _csr_dict(self.P),
            "q": [float(x) for x in self.q],
            "A": _csr_dict(self.A),
            "variable_lower": [_json_float(x) for x in self.variable_lower],
            "variable_upper": [_json_float(x) for x in self.variable_upper],
            "constraint_lower": [_json_float(x) for x in self.constraint_lower],
            "constraint_upper": [_json_float(x) for x in self.constraint_upper],
            "cones": [c.to_canonical_dict() for c in self.cones],
            "objective_sense": self.objective_sense.value,
            "objective_offset": float(self.objective_offset),
            "name": self.name,
            "metadata": dict(self.metadata),
            "convexity_status": self.convexity_status.value,
            "structural_hash": self.structural_hash,
            "data_hash": self.data_hash,
        }

    @property
    def n_variables(self) -> int:
        return int(self.A.shape[1])

    @property
    def n_linear_constraints(self) -> int:
        return int(self.A.shape[0])

    @property
    def n_cone_constraints(self) -> int:
        return len(self.cones)

    def objective_value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        return float(0.5 * x @ (self.P @ x) + self.q @ x + self.objective_offset)

    @classmethod
    def from_data(
        cls,
        *,
        P: sparse.spmatrix | np.ndarray,
        q: np.ndarray,
        A: sparse.spmatrix | np.ndarray,
        variable_lower: np.ndarray,
        variable_upper: np.ndarray,
        constraint_lower: np.ndarray,
        constraint_upper: np.ndarray,
        cones: tuple[ConeAffineBlock, ...] | list[ConeAffineBlock],
        objective_offset: float = 0.0,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        verify_convexity: bool = True,
    ) -> "ConicProblem":
        P_csr = sparse.csr_matrix(P, dtype=np.float64)
        status = _check_convexity(P_csr) if verify_convexity else ConvexityStatus.UNKNOWN
        return cls(
            P=P_csr,
            q=np.asarray(q, dtype=np.float64),
            A=sparse.csr_matrix(A, dtype=np.float64),
            variable_lower=np.asarray(variable_lower, dtype=np.float64),
            variable_upper=np.asarray(variable_upper, dtype=np.float64),
            constraint_lower=np.asarray(constraint_lower, dtype=np.float64),
            constraint_upper=np.asarray(constraint_upper, dtype=np.float64),
            cones=tuple(cones),
            objective_offset=float(objective_offset),
            name=name,
            metadata={} if metadata is None else metadata,
            convexity_status=status,
        )
