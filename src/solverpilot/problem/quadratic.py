from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, eigsh

from .enums import ConvexityStatus, ObjectiveSense, VariableDomain
from .hashing import _hash_parts, hash_sparse_matrix
from .linear import LinearProblem, _readonly_csr


def _canonicalize_hessian(
    P: sparse.csr_matrix,
    *,
    atol: float = 1e-12,
    rtol: float = 1e-12,
) -> sparse.csr_matrix:
    """Return the single symmetric Hessian used by every SolverPilot component.

    Symmetry is checked *pairwise*.  An unrelated large coefficient is therefore
    unable to relax the tolerance for a small asymmetric pair.  Once accepted,
    the Hessian is explicitly symmetrized so hash/objective/backend semantics all
    consume the same matrix.
    """
    diff = (P - P.T).tocoo()
    if diff.nnz:
        rows = diff.row
        cols = diff.col
        pij = np.asarray(P[rows, cols]).reshape(-1)
        pji = np.asarray(P[cols, rows]).reshape(-1)
        scale = np.maximum.reduce(
            [np.ones(diff.nnz, dtype=np.float64), np.abs(pij), np.abs(pji)]
        )
        allowed = float(atol) + float(rtol) * scale
        if np.any(np.abs(diff.data) > allowed):
            raise ValueError("P must be symmetric within pairwise numerical tolerance")
    return _readonly_csr(((P + P.T) * 0.5).tocsr())


def _scaled_psd_matrix(P: sparse.csr_matrix) -> sparse.csr_matrix:
    """Diagonal congruence scaling that preserves positive semidefiniteness.

    Scaling each row/column by its own magnitude prevents a huge, unrelated
    coefficient from masking a materially negative eigenvalue elsewhere.
    """
    if P.shape[0] == 0:
        return P
    row_max = np.asarray(abs(P).max(axis=1).toarray(), dtype=np.float64).reshape(-1)
    row_max = np.maximum(1.0, row_max)
    inv_sqrt = 1.0 / np.sqrt(row_max)
    D = sparse.diags(inv_sqrt, offsets=0, format="csr")
    scaled = (D @ P @ D).tocsr()
    scaled.sum_duplicates()
    scaled.eliminate_zeros()
    scaled.sort_indices()
    return scaled


def _check_convexity(
    P: sparse.csr_matrix,
    *,
    atol: float = 1e-10,
    rtol: float = 1e-9,
) -> ConvexityStatus:
    n = P.shape[0]
    if n == 0 or P.nnz == 0:
        return ConvexityStatus.CONFIRMED
    try:
        sym = _canonicalize_hessian(P)
    except ValueError:
        return ConvexityStatus.REJECTED
    scaled = _scaled_psd_matrix(sym)
    row_sums = np.asarray(abs(scaled).sum(axis=1), dtype=np.float64).reshape(-1)
    scale = max(1.0, float(np.max(row_sums)) if row_sums.size else 0.0)
    # The matrix has already undergone a PSD-preserving congruence scaling.  The
    # remaining tolerance is therefore about the local normalized matrix, not the
    # largest raw coefficient in the original problem.
    tol = float(atol) + float(rtol) * scale
    try:
        if n <= 256:
            lam_min = float(np.linalg.eigvalsh(scaled.toarray())[0])
        else:
            lam_min = float(eigsh(scaled, k=1, which="SA", return_eigenvectors=False)[0])
    except (np.linalg.LinAlgError, ArpackError, ArpackNoConvergence):
        return ConvexityStatus.UNKNOWN
    return ConvexityStatus.CONFIRMED if lam_min >= -tol else ConvexityStatus.REJECTED


def require_confirmed_convexity(problem: "QuadraticProblem") -> None:
    """Fail closed before any backend consumes a quadratic problem directly."""
    if problem.convexity_status is not ConvexityStatus.CONFIRMED:
        raise ValueError("unverified quadratic problem cannot be solved as convex_qp")


@dataclass(frozen=True, slots=True)
class QuadraticProblem:
    linear: LinearProblem
    P: sparse.csr_matrix
    convexity_status: ConvexityStatus = ConvexityStatus.UNKNOWN
    metadata: dict[str, object] = field(default_factory=dict, compare=False, repr=False)
    structural_hash: str = field(init=False)
    data_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.linear.has_integer_variables:
            raise ValueError("v0.1 QuadraticProblem supports continuous convex QP only")
        P_raw = _readonly_csr(self.P)
        n = self.linear.n_variables
        if P_raw.shape != (n, n):
            raise ValueError(f"P must have shape ({n}, {n}), got {P_raw.shape}")
        P = _canonicalize_hessian(P_raw)
        status = ConvexityStatus(self.convexity_status)
        if status is ConvexityStatus.REJECTED:
            raise ValueError("nonconvex quadratic objectives are out of scope for v0.1")
        if status is ConvexityStatus.CONFIRMED:
            checked = _check_convexity(P)
            if checked is not ConvexityStatus.CONFIRMED:
                raise ValueError("P is not independently confirmed positive semidefinite")
        object.__setattr__(self, "P", P)
        object.__setattr__(self, "convexity_status", status)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(
            self,
            "structural_hash",
            _hash_parts([self.linear.structural_hash.encode(), hash_sparse_matrix(P.astype(bool)).encode()]),
        )
        object.__setattr__(
            self,
            "data_hash",
            _hash_parts([self.linear.data_hash.encode(), hash_sparse_matrix(P).encode()]),
        )

    @property
    def n_variables(self) -> int:
        return self.linear.n_variables

    @property
    def n_constraints(self) -> int:
        return self.linear.n_constraints

    @property
    def objective_sense(self) -> ObjectiveSense:
        """Expose the common problem-level objective sense used by runtime code."""
        return self.linear.objective_sense

    @property
    def objective_offset(self) -> float:
        return float(self.linear.objective_offset)

    @classmethod
    def from_data(
        cls,
        *,
        P: sparse.spmatrix | np.ndarray,
        A: sparse.spmatrix | np.ndarray,
        q: Sequence[float] | np.ndarray,
        variable_lower: Sequence[float] | np.ndarray,
        variable_upper: Sequence[float] | np.ndarray,
        constraint_lower: Sequence[float] | np.ndarray,
        constraint_upper: Sequence[float] | np.ndarray,
        objective_sense: ObjectiveSense | str = ObjectiveSense.MINIMIZE,
        objective_offset: float = 0.0,
        verify_convexity: bool = True,
        metadata: dict[str, object] | None = None,
    ) -> "QuadraticProblem":
        sense = ObjectiveSense(objective_sense)
        if sense is ObjectiveSense.MAXIMIZE:
            raise ValueError("v0.1 QuadraticProblem currently supports convex minimization only")
        linear = LinearProblem.from_data(
            A=A,
            c=q,
            variable_lower=variable_lower,
            variable_upper=variable_upper,
            constraint_lower=constraint_lower,
            constraint_upper=constraint_upper,
            domains=[VariableDomain.CONTINUOUS] * sparse.csr_matrix(A).shape[1],
            objective_sense=sense,
            objective_offset=objective_offset,
            metadata={} if metadata is None else metadata,
        )
        P_csr = _canonicalize_hessian(_readonly_csr(P))
        status = _check_convexity(P_csr) if verify_convexity else ConvexityStatus.UNKNOWN
        if status is ConvexityStatus.REJECTED:
            raise ValueError("P is not positive semidefinite within scale-aware tolerance")
        return cls(linear=linear, P=P_csr, convexity_status=status, metadata={} if metadata is None else metadata)
