from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import sparse

from solverpilot.problem import ConvexityStatus, LinearProblem, QuadraticProblem, VariableDomain


@dataclass(frozen=True, slots=True)
class DistributionStats:
    minimum: float
    q25: float
    median: float
    mean: float
    q75: float
    maximum: float


@dataclass(frozen=True, slots=True)
class ProblemFingerprint:
    problem_class: str
    n_variables: int
    n_constraints: int
    nnz_a: int
    density_a: float
    binary_fraction: float
    integer_fraction: float
    equality_fraction: float
    fully_bounded_variable_fraction: float
    free_variable_fraction: float
    objective_nnz_fraction: float
    coefficient_dynamic_range_log10: float | None
    objective_dynamic_range_log10: float | None
    rhs_dynamic_range_log10: float | None
    row_nnz: DistributionStats
    col_nnz: DistributionStats
    singleton_rows: int
    singleton_cols: int
    positive_coefficient_fraction: float
    negative_coefficient_fraction: float
    quadratic_nnz: int | None
    quadratic_density: float | None
    convexity_status: str | None
    structural_hash: str
    estimated_numeric_bytes: int


def _distribution(values: np.ndarray) -> DistributionStats:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return DistributionStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    q25, med, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    return DistributionStats(
        minimum=float(np.min(values)),
        q25=float(q25),
        median=float(med),
        mean=float(np.mean(values)),
        q75=float(q75),
        maximum=float(np.max(values)),
    )


def _dynamic_range_log10(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    finite_nonzero = np.abs(values[np.isfinite(values) & (values != 0.0)])
    if finite_nonzero.size == 0:
        return None
    lo = float(np.min(finite_nonzero))
    hi = float(np.max(finite_nonzero))
    if lo <= 0.0 or hi <= 0.0:
        return None
    return float(math.log10(hi / lo))


def _estimated_bytes_linear(problem: LinearProblem) -> int:
    A = problem.A
    sparse_bytes = int(A.data.nbytes + A.indices.nbytes + A.indptr.nbytes)
    dense_bytes = sum(
        int(arr.nbytes)
        for arr in (
            problem.c,
            problem.variable_lower,
            problem.variable_upper,
            problem.constraint_lower,
            problem.constraint_upper,
            problem.domains,
        )
    )
    return sparse_bytes + dense_bytes


def inspect_problem(problem: LinearProblem | QuadraticProblem) -> ProblemFingerprint:
    if isinstance(problem, QuadraticProblem):
        if problem.convexity_status is not ConvexityStatus.CONFIRMED:
            raise ValueError("unverified quadratic problem cannot be inspected/routed as convex_qp")
        linear = problem.linear
        problem_class = "convex_qp"
        quadratic_nnz: int | None = int(problem.P.nnz)
        n = problem.n_variables
        quadratic_density: float | None = (
            float(problem.P.nnz / (n * n)) if n > 0 else 0.0
        )
        convexity_status: str | None = ConvexityStatus(problem.convexity_status).value
        extra_bytes = int(problem.P.data.nbytes + problem.P.indices.nbytes + problem.P.indptr.nbytes)
    elif isinstance(problem, LinearProblem):
        linear = problem
        problem_class = "milp" if problem.has_integer_variables else "lp"
        quadratic_nnz = None
        quadratic_density = None
        convexity_status = None
        extra_bytes = 0
    else:
        raise TypeError(f"unsupported problem type: {type(problem)!r}")

    m, n = linear.A.shape
    size = m * n
    density = float(linear.A.nnz / size) if size else 0.0

    domains = linear.domains
    if n:
        binary_fraction = float(np.mean(domains == VariableDomain.BINARY.value))
        integer_fraction = float(
            np.mean(
                np.isin(
                    domains,
                    [VariableDomain.INTEGER.value, VariableDomain.BINARY.value],
                )
            )
        )
        fully_bounded = float(
            np.mean(np.isfinite(linear.variable_lower) & np.isfinite(linear.variable_upper))
        )
        free_fraction = float(
            np.mean(np.isneginf(linear.variable_lower) & np.isposinf(linear.variable_upper))
        )
        objective_nnz_fraction = float(np.count_nonzero(linear.c) / n)
    else:
        binary_fraction = integer_fraction = fully_bounded = free_fraction = 0.0
        objective_nnz_fraction = 0.0

    if m:
        equality_mask = (
            np.isfinite(linear.constraint_lower)
            & np.isfinite(linear.constraint_upper)
            & (linear.constraint_lower == linear.constraint_upper)
        )
        equality_fraction = float(np.mean(equality_mask))
    else:
        equality_fraction = 0.0

    row_counts = np.diff(linear.A.indptr)
    col_counts = np.diff(linear.A.tocsc().indptr)

    data = linear.A.data
    if data.size:
        positive_fraction = float(np.mean(data > 0.0))
        negative_fraction = float(np.mean(data < 0.0))
    else:
        positive_fraction = negative_fraction = 0.0

    finite_rhs = np.concatenate(
        [
            linear.constraint_lower[np.isfinite(linear.constraint_lower)],
            linear.constraint_upper[np.isfinite(linear.constraint_upper)],
            linear.variable_lower[np.isfinite(linear.variable_lower)],
            linear.variable_upper[np.isfinite(linear.variable_upper)],
        ]
    )

    return ProblemFingerprint(
        problem_class=problem_class,
        n_variables=n,
        n_constraints=m,
        nnz_a=int(linear.A.nnz),
        density_a=density,
        binary_fraction=binary_fraction,
        integer_fraction=integer_fraction,
        equality_fraction=equality_fraction,
        fully_bounded_variable_fraction=fully_bounded,
        free_variable_fraction=free_fraction,
        objective_nnz_fraction=objective_nnz_fraction,
        coefficient_dynamic_range_log10=_dynamic_range_log10(linear.A.data),
        objective_dynamic_range_log10=_dynamic_range_log10(linear.c),
        rhs_dynamic_range_log10=_dynamic_range_log10(finite_rhs),
        row_nnz=_distribution(row_counts),
        col_nnz=_distribution(col_counts),
        singleton_rows=int(np.count_nonzero(row_counts == 1)),
        singleton_cols=int(np.count_nonzero(col_counts == 1)),
        positive_coefficient_fraction=positive_fraction,
        negative_coefficient_fraction=negative_fraction,
        quadratic_nnz=quadratic_nnz,
        quadratic_density=quadratic_density,
        convexity_status=convexity_status,
        structural_hash=problem.structural_hash,
        estimated_numeric_bytes=_estimated_bytes_linear(linear) + extra_bytes,
    )
