"""Bounded auxiliary-LP witness recovery, independently checked on original data."""

from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy import sparse
from solverpilot.problem import LinearProblem, ObjectiveSense
from solverpilot._immutability import readonly_array
from .result import ValidationTolerances
from .optimality import verify_infeasibility, verify_unboundedness


@dataclass(frozen=True, slots=True)
class LPCertificate:
    problem_data_hash: str
    kind: str
    verified: bool
    reason: str
    dual: np.ndarray | None = None
    point: np.ndarray | None = None
    direction: np.ndarray | None = None
    elapsed_s: float = 0.0

    def __post_init__(self):
        for name in ("dual", "point", "direction"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, readonly_array(getattr(self, name), dtype=float))


def verify_lp_certificate(problem, certificate, *, tolerances=None):
    """Recheck actual witnesses; a serialized verified flag is not trusted."""
    if (
        not isinstance(certificate, LPCertificate)
        or not isinstance(problem, LinearProblem)
        or problem.has_integer_variables
        or certificate.problem_data_hash != problem.data_hash
    ):
        return False
    if certificate.kind == "infeasible":
        return certificate.dual is not None and verify_infeasibility(problem, certificate.dual)
    if certificate.kind == "unbounded":
        return (
            certificate.point is not None
            and certificate.direction is not None
            and verify_unboundedness(
                problem, certificate.point, certificate.direction, tolerances=tolerances
            )
        )
    return False


def recover_lp_certificate(
    problem, *, termination="infeasible_or_unbounded", time_limit_s=5.0, tolerances=None
):
    if not isinstance(problem, LinearProblem) or problem.has_integer_variables:
        raise TypeError("certificate recovery supports continuous LinearProblem only")
    if isinstance(time_limit_s, bool) or not np.isfinite(time_limit_s) or time_limit_s <= 0:
        raise ValueError("certificate recovery time_limit_s must be finite and positive")
    if termination not in {"infeasible", "unbounded", "infeasible_or_unbounded"}:
        raise ValueError("recovery requires an infeasible/unbounded termination")
    start = perf_counter()
    tol = tolerances or ValidationTolerances()
    if not isinstance(tol, ValidationTolerances):
        raise TypeError("tolerances must be ValidationTolerances")

    def result(kind, verified, reason, **kwargs):
        return LPCertificate(
            problem.data_hash, kind, verified, reason, elapsed_s=perf_counter() - start, **kwargs
        )

    if problem.n_constraints + problem.n_variables > 10000 or problem.A.nnz > 1000000:
        return result("unknown", False, "auxiliary recovery size budget exceeded")
    from scipy.optimize import linprog

    B = sparse.vstack([problem.A, sparse.eye(problem.n_variables)], format="csr")
    lo = np.r_[problem.constraint_lower, problem.variable_lower]
    hi = np.r_[problem.constraint_upper, problem.variable_upper]
    upper = np.flatnonzero(np.isfinite(hi))
    lower = np.flatnonzero(np.isfinite(lo))
    G = sparse.vstack([B[upper], -B[lower]], format="csr")
    h = np.r_[hi[upper], -lo[lower]]

    def lp(c, **kwargs):
        remaining = time_limit_s - (perf_counter() - start)
        if remaining <= 0:
            return None
        return linprog(c, method="highs-ds", options={"time_limit": remaining}, **kwargs)

    if termination in {"infeasible", "infeasible_or_unbounded"} and len(h):
        aux = lp(
            np.zeros(len(h)),
            A_eq=sparse.vstack([G.T, sparse.csr_matrix(h.reshape(1, -1))]),
            b_eq=np.r_[np.zeros(problem.n_variables), -1.0],
            bounds=(0, None),
        )
        if aux is not None and aux.x is not None:
            y = np.zeros(B.shape[0])
            np.add.at(y, upper, aux.x[: len(upper)])
            np.add.at(y, lower, -aux.x[len(upper) :])
            if verify_infeasibility(problem, y):
                return result(
                    "infeasible", True, "original-coordinate Farkas witness checked", dual=y
                )
    if termination in {"unbounded", "infeasible_or_unbounded"} and problem.n_variables:
        origin = lp(np.zeros(problem.n_variables), A_ub=G, b_ub=h, bounds=(None, None))
        if origin is not None and origin.x is not None:
            sign = 1.0 if problem.objective_sense is ObjectiveSense.MINIMIZE else -1.0
            ray = lp(sign * problem.c, A_ub=G, b_ub=np.zeros(len(h)), bounds=(-1, 1))
            if (
                ray is not None
                and ray.x is not None
                and verify_unboundedness(problem, origin.x, ray.x, tolerances=tol)
            ):
                return result(
                    "unbounded",
                    True,
                    "original feasible origin and recession direction checked",
                    point=origin.x,
                    direction=ray.x,
                )
    return result("unknown", False, "no independently verified witness recovered within budget")
