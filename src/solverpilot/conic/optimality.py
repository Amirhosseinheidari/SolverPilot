"""Independent original-model conic lower bounds with exact dual membership.

SOC/RSOC membership uses rational arithmetic. PSD membership uses the conservative
PSD evidence engine. Transcendental cone duals return unsupported, not verified.
Primal feasibility and the final optimality gap remain tolerance-qualified.
"""

from dataclasses import dataclass
from fractions import Fraction as F
import math
import numpy as np
from scipy import sparse
from solverpilot.problem.curvature import certify_psd
from solverpilot.validate import ValidationTolerances
from solverpilot.validate._certificate_arithmetic import rational, matvec, dot, box_min
from .ir import ConeKind
from .validation import validate_conic_solution


@dataclass(frozen=True, slots=True)
class ConicOptimalityCheck:
    verified: bool
    primal_valid: bool
    dual_valid: bool
    dual_bound: float | None
    gap: float | None
    stationarity_inf: float | None
    reason: str


def _dual_member(kind, value):
    if not np.isfinite(value).all():
        return False
    a = np.asarray(value)
    if kind is ConeKind.SECOND_ORDER:
        z = [rational(v) for v in a.reshape(-1)]
        return z[0] >= 0 and z[0] * z[0] >= sum((v * v for v in z[1:]), F())
    if kind is ConeKind.ROTATED_SECOND_ORDER:
        z = [rational(v) for v in a.reshape(-1)]
        return z[0] >= 0 and z[1] >= 0 and 2 * z[0] * z[1] >= sum((v * v for v in z[2:]), F())
    if kind is ConeKind.POSITIVE_SEMIDEFINITE:
        return certify_psd(a).certified
    return False


def repair_dual(kind, value):
    """Try a small inward perturbation; success is still independently checked."""
    a = np.asarray(value, dtype=float).copy()
    if _dual_member(kind, a) or not np.isfinite(a).all():
        return a
    if kind is ConeKind.SECOND_ORDER:
        a[0] = max(0.0, a[0], math.hypot(*a[1:]))
        for _ in range(4):
            a[0] = np.nextafter(a[0], np.inf)
            if _dual_member(kind, a):
                break
    elif kind is ConeKind.ROTATED_SECOND_ORDER:
        delta = np.finfo(float).eps * max(1.0, float(np.max(np.abs(a)))) * 16
        for _ in range(4):
            a[:2] = np.maximum(a[:2], 0.0) + delta
            if _dual_member(kind, a):
                break
            delta *= 16
    elif kind is ConeKind.POSITIVE_SEMIDEFINITE and a.shape[0] <= 32:
        a = 0.5 * a + 0.5 * a.T
        eig = float(np.linalg.eigvalsh(a)[0])
        shift = (
            max(0.0, -eig)
            + np.finfo(float).eps * max(1.0, float(np.linalg.norm(a, ord=np.inf))) * 16
        )
        for _ in range(4):
            a = a + np.eye(len(a)) * shift
            if _dual_member(kind, a):
                break
            shift *= 16
    return a


def _outward(value, lower):
    try:
        out = float(value)
    except OverflowError:
        return -np.inf if value < 0 else np.inf
    if (lower and F(out) > value) or (not lower and F(out) < value):
        out = float(np.nextafter(out, -np.inf if lower else np.inf))
    return out


def verify_conic_optimality(problem, x, linear_dual, cone_duals, *, tolerances=None):
    tol = tolerances or ValidationTolerances()
    bad = lambda reason: ConicOptimalityCheck(False, False, False, None, None, None, reason)
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(linear_dual, dtype=float)
        zs = tuple(np.asarray(v, dtype=float) for v in cone_duals)
    except (TypeError, ValueError):
        return bad("non-numeric witness")
    if (
        x.shape != (problem.n_variables,)
        or y.shape != (problem.n_linear_constraints + problem.n_variables,)
        or len(zs) != len(problem.cones)
    ):
        return bad("witness dimension mismatch")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return bad("nonfinite witness")
    if not certify_psd(problem.P).certified:
        return bad("objective PSD evidence unavailable")
    for block, z in zip(problem.cones, zs):
        if z.shape != block.output_shape:
            return bad("cone dual shape mismatch")
        if block.kind in {ConeKind.EXPONENTIAL, ConeKind.POWER, ConeKind.GENERALIZED_POWER}:
            return bad("independent transcendental dual membership is not implemented")
        if not _dual_member(block.kind, z):
            return bad("dual cone membership not established")
    B = sparse.vstack([problem.A, sparse.eye(problem.n_variables)], format="csr")
    lo = np.r_[problem.constraint_lower, problem.variable_lower]
    hi = np.r_[problem.constraint_upper, problem.variable_upper]
    bound = np.where(y > 0, hi, lo)
    active = y != 0
    if not np.isfinite(bound[active]).all():
        return bad("dual multiplier references an infinite bound")
    px = matvec(problem.P, x)
    bty = matvec(B.T, y)
    residual = [v + rational(q) + w for v, q, w in zip(px, problem.q, bty)]
    cone_constant = F()
    for block, z in zip(problem.cones, zs):
        flat = z.reshape(-1)
        fz = matvec(block.F.T, flat)
        residual = [a - b for a, b in zip(residual, fz)]
        cone_constant += dot(block.g, flat)
    correction = box_min(residual, problem.variable_lower, problem.variable_upper)
    if correction is None:
        return bad("stationarity residual has no finite original-domain lower bound")
    half = sum((rational(a) * b for a, b in zip(x, px)), F()) / 2
    dual = (
        -half
        - dot(y[active], bound[active])
        - cone_constant
        + correction
        + rational(problem.objective_offset)
    )
    primal = half + dot(problem.q, x) + rational(problem.objective_offset)
    gap = abs(primal - dual)
    allowed = rational(tol.objective_abs) + rational(tol.objective_rel) * max(
        F(1), abs(primal), abs(dual)
    )
    checked = validate_conic_solution(problem, x, atol=tol.feasibility, rtol=tol.feasibility_rel)
    bound_value = _outward(dual, True)
    gap_value = _outward(gap, False)
    finite = np.isfinite(bound_value) and np.isfinite(gap_value)
    return ConicOptimalityCheck(
        bool(checked.valid and finite and gap <= allowed),
        checked.valid,
        True,
        bound_value,
        gap_value,
        float(max((abs(v) for v in residual), default=F())),
        "original-domain lower bound with independently checked dual cones",
    )
