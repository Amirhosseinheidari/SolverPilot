"""Quality, named dual marginals and local QP differentiation."""

from dataclasses import dataclass
import numpy as np
from scipy import sparse
from solverpilot._immutability import readonly_array
from solverpilot.problem import LinearProblem, QuadraticProblem, ObjectiveSense
from solverpilot.runtime.unified import summarize
from solverpilot.runtime.manifest import json_value


def scaling_report(problem):
    linear = problem.linear if isinstance(problem, QuadraticProblem) else problem
    if not isinstance(linear, LinearProblem):
        raise TypeError("scaling diagnostics currently support canonical LP/QP")
    values = [np.abs(linear.A.data), np.abs(linear.c)]
    if isinstance(problem, QuadraticProblem):
        values.append(np.abs(problem.P.data))
    nonzero = np.concatenate(values)
    nonzero = nonzero[nonzero > 0]
    smallest = float(nonzero.min()) if nonzero.size else 0.0
    largest = float(nonzero.max()) if nonzero.size else 0.0
    with np.errstate(over="ignore"):
        ratio = largest / smallest if smallest else 1.0
    bounds = np.r_[linear.variable_lower, linear.variable_upper]
    finite = np.abs(bounds[np.isfinite(bounds)])
    max_bound = float(finite.max(initial=0.0))
    warnings = []
    if ratio > 1e8:
        warnings.append("coefficient magnitudes span more than eight orders of magnitude")
    if 0 < smallest < 1e-8:
        warnings.append("small coefficients may fall below native absolute tolerances")
    if max_bound > 1e8:
        warnings.append("large variable bounds can amplify stationarity error")
    return {
        "coefficient_min_nonzero": smallest,
        "coefficient_max": largest,
        "coefficient_ratio": ratio,
        "largest_finite_variable_bound": max_bound,
        "warnings": tuple(warnings),
        "automatic_scaling_applied": False,
    }


def quality_report(problem, result):
    """JSON-ready quality evidence; solver claims stay separate from bounds."""
    summary = summarize(result)
    if summary.problem_data_hash != getattr(problem, "data_hash", None):
        raise ValueError("result belongs to another problem")
    raw = getattr(result, "raw_statistics", None) or {}
    payload = {
        "schema": "solverpilot.quality.v1",
        "summary": summary,
        "validation": getattr(result, "validation", None),
        "optimality_check": raw.get("optimality_check"),
        "interpretation": "solver_reported is a backend claim; independent_numerical_bound is tolerance-qualified",
    }
    if isinstance(problem, (LinearProblem, QuadraticProblem)):
        payload["scaling"] = scaling_report(problem)
    return json_value(payload)


@dataclass(frozen=True, slots=True)
class ShadowPrice:
    kind: str
    index: int
    name: str
    bound: str
    marginal_value: float
    interpretation: str = "dual marginal; differentiability and uniqueness are not guaranteed"


def shadow_prices(model, compiled, result):
    """Map canonical bound marginals to names, in execution row orientation.

    Changing a named parameter can also change row signs/constants. Use
    parameter_sensitivity for that chain rule rather than treating a canonical
    bound marginal as the derivative of the user parameter.
    """
    from solverpilot.model import named_values

    named_values(model, compiled, result)
    problem = compiled.execution_ir
    linear = problem.linear if isinstance(problem, QuadraticProblem) else problem
    if not isinstance(linear, LinearProblem) or linear.has_integer_variables:
        raise TypeError("shadow prices require continuous LP/QP without conic transformations")
    if not result.optimality_evidence.independently_verified_optimal:
        raise ValueError("a verified numerical optimality bound is required")
    y = np.asarray(result.raw_statistics["canonical_dual"], dtype=float)
    sign = 1.0 if linear.objective_sense is ObjectiveSense.MINIMIZE else -1.0
    names = {}
    for c in model.constraints:
        for row in (
            compiled.source_map["constraints"].get(c.entity_id.value, {}).get("execution_rows", [])
        ):
            names[row] = c.name or c.entity_id.value
    for v in model.variables:
        start, stop = compiled.source_map["variables"][v.id.value]["execution_slice"]
        for i in range(start, stop):
            names[linear.n_constraints + i] = v.name or v.id.value
    lower = np.r_[linear.constraint_lower, linear.variable_lower]
    upper = np.r_[linear.constraint_upper, linear.variable_upper]
    return tuple(
        ShadowPrice(
            "constraint" if i < linear.n_constraints else "variable",
            i if i < linear.n_constraints else i - linear.n_constraints,
            names.get(i, str(i)),
            "equality"
            if lower[i] == upper[i]
            else "upper"
            if value > 0
            else "lower"
            if value < 0
            else "inactive",
            float(-sign * value),
        )
        for i, value in enumerate(y)
    )


def parameter_sensitivity(model, compiled, result, parameter_name, *, step=1e-5):
    """Estimate model-data derivatives, then apply the dual envelope rule.

    This is a local dual marginal, not a guarantee of differentiability. Central
    differences are taken through compilation, preserving signs and offsets.
    The original model is never mutated. Domain-boundary directions can fail.
    """
    shadow_prices(model, compiled, result)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be finite and positive")
    clone = model.clone()
    parameters = [p for p in clone.parameters if p.name == parameter_name]
    if len(parameters) != 1:
        raise ValueError("parameter name must identify exactly one parameter")
    parameter = parameters[0]
    baseline = parameter.value
    p = compiled.execution_ir
    linear = p.linear if isinstance(p, QuadraticProblem) else p
    sign = 1.0 if linear.objective_sense is ObjectiveSense.MINIMIZE else -1.0
    y = np.asarray(result.raw_statistics["canonical_dual"])
    x = result.x
    active = y != 0

    def lagrangian(problem):
        data = problem.linear if isinstance(problem, QuadraticProblem) else problem
        if data.A.shape != linear.A.shape or problem.n_variables != p.n_variables:
            raise ValueError("parameter perturbation changes model structure")
        bound = np.where(
            y > 0,
            np.r_[data.constraint_upper, data.variable_upper],
            np.r_[data.constraint_lower, data.variable_lower],
        )
        if not np.isfinite(bound[active]).all():
            raise ValueError("active bound changes domain")
        values = np.r_[data.A @ x, x]
        value = data.c @ x + data.objective_offset + sign * y[active] @ (values - bound)[active]
        if isinstance(problem, QuadraticProblem):
            value += 0.5 * x @ problem.P @ x
        return value

    gradient = np.empty(baseline.shape)
    for index in np.ndindex(baseline.shape):
        h = step * max(1.0, abs(float(baseline[index])))
        plus, minus = baseline.copy(), baseline.copy()
        plus[index] += h
        minus[index] -= h
        parameter.value = plus
        forward = lagrangian(clone.compile().execution_ir)
        parameter.value = minus
        backward = lagrangian(clone.compile().execution_ir)
        gradient[index] = (forward - backward) / (2 * h)
        parameter.value = baseline
    return {
        "parameter": parameter_name,
        "marginal": readonly_array(gradient),
        "method": "central model-data differences with dual envelope",
        "differentiability_guaranteed": False,
        "relative_step": step,
    }


@dataclass(frozen=True, slots=True)
class QPDerivative:
    dx: np.ndarray
    objective_derivative: float
    active_rows: tuple[int, ...]
    condition_number: float
    scope: str = "local derivative under unchanged active set and strict complementarity"

    def __post_init__(self):
        object.__setattr__(self, "dx", readonly_array(self.dx))


def differentiate_qp(
    problem,
    result,
    *,
    dq=None,
    dP=None,
    dA=None,
    dl=None,
    du=None,
    active_tolerance=1e-6,
    max_dimension=512,
):
    """Differentiate one direction of a regular convex QP KKT system.

    dl/du follow canonical ordering: linear rows, then variable bounds.
    Rejects degeneracy, dependent active rows, ill-conditioning and unsupported
    sizes. It never supplies a heuristic derivative at a kink or nonunique point.
    """
    if not isinstance(problem, QuadraticProblem):
        raise TypeError("a convex QP is required")
    if not np.isfinite(active_tolerance) or active_tolerance <= 0:
        raise ValueError("active_tolerance must be finite and positive")
    if (
        result.trace.problem_data_hash != problem.data_hash
        or not result.optimality_evidence.independently_verified_optimal
    ):
        raise ValueError("a matching independently checked numerical optimum is required")
    n = problem.n_variables
    linear = problem.linear
    B = sparse.vstack([linear.A, sparse.eye(n)], format="csr")
    m = B.shape[0]

    def data(value, shape):
        value = (
            np.zeros(shape)
            if value is None
            else (value.toarray() if sparse.issparse(value) else np.asarray(value, dtype=float))
        )
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"direction must be finite with shape {shape}")
        return value

    if n > max_dimension:
        raise ValueError("QP derivative exceeds configured dense dimension limit")
    dq, dP, dA = data(dq, (n,)), data(dP, (n, n)), data(dA, (linear.n_constraints, n))
    dl, du = data(dl, (m,)), data(du, (m,))
    if not np.allclose(dP, dP.T, atol=0.0, rtol=0.0):
        raise ValueError("dP must be symmetric")
    lower, upper = (
        np.r_[linear.constraint_lower, linear.variable_lower],
        np.r_[linear.constraint_upper, linear.variable_upper],
    )
    x = result.x
    y = np.asarray(result.raw_statistics["canonical_dual"])
    bx = B @ x
    equality = np.isfinite(lower) & (lower == upper)
    low_active = np.isfinite(lower) & (np.abs(bx - lower) <= active_tolerance)
    high_active = np.isfinite(upper) & (np.abs(bx - upper) <= active_tolerance)
    active = low_active | high_active
    if np.any(active & ~equality & (np.abs(y) <= active_tolerance)):
        raise ValueError("degenerate active bound; no differentiability guarantee")
    if np.any(equality & (dl != du)):
        raise ValueError("direction changes equality into an interval")
    C = B[active].toarray()
    if n + len(C) > max_dimension:
        raise ValueError("KKT system exceeds configured dense dimension limit")
    if len(C) and np.linalg.matrix_rank(C) != len(C):
        raise ValueError("dependent active constraints; derivative is not uniquely determined")
    dB = np.vstack([dA, np.zeros((n, n))])
    db = np.where(high_active & ~equality, du, dl)
    K = np.block([[problem.P.toarray(), C.T], [C, np.zeros((len(C), len(C)))]])
    condition = float(np.linalg.cond(K))
    if not np.isfinite(condition) or condition > 1e12:
        raise ValueError("singular or ill-conditioned KKT system")
    rhs = np.r_[-dq - dP @ x - dB.T @ y, (db - dB @ x)[active]]
    dx = np.linalg.solve(K, rhs)[:n]
    derivative = float(0.5 * x @ dP @ x + dq @ x + (problem.P @ x + linear.c) @ dx)
    return QPDerivative(dx, derivative, tuple(np.flatnonzero(active).tolist()), condition)
