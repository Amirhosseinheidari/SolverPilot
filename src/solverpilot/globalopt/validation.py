from dataclasses import replace
import numpy as np
from solverpilot.validate import (
    CandidateSolution,
    ValidationReport,
    ValidationTolerances,
    validate_solution,
)
from solverpilot.validate.core import _scaled_feasibility, _max_violation
from solverpilot.problem import VariableDomain
from .problem import GlobalQuadraticProblem
from .expressions import evaluate_node


def validate_global_solution(problem, x, *, objective_reported=None, tolerances=None):
    tol = tolerances or ValidationTolerances()
    bad = lambda reason: ValidationReport(
        False, np.inf, np.inf, np.inf, None, objective_reported, None, False, (reason,)
    )
    if x is None:
        return bad("no primal candidate")
    x = np.asarray(x, dtype=float)
    if x.shape != (problem.n_variables,) or not np.isfinite(x).all():
        return bad("candidate shape or finite-value check failed")
    if isinstance(problem, GlobalQuadraticProblem):
        base = validate_solution(problem.linear, CandidateSolution(x), tolerances=tol)
        with np.errstate(all="ignore"):
            objective = problem.objective_value(x)
    else:
        p = problem.relaxation
        _, bound_ok = _scaled_feasibility(
            x, p.variable_lower, p.variable_upper, atol=tol.feasibility, rtol=tol.feasibility_rel
        )
        bv = _max_violation(x, p.variable_lower, p.variable_upper)
        mask = np.array([d != VariableDomain.CONTINUOUS.value for d in problem.domains], dtype=bool)
        iv = float(np.max(np.abs(x[mask] - np.rint(x[mask])), initial=0.0))
        rows = list(p.constraints)
        rows.extend(
            i.constraint
            for i in problem.indicators
            if abs(x[i.variable_index] - i.active_value) <= tol.integrality
        )
        values = []
        lowers = []
        uppers = []
        try:
            for c in rows:
                value = np.asarray(evaluate_node(problem, c.node, x), dtype=float).reshape(-1)
                if value.shape != c.lower.shape:
                    return bad("constraint evaluation shape mismatch")
                values.extend(value)
                lowers.extend(c.lower)
                uppers.extend(c.upper)
            activity = np.array(values)
            lo = np.array(lowers)
            hi = np.array(uppers)
            _, row_ok = _scaled_feasibility(
                activity, lo, hi, atol=tol.feasibility, rtol=tol.feasibility_rel
            )
            cv = _max_violation(activity, lo, hi)
            objective = float(evaluate_node(problem, p.objective_node, x))
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            return bad("nonlinear evaluation failed")
        warnings = tuple(
            name
            for ok, name in [
                (bound_ok, "variable bound violation"),
                (row_ok, "constraint violation"),
                (iv <= tol.integrality, "integrality violation"),
            ]
            if not ok
        )
        base = ValidationReport(
            bound_ok and row_ok and iv <= tol.integrality,
            bv,
            cv,
            iv if mask.any() else None,
            objective,
            None,
            None,
            None,
            warnings,
        )
    finite = np.isfinite(objective)
    difference = None if objective_reported is None else abs(objective - objective_reported)
    consistent = (
        None
        if objective_reported is None
        else bool(
            finite
            and np.isfinite(objective_reported)
            and difference
            <= tol.objective_abs + tol.objective_rel * max(abs(objective), abs(objective_reported))
        )
    )
    return replace(
        base,
        valid=bool(base.valid and finite and consistent is not False),
        objective_recomputed=objective,
        objective_reported=objective_reported,
        objective_difference=difference,
        objective_consistent=consistent,
        warnings=base.warnings
        + (
            () if finite and consistent is not False else ("objective mismatch or nonfinite value",)
        ),
    )
