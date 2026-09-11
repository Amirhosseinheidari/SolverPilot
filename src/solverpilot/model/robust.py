"""Explicit robust counterparts for independent coefficient boxes/scenarios."""

import numpy as np
from .atoms import abs


def robust_leq(model, coefficients, radius, variables, bound, *, name=None):
    """Add (a+u).x <= bound for every |u_i| <= radius_i.

    The exact box counterpart is a.x + radius.|x| <= bound.
    This represents coefficient uncertainty, not uncertain decision variables.
    """
    a, r = np.asarray(coefficients, dtype=float), np.asarray(radius, dtype=float)
    if (
        variables._model is not model
        or variables.ndim != 1
        or a.shape != variables.shape
        or r.shape != a.shape
    ):
        raise ValueError("coefficient/radius vectors must match variables from this model")
    if not np.isfinite(a).all() or not np.isfinite(r).all() or np.any(r < 0):
        raise ValueError("coefficients must be finite and box radii nonnegative")
    return model.add(
        model.constant(a) @ variables + model.constant(r) @ abs(variables) <= bound, name=name
    )


def minimize_worst_case(model, variables, scenario_costs, *, offsets=None):
    """Set an explicit minimax linear objective over finitely many scenarios."""
    costs = np.asarray(scenario_costs, dtype=float)
    if (
        variables._model is not model
        or variables.ndim != 1
        or costs.ndim != 2
        or costs.shape[1] != variables.size
        or costs.shape[0] == 0
    ):
        raise ValueError("scenario costs must be a nonempty matrix matching this model vector")
    shifts = np.zeros(costs.shape[0]) if offsets is None else np.asarray(offsets, dtype=float)
    if (
        shifts.shape != (costs.shape[0],)
        or not np.isfinite(costs).all()
        or not np.isfinite(shifts).all()
    ):
        raise ValueError("scenario costs and offsets must be finite with matching shapes")
    worst = model.variable(name="worst_case_cost")
    model.add(
        model.constant(costs) @ variables + model.constant(shifts) <= worst,
        name="scenario_cost_bounds",
    )
    model.minimize(worst)
    return worst
