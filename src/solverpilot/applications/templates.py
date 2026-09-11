"""Small transparent, parameterized application models with named constraints.

These are starting templates with explicit assumptions, not domain-complete
planning products. Inputs remain the caller's responsibility.
"""

import numpy as np
from scipy import sparse
from solverpilot.model import Model


def _finite(value, name, *, nonnegative=False):
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all() or (nonnegative and np.any(value < 0)):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return value


def production_model(profit, resources, capacity, *, maximum=None):
    """Continuous production quantities maximizing profit under resource caps."""
    profit = _finite(profit, "profit")
    resources = sparse.csr_matrix(resources, dtype=float)
    capacity = _finite(capacity, "capacity", nonnegative=True)
    if profit.ndim != 1 or capacity.ndim != 1 or resources.shape != (capacity.size, profit.size):
        raise ValueError("resources must have shape (resources, products)")
    if not np.isfinite(resources.data).all() or np.any(resources.data < 0):
        raise ValueError("resource consumption must be finite and nonnegative")
    model = Model("production")
    x = model.variable(
        profit.size, lower=0, upper=np.inf if maximum is None else maximum, name="production"
    )
    gains = model.parameter(profit.shape, value=profit, name="profit")
    limits = model.parameter(capacity.shape, value=capacity, sign="nonnegative", name="capacity")
    model.add(model.constant(resources) @ x <= limits, name="resource_capacity")
    model.maximize(gains @ x)
    return model


def transportation_model(costs, supply, demand):
    """Continuous shipments; supply is an upper cap, demand a required minimum."""
    costs = _finite(costs, "costs", nonnegative=True)
    supply = _finite(supply, "supply", nonnegative=True)
    demand = _finite(demand, "demand", nonnegative=True)
    if supply.ndim != 1 or demand.ndim != 1 or costs.shape != (supply.size, demand.size):
        raise ValueError("costs must have shape (sources, destinations)")
    model = Model("transportation")
    flow = model.variable(costs.shape, lower=0, name="shipment")
    available = model.parameter(supply.shape, value=supply, sign="nonnegative", name="supply")
    required = model.parameter(demand.shape, value=demand, sign="nonnegative", name="demand")
    prices = model.parameter(costs.shape, value=costs, sign="nonnegative", name="cost")
    model.add(flow.sum(axis=1) <= available, name="supply_capacity")
    model.add(flow.sum(axis=0) >= required, name="demand_requirement")
    model.minimize((prices * flow).sum())
    return model


def energy_dispatch_model(demand, unit_cost, capacity):
    """Single-node dispatch per period; no network, ramping or commitment costs."""
    demand = _finite(demand, "demand", nonnegative=True)
    costs = _finite(unit_cost, "unit_cost", nonnegative=True)
    capacity = _finite(capacity, "capacity", nonnegative=True)
    if demand.ndim != 1 or costs.ndim != 1 or capacity.shape != costs.shape:
        raise ValueError("demand and unit costs must be vectors; capacity matches units")
    model = Model("energy_dispatch")
    generation = model.variable(
        (demand.size, costs.size), lower=0, upper=capacity, name="generation"
    )
    load = model.parameter(demand.shape, value=demand, sign="nonnegative", name="demand")
    price = model.parameter(costs.shape, value=costs, sign="nonnegative", name="unit_cost")
    model.add(generation.sum(axis=1) == load, name="power_balance")
    model.minimize((generation * price).sum())
    return model


def mpc_model(
    A,
    B,
    Q,
    R,
    horizon,
    initial_state,
    *,
    target=None,
    state_lower=-np.inf,
    state_upper=np.inf,
    control_lower=-np.inf,
    control_upper=np.inf,
):
    """Finite-horizon linear control with quadratic state/control costs.

    The state cost includes the terminal state. Dynamics and weights are fixed;
    initial_state and target are named parameters for receding-horizon updates.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    A, B, Q, R = [_finite(v, k) for v, k in zip((A, B, Q, R), ("A", "B", "Q", "R"))]
    initial = _finite(initial_state, "initial_state")
    if (
        initial.ndim != 1
        or B.ndim != 2
        or A.shape != (initial.size, initial.size)
        or B.shape[0] != initial.size
    ):
        raise ValueError("dynamics dimensions must match state and control vectors")
    n, controls = B.shape
    if Q.shape != (n, n) or R.shape != (controls, controls):
        raise ValueError("weight dimensions do not match")
    for matrix in (Q, R):
        if not np.array_equal(matrix, matrix.T) or np.linalg.eigvalsh(matrix).min(initial=0.0) < 0:
            raise ValueError("cost weights must be symmetric positive semidefinite")
    reference = np.zeros(n) if target is None else _finite(target, "target")
    if reference.shape != (n,):
        raise ValueError("target must match initial_state")
    model = Model("linear_mpc")
    x = model.variable((horizon + 1, n), lower=state_lower, upper=state_upper, name="state")
    u = model.variable(
        (horizon, controls), lower=control_lower, upper=control_upper, name="control"
    )
    initial_parameter = model.parameter(n, value=initial, name="initial_state")
    reference_parameter = model.parameter(n, value=reference, name="target")
    model.add(x[0] == initial_parameter, name="initial_condition")
    objective = model.constant(0.0)
    for t in range(horizon + 1):
        error = x[t] - reference_parameter
        objective = objective + (error @ model.constant(Q)) @ error
        if t < horizon:
            model.add(
                x[t + 1] == model.constant(A) @ x[t] + model.constant(B) @ u[t],
                name=f"dynamics[{t}]",
            )
            objective = objective + (u[t] @ model.constant(R)) @ u[t]
    model.minimize(objective)
    return model
