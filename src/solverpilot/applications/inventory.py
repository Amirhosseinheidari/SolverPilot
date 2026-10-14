"""A small advance-commitment inventory fixture with an independent oracle.

This is a runnable teaching/qualification example, not an industrial data set
or an adaptive multistage policy. Holding includes terminal inventory; shortages
are lost demand. All production releases are chosen before observing scenarios.
"""
from dataclasses import dataclass
from itertools import product
from fractions import Fraction

import numpy as np

from solverpilot.model import Model, piecewise_linear_epigraph
from solverpilot.model.robust import minimize_cvar


@dataclass(frozen=True, slots=True)
class InventoryExample:
    model: Model
    batches: object
    setups: object
    inventory: object
    shortage: object
    cost_curves: tuple
    losses: tuple
    cvar: object
    production_values: tuple


def _cost_values(production_values):
    raw = np.asarray(production_values)
    if raw.shape != (3,) or raw.dtype.kind not in 'iuf':
        raise ValueError("production values must be three numeric constants")
    values = tuple(float(v) for v in raw)
    if not np.isfinite(values).all() or values[0] != 0. or not 0. <= values[1] <= values[2]:
        raise ValueError("production values must be finite nondecreasing costs starting at zero")
    if Fraction.from_float(values[2]) < 2*Fraction.from_float(values[1]) or values[2] > 8.:
        raise ValueError("this fixture needs a convex curve with costs at most 8")
    return values


def build_inventory_example(*, production_values=(0., 2., 5.)):
    """Two releases, batch size 2, capacity 2 batches, lead time one period."""
    values = _cost_values(production_values)
    demand = np.array([[2., 2.], [2., 4.]])
    model = Model("advance_commitment_inventory")
    batches = model.integer(2, lower=0., upper=2., name="release_batches")
    setups = model.binary(2, name="release_setup")
    quantity = 2. * batches
    model.add(batches <= 2. * setups, name="setup_capacity")
    model.add(batches >= setups, name="setup_only_for_positive_release")
    inventory = model.variable((2, 2), lower=0., upper=8., name="ending_inventory")
    shortage = model.variable((2, 2), lower=0., upper=demand, name="lost_demand")
    curves = tuple(piecewise_linear_epigraph(model, quantity[t], [0., 2., 4.], values,
                                            name=f"production_cost[{t}]") for t in range(2))
    # Release at t=0 arrives in demand period 1; release at t=1 arrives in
    # demand period 2. There is no receipt in the release period itself.
    for scenario in range(2):
        for t in range(2):
            previous = 0. if t == 0 else inventory[scenario, t - 1]
            model.add(inventory[scenario, t] - shortage[scenario, t]
                      == previous + quantity[t] - demand[scenario, t],
                      name=f"stock_balance[scenario={scenario},period={t + 1}]")
    base = 3. * setups.sum() + curves[0].output + curves[1].output
    losses = tuple(base + .5 * inventory[s].sum() + 6. * shortage[s].sum() for s in range(2))
    cvar = minimize_cvar(model, model._concat(*losses), np.eye(2),
                         probabilities=[.5, .5], confidence=.5)
    return InventoryExample(model, batches, setups, inventory, shortage, curves, losses, cvar, values)


def enumerate_release_reference(*, production_values=(0., 2., 5.)):
    """Nine plans; independent stock-flow arithmetic, no modeling/solver calls.

    Positive shortage and holding costs make maximal available service optimal
    for these fixed release plans. With two equiprobable scenarios and alpha=.5,
    upper-tail CVaR equals the larger scenario loss.
    """
    values = _cost_values(production_values)
    costs = {0: values[0], 2: values[1], 4: values[2]}
    rows = []
    for first, second in product((0, 2, 4), repeat=2):
        quantities = (first, second)
        base = costs[first] + costs[second] + 3 * sum(q > 0 for q in quantities)
        losses, flows = [], []
        for demands in [(2, 2), (2, 4)]:
            stock = 0
            loss = base
            flow = []
            for receipt, demand in zip(quantities, demands):
                available = stock + receipt
                served = min(available, demand)
                lost = demand - served
                stock = available - served
                loss += .5 * stock + 6 * lost
                flow.append({"receipt": receipt, "demand": demand, "served": served,
                             "lost": lost, "ending_stock": stock})
            losses.append(loss)
            flows.append(flow)
        rows.append({"quantities": quantities, "losses": tuple(losses),
                     "cvar": max(losses), "expected_loss": sum(losses) / 2., "flows": flows})
    best = min(row['cvar'] for row in rows)
    plans = tuple(row['quantities'] for row in rows if row['cvar'] == best)
    return {"plans": rows, "optimal_cvar": best, "optimal_quantities": plans,
            "unique_release_plan": len(plans) == 1, "scope": "finite_fixture_release_enumeration_with_analytic_recourse"}


def inventory_reference_agrees(reference, quantities, losses, *, atol=1e-6):
    """Check both prioritized objectives against the independent finite oracle."""
    quantities, losses = np.asarray(quantities, dtype=float), np.asarray(losses, dtype=float)
    if quantities.shape != (2,) or losses.shape != (2,) or not np.isfinite(quantities).all() or not np.isfinite(losses).all():
        return False
    optimal_rows = [row for row in reference['plans'] if row['cvar'] == reference['optimal_cvar']]
    expected = min(row['expected_loss'] for row in optimal_rows)
    matches = any(np.allclose(quantities, row['quantities'], atol=atol, rtol=0.) for row in optimal_rows)
    return bool(matches and abs(float(losses.max()) - reference['optimal_cvar']) <= atol
                and abs(float(losses.mean()) - expected) <= atol)


def solve_inventory_example(*, production_values=(0., 2., 5.), budget=None):
    """Minimize CVaR then expected loss to resolve nonunique low-loss recourse."""
    from solverpilot.model.compiler import _Evaluator
    from solverpilot.model.atoms import evaluate
    from solverpilot.runtime.multiobjective import solve_lexicographic
    example = build_inventory_example(production_values=production_values)
    model = example.model
    compiled = model.compile()
    problem = compiled.execution_ir
    offsets = {key: tuple(value['execution_slice']) for key, value in compiled.source_map['variables'].items()}
    expected = .5 * (example.losses[0] + example.losses[1])
    polynomial = _Evaluator(model, offsets).evaluate(expected._node).item()
    secondary = np.zeros(problem.n_variables)
    for i, value in polynomial.linear.items():
        secondary[i] = value
    result = solve_lexicographic(problem, [problem.c, secondary], tolerance=1e-7, budget=budget)
    if not result.completed or result.x is None:
        return {'completed': False, 'termination_reason': result.termination_reason}
    original = compiled.reconstruct_primal(result.x)
    validation = compiled.validate_original(model, result.x)
    pwl = [curve.validate_original(compiled, result.x) for curve in example.cost_curves]
    quantities = 2. * np.asarray(evaluate(model, example.batches._node, original))
    losses = tuple(float(evaluate(model, loss._node, original)) for loss in example.losses)
    reference = enumerate_release_reference(production_values=example.production_values)
    verified = bool(validation.valid and all(check.valid for check in pwl)
                    and inventory_reference_agrees(reference, quantities, losses))
    return {
        'schema': 'solverpilot.inventory-fixture.v1', 'completed': True,
        'original_primal_valid': validation.valid, 'pwl_source_valid': [check.valid for check in pwl],
        'reference_agrees': verified, 'release_quantities': quantities.tolist(),
        'release_times': [0, 1], 'receipt_periods': [1, 2], 'lead_time_periods': 1,
        'scenario_demands': [[2, 2], [2, 4]], 'scenario_probabilities': [.5, .5],
        'scenario_losses': list(losses), 'confidence': .5, 'cvar': max(losses),
        'expected_loss': sum(losses) / 2.,
        'ending_inventory': np.asarray(evaluate(model, example.inventory._node, original)).tolist(),
        'lost_demand': np.asarray(evaluate(model, example.shortage._node, original)).tolist(),
        'reference': reference, 'source_semantic_hash': compiled.semantic_hash,
        'source_data_hash': compiled.data_hash,
        'scope': 'synthetic advance commitment fixture; numerical lexicographic locks; no participant or industrial qualification',
    }
