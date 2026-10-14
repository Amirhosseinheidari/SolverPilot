"""Explicit robust counterparts for independent coefficient boxes/scenarios."""

import numpy as np
import builtins
from dataclasses import dataclass
from .atoms import abs, norm
from solverpilot._immutability import readonly_array


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


@dataclass(frozen=True, slots=True)
class CVaRModel:
    threshold: object
    excess: object
    objective: object
    probabilities: np.ndarray
    confidence: float

    def __post_init__(self):
        object.__setattr__(self,'probabilities',readonly_array(self.probabilities,dtype=float))


def minimize_cvar(model, variables, scenario_costs, *, probabilities=None, confidence=.95, offsets=None):
    """Finite-scenario upper-tail CVaR: eta + E[(loss-eta)+]/(1-alpha).

    Probabilities must sum to one within 1e-12; they are never silently normalized.
    alpha=0 gives expected loss; alpha=1 is rejected (use worst-case instead).
    """
    costs=np.asarray(scenario_costs,dtype=float)
    if variables._model is not model or variables.ndim!=1 or costs.ndim!=2 or costs.shape[1]!=variables.size or costs.shape[0]==0:
        raise ValueError('scenario costs must be a nonempty finite matrix matching this model vector')
    if isinstance(confidence,(bool,np.bool_)) or not np.isfinite(confidence) or not 0<=confidence<1:
        raise ValueError('confidence must be finite in [0,1)')
    n=costs.shape[0]
    p=np.full(n,1/n) if probabilities is None else np.asarray(probabilities,dtype=float)
    shifts=np.zeros(n) if offsets is None else np.asarray(offsets,dtype=float)
    if p.shape!=(n,) or shifts.shape!=(n,) or not np.isfinite(costs).all() or not np.isfinite(shifts).all() or not np.isfinite(p).all() or np.any(p<0) or builtins.abs(float(p.sum())-1)>1e-12:
        raise ValueError('finite probabilities must be nonnegative, sum to one and match scenarios')
    losses=model.constant(costs)@variables+model.constant(shifts)
    if confidence==0:
        objective=model.constant(p)@losses
        model.minimize(objective)
        return CVaRModel(model.constant(0.),model.constant(np.zeros(n)),objective,p,0.)
    weights=p/(1-float(confidence))
    from fractions import Fraction
    if sum((Fraction(float(v)) for v in weights),Fraction())<=1:
        raise ValueError('confidence is too close to zero for represented probabilities; use confidence=0')
    threshold=model.variable(name='cvar_threshold'); excess=model.variable(n,lower=0,name='cvar_excess')
    model.add(losses-threshold<=excess,name='cvar_scenario_excess')
    objective=threshold+model.constant(weights)@excess
    model.minimize(objective)
    return CVaRModel(threshold,excess,objective,p,float(confidence))


def ellipsoidal_leq(model, coefficients, factor, variables, bound, *, radius=1., name=None):
    """Robust a.x <= b for a in {center + factor@u: ||u||2 <= radius}."""
    a=np.asarray(coefficients,dtype=float); f=np.asarray(factor,dtype=float)
    if variables._model is not model or variables.ndim!=1 or a.shape!=variables.shape or f.ndim!=2 or f.shape[0]!=variables.size:
        raise ValueError('center/factor must match this model vector')
    if isinstance(radius,(bool,np.bool_)) or not np.isfinite(radius) or radius<0 or not np.isfinite(a).all() or not np.isfinite(f).all() or not np.isfinite(bound):
        raise ValueError('ellipsoid data must be finite and radius nonnegative')
    expression=model.constant(a)@variables
    if radius and f.shape[1]: expression=expression+float(radius)*norm(model.constant(f.T)@variables,2)
    return model.add(expression<=bound,name=name)
