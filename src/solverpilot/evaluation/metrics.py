from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import mean
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class PortfolioMetrics:
    metric: str
    lower_is_better: bool
    solver_mean_costs: dict[str, float]
    sbs_solver: str
    sbs_cost: float
    vbs_cost: float
    policy_cost: float
    gap_closure: float | None
    policy_wins: int
    policy_ties: int
    instances: int


def _validate_cost_table(costs: Mapping[str, Sequence[float]]) -> int:
    if not costs:
        raise ValueError("costs must contain at least one solver")
    lengths = {len(v) for v in costs.values()}
    if len(lengths) != 1:
        raise ValueError("all solver cost vectors must have the same length")
    n = lengths.pop()
    if n == 0:
        raise ValueError("cost vectors must not be empty")
    for solver, values in costs.items():
        for value in values:
            value = float(value)
            if not isfinite(value) or value < 0:
                raise ValueError(f"cost for {solver!r} must be finite and non-negative")
    return n


def portfolio_metrics(
    costs: Mapping[str, Sequence[float]],
    policy_choices: Sequence[str],
    *,
    metric: str = "wall_time_s",
    tie_relative_tolerance: float = 0.01,
    policy_costs: Sequence[float] | None = None,
) -> PortfolioMetrics:
    """Evaluate a lower-is-better algorithm-selection portfolio.

    Costs must include every cost paid by a deployable policy (e.g. inspection/planning
    overhead when evaluating an automatic policy). SBS is the best single solver by mean
    instance cost. VBS is the per-instance oracle. Gap closure measures how much of the
    SBS-to-VBS opportunity is captured by the policy.
    """

    n = _validate_cost_table(costs)
    if len(policy_choices) != n:
        raise ValueError("policy_choices length must equal the number of instances")
    unknown = sorted(set(policy_choices) - set(costs))
    if unknown:
        raise ValueError(f"policy selected unknown solvers: {unknown}")
    if tie_relative_tolerance < 0:
        raise ValueError("tie_relative_tolerance must be non-negative")

    means = {solver: float(mean(map(float, values))) for solver, values in costs.items()}
    # Stable lexicographic tie-break prevents dict insertion order from becoming semantics.
    sbs_solver = min(means, key=lambda s: (means[s], s))
    sbs_cost = means[sbs_solver]

    vbs_values = [min(float(costs[s][i]) for s in costs) for i in range(n)]
    vbs_cost = float(mean(vbs_values))

    chosen_solver_values = [float(costs[choice][i]) for i, choice in enumerate(policy_choices)]
    if policy_costs is None:
        policy_values = chosen_solver_values
    else:
        if len(policy_costs) != n:
            raise ValueError("policy_costs length must equal the number of instances")
        policy_values = [float(v) for v in policy_costs]
        if any(not isfinite(v) or v < 0 for v in policy_values):
            raise ValueError("policy_costs must be finite and non-negative")
    policy_cost = float(mean(policy_values))

    denominator = sbs_cost - vbs_cost
    if denominator <= max(1e-15, 1e-12 * max(abs(sbs_cost), abs(vbs_cost), 1.0)):
        gap_closure = None
    else:
        gap_closure = (sbs_cost - policy_cost) / denominator

    wins = 0
    ties = 0
    for i, choice in enumerate(policy_choices):
        chosen = chosen_solver_values[i]
        best = vbs_values[i]
        if best == 0.0:
            is_tie = chosen == 0.0
        else:
            is_tie = (chosen - best) / best <= tie_relative_tolerance
        if is_tie:
            ties += 1
        # Strict win means the chosen solver is uniquely faster beyond tolerance.
        competitors = [float(costs[s][i]) for s in costs if s != choice]
        if competitors and all(chosen < c * (1.0 - tie_relative_tolerance) for c in competitors):
            wins += 1

    return PortfolioMetrics(
        metric=metric,
        lower_is_better=True,
        solver_mean_costs=means,
        sbs_solver=sbs_solver,
        sbs_cost=sbs_cost,
        vbs_cost=vbs_cost,
        policy_cost=policy_cost,
        gap_closure=None if gap_closure is None else float(gap_closure),
        policy_wins=wins,
        policy_ties=ties,
        instances=n,
    )


def performance_profile(
    costs: Mapping[str, Sequence[float]],
    taus: Sequence[float],
) -> dict[str, list[float]]:
    """Compute Dolan-Moré style performance profile values for finite costs.

    For each tau >= 1, returns the fraction of instances where a solver's cost is at
    most tau times the best cost on that instance.
    """

    n = _validate_cost_table(costs)
    taus = [float(t) for t in taus]
    if any(t < 1.0 or not isfinite(t) for t in taus):
        raise ValueError("all taus must be finite and >= 1")

    best = [min(float(costs[s][i]) for s in costs) for i in range(n)]
    out: dict[str, list[float]] = {}
    for solver, values in costs.items():
        fractions: list[float] = []
        for tau in taus:
            count = 0
            for value, b in zip(values, best):
                value = float(value)
                ratio = 1.0 if b == 0.0 and value == 0.0 else (float("inf") if b == 0.0 else value / b)
                if ratio <= tau:
                    count += 1
            fractions.append(count / n)
        out[solver] = fractions
    return out
