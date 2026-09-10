from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import median
from typing import Mapping

from solverpilot.evaluation import portfolio_metrics

from .summary import (
    _get_cost,
    _paired_bootstrap_policy_ratio,
    _require_comparable_environment,
    _require_experiment_identity,
    _require_instance_identity,
    _require_protocol_identity,
    _row_is_benchmark_success,
    load_rows,
)


def _finite_nonnegative(value: object, *, label: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(out) or out < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return out


def load_policy_map(path: str | Path) -> dict[str, dict]:
    """Load a frozen external selector policy without inventing missing overhead."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("policy map must be a JSON object keyed by instance")
    out: dict[str, dict] = {}
    for instance, raw in payload.items():
        if isinstance(raw, str):
            out[str(instance)] = {
                "backend": raw,
                "overhead_s": None,
                "overhead_measured": False,
                "overhead_components_s": {},
            }
            continue
        if not isinstance(raw, dict) or not isinstance(raw.get("backend"), str):
            raise ValueError(f"policy entry for {instance!r} must provide backend")
        components_raw = raw.get("overhead_components_s", {})
        if components_raw is None:
            components_raw = {}
        if not isinstance(components_raw, Mapping):
            raise ValueError(f"policy overhead_components_s for {instance!r} must be an object")
        components = {
            str(name): _finite_nonnegative(value, label=f"policy overhead component {name!r} for {instance!r}")
            for name, value in components_raw.items()
        }
        explicit_total = "overhead_s" in raw and raw.get("overhead_s") is not None
        if explicit_total:
            overhead = _finite_nonnegative(raw["overhead_s"], label=f"policy overhead for {instance!r}")
            if components:
                component_sum = sum(components.values())
                if not math.isclose(overhead, component_sum, rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError(f"policy overhead for {instance!r} disagrees with overhead_components_s: total={overhead}, components={component_sum}")
            measured = True
        elif components:
            overhead = float(sum(components.values())); measured = True
        else:
            overhead = None; measured = False
        out[str(instance)] = {
            "backend": raw["backend"],
            "overhead_s": overhead,
            "overhead_measured": measured,
            "overhead_components_s": components,
        }
    return out


def evaluate_policy_map(
    rows: list[dict],
    policy: dict[str, dict],
    *,
    cutoff_s: float,
    par_penalty: float = 10.0,
    cost_field: str = "wall_s",
    bootstrap_draws: int = 10000,
    bootstrap_seed: int = 0,
    allow_mixed_environments: bool = False,
    allow_unmeasured_overhead: bool = False,
) -> dict:
    if cutoff_s <= 0 or par_penalty < 1:
        raise ValueError("invalid cutoff/par_penalty")
    environment_ids = _require_comparable_environment(rows, allow_mixed_environments=allow_mixed_environments)
    experiment_ids = _require_experiment_identity(rows)
    protocol_id = _require_protocol_identity(rows)
    _require_instance_identity(rows)
    direct_rows = [r for r in rows if r.get("backend") != "@auto"]
    backends = sorted({str(r["backend"]) for r in direct_rows})
    instances = sorted({str(r["instance"]) for r in direct_rows})
    if set(policy) != set(instances):
        missing = sorted(set(instances) - set(policy)); extra = sorted(set(policy) - set(instances))
        raise ValueError(f"policy map must exactly cover benchmark instances; missing={missing[:10]}, extra={extra[:10]}")

    unmeasured = sorted(instance for instance in instances if not bool(policy[instance].get("overhead_measured", policy[instance].get("overhead_s") is not None)))
    if unmeasured and not allow_unmeasured_overhead:
        raise ValueError(
            "policy decision overhead is unmeasured for instances "
            f"{unmeasured[:10]}; provide overhead_s/overhead_components_s or pass "
            "allow_unmeasured_overhead=True explicitly for non-deployable analysis"
        )

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in direct_rows:
        grouped[(str(row["instance"]), str(row["backend"]))].append(row)
    costs = {b: [] for b in backends}
    for instance in instances:
        for backend in backends:
            good = []
            for r in grouped.get((instance, backend), []):
                if _row_is_benchmark_success(r):
                    v = _get_cost(r, cost_field)
                    if v is not None: good.append(min(v, cutoff_s))
            costs[backend].append(float(median(good)) if good else cutoff_s * par_penalty)

    choices = []; policy_costs = []; overhead_values: list[float] = []
    for idx, instance in enumerate(instances):
        entry = policy[instance]; backend = entry["backend"]
        if backend not in costs:
            raise ValueError(f"policy selected backend {backend!r} not present in benchmark")
        choices.append(backend)
        raw_overhead = entry.get("overhead_s")
        overhead = 0.0 if raw_overhead is None else _finite_nonnegative(raw_overhead, label=f"policy overhead for {instance!r}")
        overhead_values.append(overhead); policy_costs.append(float(costs[backend][idx] + overhead))

    metrics = portfolio_metrics(costs, choices, metric=f"PAR{par_penalty:g}@{cutoff_s:g}s", policy_costs=policy_costs)
    ratio = _paired_bootstrap_policy_ratio(policy_costs, costs[metrics.sbs_solver], draws=bootstrap_draws, seed=bootstrap_seed)
    overhead_complete = not unmeasured
    deployable_cost_scope = cost_field in {"wall_s", "worker_solve_wall_s"}
    return {
        "schema_version": "1.1",
        "protocol_id": protocol_id,
        "environment_ids": sorted(environment_ids),
        "experiment_identities": sorted(experiment_ids),
        "mixed_environments_allowed": bool(allow_mixed_environments),
        "instances": len(instances),
        "cost_field": cost_field,
        "cutoff_s": cutoff_s,
        "par_penalty": par_penalty,
        "overhead_accounting": {
            "complete": overhead_complete,
            "unmeasured_instances": unmeasured,
            "allow_unmeasured_overhead": bool(allow_unmeasured_overhead),
            "assumed_zero_for_unmeasured": bool(unmeasured and allow_unmeasured_overhead),
            "mean_measured_or_assumed_overhead_s": sum(overhead_values) / len(overhead_values),
        },
        "cost_accounting": {
            "solver_cost_field": cost_field,
            "external_policy_overhead_added": True,
            "deployable_cost_scope": deployable_cost_scope,
            "deployable_evidence": bool(overhead_complete and deployable_cost_scope and not allow_mixed_environments),
        },
        "vbs": {"cost": metrics.vbs_cost, "is_oracle": True, "deployable": False},
        "policy_metrics": {
            "metric": metrics.metric,
            "solver_mean_costs": metrics.solver_mean_costs,
            "sbs_solver": metrics.sbs_solver,
            "sbs_cost": metrics.sbs_cost,
            "vbs_cost": metrics.vbs_cost,
            "policy_cost": metrics.policy_cost,
            "gap_closure": metrics.gap_closure,
            "policy_wins": metrics.policy_wins,
            "policy_ties": metrics.policy_ties,
            "policy_to_sbs_ratio_bootstrap_95pct": ratio,
        },
        "choices": {instance: policy[instance] for instance in instances},
    }


def evaluate_policy_jsonl(result_paths: list[str | Path], policy_path: str | Path, **kwargs) -> dict:
    return evaluate_policy_map(load_rows(result_paths), load_policy_map(policy_path), **kwargs)
