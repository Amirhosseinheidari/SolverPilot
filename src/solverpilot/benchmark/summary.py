from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import json
import math
import random
from pathlib import Path
from statistics import median
from typing import Iterable

from solverpilot.evaluation import performance_profile, portfolio_metrics

from .integrity import (
    CURRENT_BENCHMARK_ROW_SCHEMA,
    canonical_sha256,
    experiment_identity as compute_experiment_identity,
    verify_row_integrity,
)


_SUCCESS_REFERENCE = {"objective_matches_optimum", "meets_or_beats_best_known", "infeasible_matches"}


def _independently_verified_optimal(row: dict) -> bool:
    evidence = row.get("optimality_evidence")
    return isinstance(evidence, dict) and evidence.get("independently_verified_optimal") is True


def _row_is_benchmark_success(row: dict) -> bool:
    if row.get("state") != "solved" or row.get("validated") is not True:
        return False
    check = row.get("reference_check")
    if check in _SUCCESS_REFERENCE:
        return True
    if check == "not_checkable":
        # Backend-reported optimality is not an independent proof.  Without a frozen
        # reference objective, exact-optimization metrics require the explicit trust
        # evidence produced by SolverPilot's result layer.
        return row.get("public_status") == "valid_optimal" and _independently_verified_optimal(row)
    return False


def _get_cost(row: dict, field: str) -> float | None:
    current = row
    for part in field.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if current is None:
        return None
    try:
        value = float(current)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def load_rows(paths: Iterable[str | Path]) -> list[dict]:
    rows: list[dict] = []
    seen_ids: set[str] = set()
    for p in paths:
        path = Path(p)
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            verify_row_integrity(row, context=f"{path}:{line_no}")
            run_id = row.get("run_id")
            if not isinstance(run_id, str):
                raise ValueError(f"{path}:{line_no}: missing run_id")
            if run_id in seen_ids:
                # Identical run IDs across merged shard/result files are ambiguous and must be rejected.
                raise ValueError(f"duplicate run_id while merging: {run_id}")
            seen_ids.add(run_id)
            rows.append(row)
    return rows


def _environment_ids(rows: list[dict]) -> set[str]:
    if any(not str(r.get("environment_id") or "").strip() for r in rows):
        raise ValueError("every timing row must provide a non-empty environment_id")
    return {str(r["environment_id"]).strip() for r in rows}


def _require_protocol_identity(rows: list[dict]) -> str:
    if any(not str(r.get("protocol_id") or "").strip() for r in rows):
        raise ValueError("every timing row must provide a non-empty protocol_id")
    ids = {str(r["protocol_id"]).strip() for r in rows}
    if len(ids) != 1:
        raise ValueError(f"cannot summarize multiple benchmark protocols together: {sorted(ids)}")
    return next(iter(ids))


def _valid_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in text)


def _require_instance_identity(rows: list[dict]) -> dict[str, str]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        instance = str(row.get("instance") or "")
        sha = row.get("instance_sha256")
        if not instance:
            raise ValueError("every benchmark row must provide a non-empty instance")
        if not _valid_sha256(sha):
            raise ValueError(f"benchmark row for {instance!r} is missing a valid instance_sha256")
        grouped[instance].add(str(sha).lower())
    mismatched = {instance: hashes for instance, hashes in grouped.items() if len(hashes) != 1}
    if mismatched:
        first = sorted(mismatched)[0]
        raise ValueError(f"instance identity mismatch for {first!r}: {sorted(mismatched[first])}")
    return {instance: next(iter(hashes)) for instance, hashes in grouped.items()}


def _require_comparable_environment(rows: list[dict], *, allow_mixed_environments: bool) -> set[str]:
    ids = _environment_ids(rows)
    if len(ids) > 1 and not allow_mixed_environments:
        raise ValueError(
            "cannot aggregate timing results from multiple benchmark environments; "
            "rerun on homogeneous hardware/software or pass allow_mixed_environments=True explicitly"
        )
    return ids


def _require_experiment_identity(rows: list[dict]) -> set[str]:
    by_environment: dict[str, set[str]] = defaultdict(set)
    ids: set[str] = set()
    for row in rows:
        raw = row.get("experiment_identity")
        schema = str(row.get("schema_version") or "")
        if raw is None:
            if schema == CURRENT_BENCHMARK_ROW_SCHEMA:
                raise ValueError("every current benchmark row must provide experiment_identity")
            continue
        text = str(raw).strip().lower()
        if not _valid_sha256(text):
            raise ValueError("benchmark row has invalid experiment_identity")
        if schema == CURRENT_BENCHMARK_ROW_SCHEMA:
            backend_set_sha256 = str(row.get("backend_set_sha256") or "").strip().lower()
            if not _valid_sha256(backend_set_sha256):
                raise ValueError("current benchmark row has invalid backend_set_sha256")
            backend_identity = row.get("backend_identity")
            backend_identity_sha256 = str(row.get("backend_identity_sha256") or "").strip().lower()
            if not isinstance(backend_identity, dict) or not _valid_sha256(backend_identity_sha256):
                raise ValueError("current benchmark row has invalid backend identity metadata")
            if canonical_sha256(backend_identity).lower() != backend_identity_sha256:
                raise ValueError("backend identity checksum mismatch")
            thread_policy = row.get("thread_policy")
            if not isinstance(thread_policy, dict):
                raise ValueError("current benchmark row is missing thread_policy")
            expected = compute_experiment_identity(
                protocol_id=str(row.get("protocol_id") or ""),
                environment_id=str(row.get("environment_id") or ""),
                backend_set_sha256=backend_set_sha256,
                thread_policy=thread_policy,
            )
            if expected.lower() != text:
                raise ValueError("experiment_identity does not match row provenance fields")
        env = str(row.get("environment_id") or "").strip()
        by_environment[env].add(text)
        ids.add(text)
    inconsistent = {env: values for env, values in by_environment.items() if len(values) > 1}
    if inconsistent:
        env = sorted(inconsistent)[0]
        raise ValueError(f"experiment identity mismatch within environment {env!r}: {sorted(inconsistent[env])}")
    return ids


def _paired_bootstrap_policy_ratio(
    policy_costs: list[float], sbs_costs: list[float], *, draws: int, seed: int
) -> dict:
    if draws < 1:
        raise ValueError("bootstrap_draws must be >= 1")
    n = len(policy_costs)
    if n != len(sbs_costs) or n == 0:
        raise ValueError("bootstrap inputs must have equal nonzero length")
    rng = random.Random(seed)
    ratios = []
    for _ in range(draws):
        idx = [rng.randrange(n) for _ in range(n)]
        p = sum(policy_costs[i] for i in idx) / n
        s = sum(sbs_costs[i] for i in idx) / n
        ratios.append(float("inf") if s == 0 else p / s)
    ratios.sort()
    def q(frac: float) -> float:
        pos = frac * (draws - 1)
        lo = int(pos); hi = min(draws - 1, lo + 1); w = pos - lo
        return ratios[lo] * (1 - w) + ratios[hi] * w
    point_s = sum(sbs_costs) / n
    point_p = sum(policy_costs) / n
    return {
        "point": float("inf") if point_s == 0 else point_p / point_s,
        "low": q(0.025),
        "median": q(0.5),
        "high": q(0.975),
        "draws": draws,
        "seed": seed,
    }


def summarize_rows(
    rows: list[dict],
    *,
    cutoff_s: float,
    par_penalty: float = 10.0,
    cost_field: str = "wall_s",
    bootstrap_draws: int = 10000,
    bootstrap_seed: int = 0,
    allow_mixed_environments: bool = False,
) -> dict:
    if cutoff_s <= 0 or par_penalty < 1:
        raise ValueError("cutoff_s must be positive and par_penalty must be >= 1")
    if not rows:
        raise ValueError("no rows to summarize")

    protocol_id = _require_protocol_identity(rows)
    instance_hashes = _require_instance_identity(rows)
    environment_ids = _require_comparable_environment(rows, allow_mixed_environments=allow_mixed_environments)
    experiment_ids = _require_experiment_identity(rows)

    backends = sorted({str(r["backend"]) for r in rows})
    instance_reps: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        instance_reps[(str(row["instance"]), str(row["backend"]))].append(row)
    instances = sorted({key[0] for key in instance_reps})

    costs: dict[str, list[float]] = {b: [] for b in backends}
    solver_stats: dict[str, dict] = {}
    rectangular = True
    for backend in backends:
        solved = 0
        valid = 0
        reference_ok = 0
        median_wall_inputs: list[float] = []
        for instance in instances:
            reps = instance_reps.get((instance, backend), [])
            if not reps:
                rectangular = False
                cost = cutoff_s * par_penalty
                costs[backend].append(cost)
                continue
            good_costs: list[float] = []
            for r in reps:
                if r.get("state") == "solved":
                    solved += 1
                    if r.get("validated") is True:
                        valid += 1
                    if _row_is_benchmark_success(r):
                        reference_ok += 1
                    if _row_is_benchmark_success(r):
                        cost_value = _get_cost(r, cost_field)
                        if cost_value is not None:
                            good_costs.append(min(cost_value, cutoff_s))
                            median_wall_inputs.append(cost_value)
            cost = median(good_costs) if good_costs else cutoff_s * par_penalty
            costs[backend].append(float(cost))
        solver_stats[backend] = {
            "rows": sum(len(instance_reps.get((i, backend), [])) for i in instances),
            "solved_rows": solved,
            "validated_rows": valid,
            "reference_ok_rows": reference_ok,
            "median_success_wall_s": None if not median_wall_inputs else float(median(median_wall_inputs)),
            "par_cost_mean": sum(costs[backend]) / len(costs[backend]),
        }

    direct_backends_for_metrics = [b for b in backends if b != "@auto"] or list(backends)
    direct_costs_for_metrics = {b: costs[b] for b in direct_backends_for_metrics}
    means = {b: sum(v) / len(v) for b, v in direct_costs_for_metrics.items()}
    sbs = min(means, key=lambda b: (means[b], b))
    vbs_choices = [min(direct_backends_for_metrics, key=lambda b: (costs[b][i], b)) for i in range(len(instances))]
    metrics = portfolio_metrics(direct_costs_for_metrics, vbs_choices, metric=f"PAR{par_penalty:g}@{cutoff_s:g}s")
    taus = (1.0, 1.1, 1.25, 1.5, 2.0, 5.0, 10.0)
    profile = performance_profile(direct_costs_for_metrics, taus)

    auto_policy = None
    if "@auto" in backends:
        direct_backends = [b for b in backends if b != "@auto"]
        if not direct_backends:
            auto_policy = {"eligible": False, "reason": "no_direct_backends"}
        else:
            direct_costs = {b: costs[b] for b in direct_backends}
            policy_choices: list[str] = []
            policy_costs: list[float] = []
            unstable: list[str] = []
            for idx, instance in enumerate(instances):
                reps = instance_reps.get((instance, "@auto"), [])
                good = [
                    r for r in reps
                    if _row_is_benchmark_success(r)
                ]
                choices = {
                    r.get("trace", {}).get("planner_selected_backend") for r in good
                    if r.get("trace", {}).get("planner_selected_backend")
                }
                if len(choices) != 1 or not choices.issubset(set(direct_backends)):
                    unstable.append(instance)
                    continue
                choice = next(iter(choices))
                policy_choices.append(choice)
                good_costs = [min(v, cutoff_s) for r in good if (v := _get_cost(r, cost_field)) is not None]
                policy_costs.append(float(median(good_costs)) if good_costs else cutoff_s * par_penalty)
            if unstable or len(policy_choices) != len(instances):
                auto_policy = {
                    "eligible": False,
                    "reason": "planner_choice_missing_unstable_or_outside_direct_portfolio",
                    "instances": len(instances),
                    "problem_instances": unstable,
                }
            elif cost_field.startswith("trace."):
                auto_policy = {
                    "eligible": False,
                    "reason": "selected cost_field excludes planner/inspection overhead; use wall_s or worker_solve_wall_s for deployable auto-policy evaluation",
                }
            else:
                pm = portfolio_metrics(
                    direct_costs, policy_choices, metric=f"PAR{par_penalty:g}@{cutoff_s:g}s", policy_costs=policy_costs
                )
                ratio_ci = _paired_bootstrap_policy_ratio(
                    policy_costs, direct_costs[pm.sbs_solver], draws=bootstrap_draws, seed=bootstrap_seed
                )
                auto_policy = {
                    "eligible": True, **asdict(pm), "choices": policy_choices,
                    "policy_to_sbs_ratio_bootstrap_95pct": ratio_ci,
                }

    return {
        "schema_version": "1.2",
        "protocol_id": protocol_id,
        "instance_identity_verified": True,
        "instance_sha256": instance_hashes,
        "environment_ids": sorted(environment_ids),
        "experiment_identities": sorted(experiment_ids),
        "mixed_environments_allowed": bool(allow_mixed_environments),
        "instances": len(instances),
        "backends": backends,
        "rectangular_input": rectangular,
        "cutoff_s": cutoff_s,
        "cost_field": cost_field,
        "cost_accounting": {
            "field": cost_field,
            "auto_decision_overhead_included": cost_field in {"wall_s", "worker_solve_wall_s"},
            "scope": (
                "controller end-to-end wall time including worker startup/parse/solve"
                if cost_field == "wall_s"
                else "SolverPilot solve call wall time including inspect/plan/build/validate/diagnose"
                if cost_field == "worker_solve_wall_s"
                else "internal trace component; not deployable policy total cost"
            ),
        },
        "par_penalty": par_penalty,
        "bootstrap": {"draws": bootstrap_draws, "seed": bootstrap_seed},
        "solver_stats": solver_stats,
        "sbs_solver": sbs,
        "sbs_cost": means[sbs],
        "vbs_cost": metrics.vbs_cost,
        "vbs": {"cost": metrics.vbs_cost, "is_oracle": True, "deployable": False},
        "sbs_vbs_gap": means[sbs] - metrics.vbs_cost,
        "relative_sbs_vbs_gap": None if means[sbs] <= 0 else (means[sbs] - metrics.vbs_cost) / means[sbs],
        "performance_profile": {"taus": list(taus), "fractions": profile},
        "auto_policy_metrics": auto_policy,
    }


def summarize_jsonl(
    paths: Iterable[str | Path],
    *,
    cutoff_s: float,
    par_penalty: float = 10.0,
    cost_field: str = "wall_s",
    bootstrap_draws: int = 10000,
    bootstrap_seed: int = 0,
    allow_mixed_environments: bool = False,
) -> dict:
    return summarize_rows(
        load_rows(paths),
        cutoff_s=cutoff_s,
        par_penalty=par_penalty,
        cost_field=cost_field,
        bootstrap_draws=bootstrap_draws,
        bootstrap_seed=bootstrap_seed,
        allow_mixed_environments=allow_mixed_environments,
    )
