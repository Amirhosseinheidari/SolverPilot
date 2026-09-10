from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy

from solverpilot import ObjectiveSense, PublicStatus, SolveBudget, read_mps, solve
from solverpilot.evaluation import parse_miplib_solu, parse_test_manifest


def _instance_key(filename: str) -> str:
    name = filename
    for suffix in (".mps.gz", ".mps", ".lp.gz", ".lp"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _close(observed: float, reference: float, *, atol: float, rtol: float) -> bool:
    return bool(np.isclose(float(observed), float(reference), atol=atol, rtol=rtol))


def _reference_check(*, result, problem, reference, atol: float, rtol: float) -> str:
    if reference is None or reference.status == "unknown":
        return "not_checkable"
    if reference.status == "infeasible":
        if result.status is PublicStatus.INFEASIBLE:
            return "infeasible_matches"
        if result.status is PublicStatus.INFEASIBLE_OR_UNBOUNDED:
            return "not_proven_infeasible"
        return "infeasibility_mismatch"
    if result.objective is None or reference.objective is None:
        return "missing_solution"
    observed = float(result.objective)
    target = float(reference.objective)
    if reference.status == "optimal":
        return "objective_matches_optimum" if _close(observed, target, atol=atol, rtol=rtol) else "objective_mismatch"
    if reference.status == "best_known":
        tol = atol + rtol * max(1.0, abs(target))
        if problem.objective_sense is ObjectiveSense.MINIMIZE:
            good = observed <= target + tol
        else:
            good = observed >= target - tol
        return "meets_or_beats_best_known" if good else "worse_than_best_known"
    return "not_checkable"


def run_manifest(
    *,
    manifest_text: str,
    solu_text: str,
    data_dir: Path,
    backend: str | None,
    time_limit_s: float | None,
    objective_atol: float = 1e-6,
    objective_rtol: float = 1e-6,
) -> dict[str, object]:
    filenames = parse_test_manifest(manifest_text)
    references = parse_miplib_solu(solu_text)
    rows: list[dict[str, object]] = []
    started = perf_counter()

    for filename in filenames:
        path = data_dir / filename
        key = _instance_key(filename)
        ref = references.get(key)
        if not path.exists():
            rows.append(
                {
                    "filename": filename,
                    "instance": key,
                    "state": "missing",
                    "reference_status": None if ref is None else ref.status,
                    "reference_objective": None if ref is None else ref.objective,
                }
            )
            continue

        parse_t0 = perf_counter()
        try:
            problem = read_mps(path)
        except Exception as exc:
            rows.append(
                {
                    "filename": filename,
                    "instance": key,
                    "state": "parse_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "reference_status": None if ref is None else ref.status,
                    "reference_objective": None if ref is None else ref.objective,
                }
            )
            continue
        parse_s = perf_counter() - parse_t0

        try:
            result = solve(
                problem,
                backend=backend,
                budget=None if time_limit_s is None else SolveBudget(wall_time_s=time_limit_s),
            )
            rows.append(
                {
                    "filename": filename,
                    "instance": key,
                    "state": "solved",
                    "parse_s": parse_s,
                    "problem_class": result.trace.problem_class,
                    "n_variables": problem.n_variables,
                    "n_constraints": problem.n_constraints,
                    "nnz": problem.nnz,
                    "status": result.status.value,
                    "backend": result.trace.backend,
                    "backend_status": result.backend_status,
                    "objective": result.objective,
                    "validation_valid": None if result.validation is None else result.validation.valid,
                    "total_s": result.trace.timings.total_s,
                    "solve_s": result.trace.timings.solve_s,
                    "reference_status": None if ref is None else ref.status,
                    "reference_objective": None if ref is None else ref.objective,
                    "reference_check": _reference_check(
                        result=result,
                        problem=problem,
                        reference=ref,
                        atol=objective_atol,
                        rtol=objective_rtol,
                    ),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "filename": filename,
                    "instance": key,
                    "state": "solve_error",
                    "parse_s": parse_s,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "reference_status": None if ref is None else ref.status,
                    "reference_objective": None if ref is None else ref.objective,
                }
            )

    state_counts: dict[str, int] = {}
    for row in rows:
        state = str(row["state"])
        state_counts[state] = state_counts.get(state, 0) + 1
    solved_rows = [r for r in rows if r["state"] == "solved"]
    checkable = [r for r in solved_rows if r.get("reference_check") not in {None, "not_checkable"}]
    match_labels = {"objective_matches_optimum", "meets_or_beats_best_known", "infeasible_matches"}
    mismatch_labels = {"objective_mismatch", "infeasibility_mismatch", "worse_than_best_known"}
    inconclusive_labels = {"missing_solution", "not_proven_infeasible"}
    return {
        "benchmark": "manifest-driven MPS benchmark runner",
        "instances_in_manifest": len(filenames),
        "state_counts": state_counts,
        "reference_checks": len(checkable),
        "reference_matches": sum(r.get("reference_check") in match_labels for r in checkable),
        "reference_mismatches": sum(r.get("reference_check") in mismatch_labels for r in checkable),
        "reference_inconclusive": sum(r.get("reference_check") in inconclusive_labels for r in checkable),
        "reference_check_counts": {
            label: sum(r.get("reference_check") == label for r in solved_rows)
            for label in sorted({str(r.get("reference_check")) for r in solved_rows})
        },
        "backend_request": backend,
        "time_limit_s": time_limit_s,
        "objective_atol": objective_atol,
        "objective_rtol": objective_rtol,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "wall_s": perf_counter() - started,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--solu", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--time-limit-s", type=float, default=None)
    args = parser.parse_args()

    payload = run_manifest(
        manifest_text=args.manifest.read_text(encoding="utf-8"),
        solu_text=args.solu.read_text(encoding="utf-8"),
        data_dir=args.data_dir,
        backend=args.backend,
        time_limit_s=args.time_limit_s,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("instances_in_manifest", "state_counts", "reference_checks", "reference_matches", "reference_mismatches", "reference_inconclusive")}, indent=2))


if __name__ == "__main__":
    main()
