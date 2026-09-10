from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from solverpilot import SolveBudget, read_mps, solve

from .registry import benchmark_registry


def _trace_payload(trace) -> dict[str, object]:
    total = float(trace.timings.total_s)
    solve_s = float(trace.timings.solve_s)
    return {
        "backend_version": trace.backend_version,
        "inspect_s": float(trace.timings.inspect_s),
        "plan_s": float(trace.timings.plan_s),
        "backend_build_s": float(trace.timings.backend_build_s),
        "solve_s": solve_s,
        "validate_s": float(trace.timings.validate_s),
        "diagnose_s": float(trace.timings.diagnose_s),
        "total_s": total,
        "runtime_overhead_s": max(0.0, total - solve_s),
        "reuse_applied": trace.reuse_applied,
        "reuse_mode": trace.reuse_mode,
        "planner_selected_backend": trace.planner_selected_backend,
        "planner_evidence_level": trace.planner_evidence_level,
    }


def solve_file_worker(
    *,
    path: str | Path,
    backend: str,
    portfolio: tuple[str, ...],
    time_limit_s: float | None,
    solver_threads: int | None = None,
) -> dict:
    parse_t0 = perf_counter()
    problem = read_mps(path)
    parse_s = perf_counter() - parse_t0
    registry = benchmark_registry(portfolio)
    solve_t0 = perf_counter()
    budget = None
    if time_limit_s is not None or solver_threads is not None:
        budget = SolveBudget(wall_time_s=time_limit_s, threads=solver_threads)
    result = solve(
        problem,
        registry=registry,
        backend=None if backend == "@auto" else backend,
        budget=budget,
    )
    controller_s = perf_counter() - solve_t0
    evidence = result.optimality_evidence
    return {
        "parse_s": parse_s,
        "worker_solve_wall_s": controller_s,
        "public_status": result.status.value,
        "objective": result.objective,
        "validated": None if result.validation is None else result.validation.valid,
        "optimality_evidence": {
            "backend_reported_optimal": evidence.backend_reported_optimal,
            "primal_validated": evidence.primal_validated,
            "dual_verified": evidence.dual_verified,
            "gap_verified": evidence.gap_verified,
            "certificate_verified": evidence.certificate_verified,
            "independently_verified_optimal": evidence.independently_verified_optimal,
        },
        "trace": _trace_payload(result.trace),
    }
