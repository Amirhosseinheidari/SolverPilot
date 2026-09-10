from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np

from solverpilot import ObjectiveSense, PublicStatus, SolveBudget, read_mps, solve
from solverpilot.backends import BackendRegistry
from solverpilot.evaluation import assess_benchmark_row, parse_miplib_solu, parse_test_manifest

from .acquire import sha256_file
from .environment import (
    benchmark_environment_fingerprint,
    capture_environment,
    thread_environment,
)
from .integrity import (
    CURRENT_BENCHMARK_ROW_SCHEMA,
    attach_row_integrity,
    canonical_sha256,
    experiment_identity as compute_experiment_identity,
    verify_row_integrity,
)
from .model import RunSpec
from .registry import benchmark_registry


_BENCHMARK_PACKAGES = (
    "solverpilot",
    "numpy",
    "scipy",
    "highspy",
    "osqp",
    "pyscipopt",
    "nlopt",
    "casadi",
    "benchopt",
)


def _instance_key(filename: str) -> str:
    lower = filename.lower()
    for suffix in (".mps.gz", ".mps", ".lp.gz", ".lp"):
        if lower.endswith(suffix):
            return filename[: -len(suffix)]
    return Path(filename).stem


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
        return "objective_matches_optimum" if np.isclose(observed, target, atol=atol, rtol=rtol) else "objective_mismatch"
    tol = atol + rtol * max(1.0, abs(target))
    if problem.objective_sense is ObjectiveSense.MINIMIZE:
        return "meets_or_beats_best_known" if observed <= target + tol else "worse_than_best_known"
    return "meets_or_beats_best_known" if observed >= target - tol else "worse_than_best_known"


def _row_id(protocol_id: str, instance: str, instance_sha256: str, backend: str, repetition: int) -> str:
    raw = f"{protocol_id}\0{instance}\0{instance_sha256}\0{backend}\0{repetition}".encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def _protocol_id(spec: RunSpec, registry: BackendRegistry | None) -> str:
    backend_versions: dict[str, str | None] = {}
    if registry is not None:
        for backend in registry.all():
            backend_versions[backend.manifest.name] = backend.manifest.version
    payload = {
        "schema_version": CURRENT_BENCHMARK_ROW_SCHEMA,
        "manifest_sha256": sha256_file(spec.manifest),
        "reference_sha256": None if spec.reference is None else sha256_file(spec.reference),
        "backends": list(spec.backends),
        "backend_versions": backend_versions,
        "repetitions": spec.repetitions,
        "time_limit_s": spec.time_limit_s,
        "hard_timeout_s": spec.hard_timeout_s,
        "objective_atol": spec.objective_atol,
        "objective_rtol": spec.objective_rtol,
        "seed": spec.seed,
        "thread_env_limit": spec.thread_env_limit,
        "solver_threads": spec.solver_threads,
        "worker_python_mode": spec.worker_python_mode,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    completed: set[str] = set()
    valid_end = 0
    for idx, raw in enumerate(lines):
        line_no = idx + 1
        text = raw.decode("utf-8", errors="strict").strip()
        if not text:
            valid_end += len(raw)
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            # A killed worker/controller may leave only the final JSONL record truncated.
            if idx == len(lines) - 1:
                with path.open("r+b") as f:
                    f.truncate(valid_end)
                break
            raise ValueError(f"invalid JSONL at line {line_no}") from exc
        verify_row_integrity(row, context=f"{path}:line {line_no}")
        valid_end += len(raw)
        if row.get("complete") is True and isinstance(row.get("run_id"), str):
            completed.add(row["run_id"])
    return completed


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    if row.get("state") == "solved" and "proof_assessment" not in row:
        row["proof_assessment"] = assess_benchmark_row(row).to_dict()
    payload = attach_row_integrity(row)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _resolve_instance(data_dir: Path, filename: str) -> Path | None:
    candidates = [data_dir / filename, data_dir / "instances" / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(data_dir.rglob(filename))
    return matches[0] if len(matches) == 1 else None


def _stable_shard_index(filename: str, shard_count: int) -> int:
    digest = hashlib.sha256(filename.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % shard_count


def _task_order(filename: str, *, backends: tuple[str, ...], repetitions: int, seed: int) -> list[tuple[str, int]]:
    tasks = [(backend, rep) for backend in backends for rep in range(repetitions)]
    raw = hashlib.sha256(f"{seed}\0{filename}".encode()).digest()
    local_seed = int.from_bytes(raw[:8], "big")
    random.Random(local_seed).shuffle(tasks)
    return tasks


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


def _clean_pythonpath() -> str:
    """Build an explicit import path for ``python -S`` benchmark workers.

    ``-S`` deliberately disables ``site`` and therefore platform/user sitecustomize
    hooks.  We re-expose only already-resolved import roots from the controller,
    excluding known hook-only paths.  The exact mode is protocol-versioned.
    """

    blocked_components = {"python-hooks", "pyvenv-overrides"}

    def is_blocked(path: str) -> bool:
        # ``sys.path`` may contain Windows paths even when tests run elsewhere.
        # Normalize separators textually so the hook filter is OS-agnostic.
        components = {part.casefold() for part in path.replace("\\", "/").split("/") if part}
        return bool(components & blocked_components)

    roots: list[str] = []
    for raw in sys.path:
        path = os.getcwd() if raw == "" else raw
        if not path or is_blocked(path):
            continue
        try:
            resolved = str(Path(path).resolve())
        except Exception:
            resolved = str(path)
        if resolved not in roots and Path(resolved).exists():
            roots.append(resolved)
    return os.pathsep.join(roots)


def _isolated_solve(
    path: Path,
    *,
    backend: str,
    portfolio: tuple[str, ...],
    time_limit_s: float | None,
    hard_timeout_s: float | None,
    thread_env_limit: int | None,
    solver_threads: int | None,
    worker_python_mode: str,
) -> dict:
    cmd = [sys.executable]
    if worker_python_mode == "no_site":
        cmd.append("-S")
    cmd.extend(["-m", "solverpilot.cli.benchmark_worker", "--path", str(path), "--backend", backend])
    for name in portfolio:
        if name != "@auto":
            cmd.extend(["--portfolio", name])
    if time_limit_s is not None:
        cmd.extend(["--time-limit-s", str(time_limit_s)])
    if solver_threads is not None:
        cmd.extend(["--solver-threads", str(solver_threads)])
    env = os.environ.copy()
    env.update(thread_environment(thread_env_limit))
    if worker_python_mode == "no_site":
        env["PYTHONPATH"] = _clean_pythonpath()
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=hard_timeout_s, env=env)
    except subprocess.TimeoutExpired as exc:
        return {"worker_state": "hard_timeout", "error_type": "TimeoutExpired", "error": str(exc)}
    if proc.returncode != 0:
        err = proc.stderr.strip()
        try:
            payload = json.loads(err.splitlines()[-1]) if err else {}
        except Exception:
            payload = {}
        return {
            "worker_state": "worker_error",
            "error_type": payload.get("error_type", "WorkerProcessError"),
            "error": payload.get("error", err[-2000:]),
            "worker_returncode": proc.returncode,
        }
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {"worker_state": "worker_error", "error_type": type(exc).__name__, "error": "invalid worker JSON output"}
    return {"worker_state": "solved", **payload}


def _backend_identity(registry: BackendRegistry, backends: tuple[str, ...]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for name in backends:
        if name == "@auto":
            continue
        backend = registry.get(name)
        manifest = backend.manifest
        out[name] = {
            "name": manifest.name,
            "version": manifest.version,
            "capabilities": {k.value: v.value for k, v in sorted(manifest.capabilities.items(), key=lambda kv: kv[0].value)},
            "metadata": dict(manifest.metadata),
            "implementation": f"{type(backend).__module__}.{type(backend).__qualname__}",
        }
    return out


def _experiment_identity(backend_set_sha256: str, environment_id: str, thread_policy: dict[str, object], protocol_id: str) -> str:
    return compute_experiment_identity(
        protocol_id=protocol_id, environment_id=environment_id,
        backend_set_sha256=backend_set_sha256, thread_policy=thread_policy,
    )


def run_benchmark(spec: RunSpec, *, registry: BackendRegistry | None = None) -> dict:
    registry = benchmark_registry(spec.backends) if registry is None else registry
    manifest_text = spec.manifest.read_text(encoding="utf-8")
    filenames = parse_test_manifest(manifest_text)
    refs = {} if spec.reference is None else parse_miplib_solu(spec.reference.read_text(encoding="utf-8"))
    completed = _load_completed(spec.output_jsonl) if spec.resume else set()
    protocol_id = _protocol_id(spec, registry)
    if not spec.resume and spec.output_jsonl.exists():
        spec.output_jsonl.unlink()

    environment = capture_environment(packages=_BENCHMARK_PACKAGES)
    environment_id = benchmark_environment_fingerprint(
        environment,
        thread_env_limit=spec.thread_env_limit,
        solver_threads=spec.solver_threads,
        worker_python_mode=spec.worker_python_mode,
    )
    thread_policy = {
        "thread_env_limit": spec.thread_env_limit,
        "solver_threads": spec.solver_threads,
        "worker_python_mode": spec.worker_python_mode,
        "env_control": "process_start" if spec.thread_env_limit is not None else "none",
        "backend_thread_control": "requested" if spec.solver_threads is not None else "not_requested",
    }
    backend_identity = _backend_identity(registry, tuple(b for b in spec.backends if b != "@auto"))
    backend_set_sha256 = canonical_sha256(backend_identity)
    experiment_identity = _experiment_identity(backend_set_sha256, environment_id, thread_policy, protocol_id)

    chosen_files = [f for f in filenames if _stable_shard_index(f, spec.shard_count) == spec.shard_index]
    rng = random.Random(spec.seed)
    rng.shuffle(chosen_files)

    counters = {"scheduled": 0, "skipped_resume": 0, "missing": 0, "completed": 0, "errors": 0}
    started = perf_counter()
    for filename in chosen_files:
        instance = _instance_key(filename)
        path = _resolve_instance(spec.dataset_dir, filename)
        tasks = _task_order(filename, backends=spec.backends, repetitions=spec.repetitions, seed=spec.seed)
        if path is None:
            for backend, rep in tasks:
                run_id = _row_id(protocol_id, instance, "missing", backend, rep)
                if run_id in completed:
                    counters["skipped_resume"] += 1
                    continue
                _append_jsonl(spec.output_jsonl, {
                    "schema_version": CURRENT_BENCHMARK_ROW_SCHEMA,
                    "protocol_id": protocol_id,
                    "environment_id": environment_id,
                    "thread_policy": thread_policy,
                    "experiment_identity": experiment_identity,
                    "backend_set_sha256": backend_set_sha256,
                    "backend_identity": backend_identity.get(backend, {}),
                    "backend_identity_sha256": canonical_sha256(backend_identity.get(backend, {})),
                    "run_id": run_id,
                    "complete": True,
                    "instance": instance,
                    "filename": filename,
                    "backend": backend,
                    "repetition": rep,
                    "state": "missing_instance",
                })
                counters["missing"] += 1
            continue
        try:
            parse_t0 = perf_counter()
            problem = read_mps(path)
            parse_s = perf_counter() - parse_t0
            file_sha = sha256_file(path)
            parse_error = None
        except Exception as exc:
            problem = None
            parse_s = None
            file_sha = sha256_file(path)
            parse_error = (type(exc).__name__, str(exc))

        for backend, rep in tasks:
            run_id = _row_id(protocol_id, instance, file_sha, backend, rep)
            counters["scheduled"] += 1
            if run_id in completed:
                counters["skipped_resume"] += 1
                continue
            base = {
                "schema_version": CURRENT_BENCHMARK_ROW_SCHEMA,
                "protocol_id": protocol_id,
                "environment_id": environment_id,
                "thread_policy": thread_policy,
                "experiment_identity": experiment_identity,
                "backend_set_sha256": backend_set_sha256,
                "backend_identity": backend_identity.get(backend, {}),
                "backend_identity_sha256": canonical_sha256(backend_identity.get(backend, {})),
                "run_id": run_id,
                "complete": True,
                "instance": instance,
                "filename": filename,
                "instance_sha256": file_sha,
                "backend": backend,
                "repetition": rep,
                "parse_s": parse_s,
            }
            if parse_error is not None:
                row = {**base, "state": "parse_error", "error_type": parse_error[0], "error": parse_error[1]}
                counters["errors"] += 1
                _append_jsonl(spec.output_jsonl, row)
                continue
            try:
                wall_t0 = perf_counter()
                ref = refs.get(instance)
                isolate = spec.hard_timeout_s is not None or spec.thread_env_limit is not None or spec.solver_threads is not None
                if isolate:
                    worker = _isolated_solve(
                        path,
                        backend=backend,
                        portfolio=spec.backends,
                        time_limit_s=spec.time_limit_s,
                        hard_timeout_s=spec.hard_timeout_s,
                        thread_env_limit=spec.thread_env_limit,
                        solver_threads=spec.solver_threads,
                        worker_python_mode=spec.worker_python_mode,
                    )
                    wall_s = perf_counter() - wall_t0
                    if worker["worker_state"] != "solved":
                        row = {
                            **base,
                            "state": worker["worker_state"],
                            "wall_s": wall_s,
                            "error_type": worker.get("error_type"),
                            "error": worker.get("error"),
                        }
                        counters["errors"] += 1
                        _append_jsonl(spec.output_jsonl, row)
                        continue

                    class _Result:
                        pass

                    result_proxy = _Result()
                    result_proxy.status = next(s for s in PublicStatus if s.value == worker["public_status"])
                    result_proxy.objective = worker.get("objective")
                    row = {
                        **base,
                        "state": "solved",
                        "wall_s": wall_s,
                        "public_status": worker["public_status"],
                        "objective": worker.get("objective"),
                        "validated": worker.get("validated"),
                        "reference_status": None if ref is None else ref.status,
                        "reference_objective": None if ref is None else ref.objective,
                        "reference_check": _reference_check(
                            result=result_proxy,
                            problem=problem,
                            reference=ref,
                            atol=spec.objective_atol,
                            rtol=spec.objective_rtol,
                        ),
                        "trace": worker.get("trace", {}),
                        "worker_parse_s": worker.get("parse_s"),
                        "worker_solve_wall_s": worker.get("worker_solve_wall_s"),
                        "optimality_evidence": worker.get("optimality_evidence"),
                    }
                else:
                    budget = None
                    if spec.time_limit_s is not None or spec.solver_threads is not None:
                        budget = SolveBudget(wall_time_s=spec.time_limit_s, threads=spec.solver_threads)
                    result = solve(
                        problem,
                        registry=registry,
                        backend=None if backend == "@auto" else backend,
                        budget=budget,
                    )
                    wall_s = perf_counter() - wall_t0
                    row = {
                        **base,
                        "state": "solved",
                        "wall_s": wall_s,
                        "public_status": result.status.value,
                        "objective": result.objective,
                        "validated": None if result.validation is None else result.validation.valid,
                        "reference_status": None if ref is None else ref.status,
                        "reference_objective": None if ref is None else ref.objective,
                        "reference_check": _reference_check(
                            result=result,
                            problem=problem,
                            reference=ref,
                            atol=spec.objective_atol,
                            rtol=spec.objective_rtol,
                        ),
                        "trace": _trace_payload(result.trace),
                        "optimality_evidence": {
                            "backend_reported_optimal": result.optimality_evidence.backend_reported_optimal,
                            "primal_validated": result.optimality_evidence.primal_validated,
                            "dual_verified": result.optimality_evidence.dual_verified,
                            "gap_verified": result.optimality_evidence.gap_verified,
                            "certificate_verified": result.optimality_evidence.certificate_verified,
                            "independently_verified_optimal": result.optimality_evidence.independently_verified_optimal,
                        },
                    }
                counters["completed"] += 1
            except Exception as exc:
                row = {**base, "state": "solve_error", "error_type": type(exc).__name__, "error": str(exc)}
                counters["errors"] += 1
            _append_jsonl(spec.output_jsonl, row)

    return {
        "schema_version": CURRENT_BENCHMARK_ROW_SCHEMA,
        "protocol_id": protocol_id,
        "environment_id": environment_id,
        "experiment_identity": experiment_identity,
        "backend_set_sha256": backend_set_sha256,
        "manifest": str(spec.manifest),
        "manifest_sha256": sha256_file(spec.manifest),
        "reference": None if spec.reference is None else str(spec.reference),
        "reference_sha256": None if spec.reference is None else sha256_file(spec.reference),
        "dataset_dir": str(spec.dataset_dir),
        "backends": list(spec.backends),
        "repetitions": spec.repetitions,
        "time_limit_s": spec.time_limit_s,
        "hard_timeout_s": spec.hard_timeout_s,
        "thread_policy": thread_policy,
        "shard": {"index": spec.shard_index, "count": spec.shard_count, "instances": len(chosen_files)},
        "counters": counters,
        "elapsed_s": perf_counter() - started,
        "environment": environment,
    }
