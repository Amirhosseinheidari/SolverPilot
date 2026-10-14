"""Versioned JSON execution manifests with explicit export and replay scope."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from functools import lru_cache
from importlib.metadata import version, PackageNotFoundError
import json
import platform
from pathlib import Path
import numpy as np
from scipy import sparse
from solverpilot._identity import execution_id, process_source_sha256


def json_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else str(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if hasattr(value, "value"):
        return json_value(value.value)
    if is_dataclass(value):
        return {
            f.name: json_value(getattr(value, f.name))
            for f in fields(value)
            if f.init and not f.name.startswith("_")
        }
    raise TypeError(f"not JSON-serializable: {type(value).__name__}")


def backend_configuration(backend):
    from .budgeting import _ConfiguredBackend

    source = backend.backend if isinstance(backend, _ConfiguredBackend) else backend
    result = {}
    if is_dataclass(source):
        for field in fields(source):
            if field.init and not field.name.startswith("_"):
                value = getattr(backend, field.name)
                try:
                    result[field.name] = json_value(value)
                except TypeError:
                    result[field.name] = {"runtime_only": type(value).__name__}
    return result


@lru_cache(maxsize=1)
def environment():
    packages = {}
    for name in (
        "solverpilot",
        "numpy",
        "scipy",
        "highspy",
        "osqp",
        "clarabel",
        "casadi",
        "ortools",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    from solverpilot import __version__

    packages["solverpilot"] = __version__
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _matrix(matrix):
    matrix = matrix.tocsr()
    return {
        "shape": list(matrix.shape),
        "data": matrix.data.tolist(),
        "indices": matrix.indices.tolist(),
        "indptr": matrix.indptr.tolist(),
    }


def _seal(payload):
    from solverpilot.benchmark.integrity import canonical_sha256
    payload = json_value(payload)
    payload.pop("integrity_sha256", None)
    payload["integrity_sha256"] = canonical_sha256(payload)
    return payload


def _verify_integrity(payload):
    from solverpilot.benchmark.integrity import canonical_sha256
    json.dumps(payload, allow_nan=False)
    recorded = payload.get("integrity_sha256")
    if not isinstance(recorded, str) or recorded != canonical_sha256(
        {k: v for k, v in payload.items() if k != "integrity_sha256"}
    ):
        raise ValueError("manifest integrity hash mismatch")


def _summary_payload(result):
    from .unified import summarize
    summary = summarize(result)
    # Deliberate allowlist: no vectors, assignments, free-text backend reasons,
    # callback objects, paths, arbitrary statistics or user metadata.
    payload = {name: getattr(summary, name) for name in (
        "run_id", "status", "objective", "feasible", "optimality", "backend",
        "problem_data_hash", "termination_evidence", "requested_time_s",
        "elapsed_s", "within_budget", "deadline_enforcement",
    )}
    for name in ("objective", "requested_time_s", "elapsed_s"):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
            payload[name] = None
    if type(payload["within_budget"]) is not bool:
        payload["within_budget"] = None
    if payload["deadline_enforcement"] not in {
        "not_reported", "not_requested", "process_deadline_with_cleanup",
        "native_soft_limit; use solve_with_deadline for process isolation",
    }:
        payload["deadline_enforcement"] = "not_reported"
    return payload


def run_manifest(problem, result, *, include_model=False, include_raw_statistics=False):
    """Export v2; model/parameters and raw backend statistics require separate opt-ins.

    The summary policy still discloses identities, objective and timings. It is
    data minimization, not anonymization. Integrity hashes detect corruption,
    not an adversary who can replace both content and hash.
    """
    from solverpilot.problem import LinearProblem, QuadraticProblem

    trace = getattr(result, "trace", None)
    actual = (
        trace.problem_data_hash if trace is not None else getattr(result, "problem_data_hash", None)
    )
    if actual is None or actual != getattr(problem, "data_hash", None):
        raise ValueError("result provenance does not match supplied problem")
    identifier = execution_id(result)
    if identifier is None:
        raise ValueError("result has no stable execution identity")
    raw = getattr(result, "raw_statistics", None) or {}
    payload = {
        "schema": "solverpilot.run.v2",
        "execution_id": identifier,
        "problem_data_hash": actual,
        "environment": environment(),
        "code_identity": {
            "source_sha256": getattr(trace, "source_sha256", None),
            "scope": "package Python source snapshot at first traced execution in process",
            "binary_attestation": False,
        },
        "backend": trace.backend if trace else result.backend,
        "parameters": (trace.parameters if trace else raw.get("run_parameters", {})) if include_model else {},
        "status": getattr(result, "status", getattr(result, "backend_status", "unknown")),
        "statistics": raw if include_raw_statistics else {},
        "summary": _summary_payload(result),
        "replay_of": getattr(trace, "replay_of", None),
        "export_policy": {
            "model_and_parameters": bool(include_model),
            "raw_backend_statistics": bool(include_raw_statistics),
            "summary_includes_objective": True,
        },
    }
    if include_model:
        if not isinstance(problem, (LinearProblem, QuadraticProblem)):
            raise TypeError("model replay currently supports canonical LP/QP")
        linear = problem.linear if isinstance(problem, QuadraticProblem) else problem
        payload["model"] = {
            "kind": "qp" if isinstance(problem, QuadraticProblem) else "lp",
            "A": _matrix(linear.A),
            "c": linear.c,
            "variable_lower": linear.variable_lower,
            "variable_upper": linear.variable_upper,
            "constraint_lower": linear.constraint_lower,
            "constraint_upper": linear.constraint_upper,
            "domains": linear.domains.tolist(),
            "objective_sense": linear.objective_sense.value,
            "objective_offset": linear.objective_offset,
        }
        if isinstance(problem, QuadraticProblem):
            payload["model"]["P"] = _matrix(problem.P)
    return _seal(payload)


def save_run(path, problem, result, *, include_model=False, include_raw_statistics=False):
    payload = run_manifest(problem, result, include_model=include_model,
                           include_raw_statistics=include_raw_statistics)
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return Path(path)


def replay_run(path, *, require_versions=True, require_source=None):
    """Replay trusted local LP/QP JSON, never executable code/pickle.

    By default v2 requires source identity; legacy v1 retains version-only
    compatibility. Explicit require_source=True also rejects legacy manifests
    without source evidence. False explicitly permits source drift.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _replay_payload(payload, require_versions=require_versions, require_source=require_source)


def _replay_payload(payload, *, require_versions=True, require_source=None):
    from dataclasses import replace
    from solverpilot.problem import LinearProblem, QuadraticProblem
    from solverpilot.validate import ValidationTolerances
    from solverpilot.plan import SolveBudget
    from .auto import builtin_backend_candidates, solve

    schema = payload.get("schema")
    if schema not in {"solverpilot.run.v1", "solverpilot.run.v2"} or "model" not in payload:
        raise ValueError("a supported manifest with model data is required")
    if require_versions and payload["environment"]["packages"] != environment()["packages"]:
        raise ValueError("package versions differ from the recorded run")
    if schema == "solverpilot.run.v2":
        _verify_integrity(payload)
        source = payload.get("code_identity", {}).get("source_sha256")
        if require_source is not False and (source is None or source != process_source_sha256()):
            raise ValueError("package source identity differs or is unavailable")
    elif require_source is True:
        raise ValueError("legacy manifest has no package source identity")
    data = dict(payload["model"])

    def matrix(value):
        return sparse.csr_matrix(
            (value["data"], value["indices"], value["indptr"]), shape=value["shape"]
        )

    kind = data.pop("kind")
    data["A"] = matrix(data["A"])
    if kind == "qp":
        data.pop("domains")
        data["P"] = matrix(data["P"])
        data["q"] = data.pop("c")
        problem = QuadraticProblem.from_data(**data)
    elif kind == "lp":
        problem = LinearProblem.from_data(**data)
    else:
        raise ValueError("unsupported model kind")
    if problem.data_hash != payload["problem_data_hash"]:
        raise ValueError("model data hash differs from manifest")
    parameters = payload["parameters"]
    backends = {b.manifest.name: b for b in builtin_backend_candidates()}
    if payload["backend"] not in backends:
        raise ValueError("replay requires an available built-in backend")
    backend = replace(backends[payload["backend"]], **parameters.get("backend_configuration", {}))
    budget = parameters.get("budget")
    result = solve(
        problem,
        backend=backend,
        tolerances=ValidationTolerances(**parameters.get("validation_tolerances", {})),
        budget=None if budget is None else SolveBudget(**budget),
    )
    return replace(result, trace=replace(result.trace, replay_of=payload.get("execution_id")))
