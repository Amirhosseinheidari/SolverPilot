"""JSON execution manifests. Complete model data is exported only by opt-in."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from functools import lru_cache
from importlib.metadata import version, PackageNotFoundError
import json
import platform
from pathlib import Path
import numpy as np
from scipy import sparse


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


def run_manifest(problem, result, *, include_model=False):
    from solverpilot.problem import LinearProblem, QuadraticProblem

    trace = getattr(result, "trace", None)
    actual = (
        trace.problem_data_hash if trace is not None else getattr(result, "problem_data_hash", None)
    )
    if actual is None or actual != getattr(problem, "data_hash", None):
        raise ValueError("result provenance does not match supplied problem")
    raw = result.raw_statistics or {}
    payload = {
        "schema": "solverpilot.run.v1",
        "problem_data_hash": actual,
        "environment": environment(),
        "backend": trace.backend if trace else result.backend,
        "parameters": trace.parameters if trace else raw.get("run_parameters", {}),
        "status": getattr(result, "status", getattr(result, "backend_status", "unknown")),
        "statistics": raw,
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
    return json_value(payload)


def save_run(path, problem, result, *, include_model=False):
    payload = run_manifest(problem, result, include_model=include_model)
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return Path(path)


def replay_run(path, *, require_versions=True):
    """Replay a trusted local LP/QP JSON manifest, never executable code/pickle."""
    from dataclasses import replace
    from solverpilot.problem import LinearProblem, QuadraticProblem
    from solverpilot.validate import ValidationTolerances
    from solverpilot.plan import SolveBudget
    from .auto import builtin_backend_candidates, solve

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "solverpilot.run.v1" or "model" not in payload:
        raise ValueError("a supported manifest with model data is required")
    if require_versions and payload["environment"]["packages"] != environment()["packages"]:
        raise ValueError("package versions differ from the recorded run")
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
    backend = replace(backends[payload["backend"]], **parameters.get("backend_configuration", {}))
    budget = parameters.get("budget")
    return solve(
        problem,
        backend=backend,
        tolerances=ValidationTolerances(**parameters.get("validation_tolerances", {})),
        budget=None if budget is None else SolveBudget(**budget),
    )
