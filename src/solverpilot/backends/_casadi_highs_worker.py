"""Process boundary for CasADi's HiGHS plugin; array transport never uses pickle."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from scipy import sparse

from solverpilot.problem import LinearProblem, QuadraticProblem
from solverpilot.problem.quadratic import require_confirmed_convexity
from .base import BackendSolveResult, BackendUnavailableError


def solve_isolated(backend, problem):
    if not isinstance(problem, (LinearProblem, QuadraticProblem)):
        raise TypeError(f"unsupported problem type: {type(problem)!r}")
    if isinstance(problem, QuadraticProblem):
        require_confirmed_convexity(problem)
    if not backend.is_available():
        raise BackendUnavailableError("CasADi conic plugin is unavailable: highs")
    linear = problem.linear if isinstance(problem, QuadraticProblem) else problem
    with tempfile.TemporaryDirectory(prefix="solverpilot-casadi-highs-") as temporary:
        root = Path(temporary)
        sparse.save_npz(root / "A.npz", linear.A)
        if isinstance(problem, QuadraticProblem):
            sparse.save_npz(root / "P.npz", problem.P)
        np.savez(root / "vectors.npz", c=linear.c, lower=linear.variable_lower,
                 upper=linear.variable_upper, row_lower=linear.constraint_lower,
                 row_upper=linear.constraint_upper, domains=linear.domains)
        (root / "options.json").write_text(json.dumps({
            "quadratic": isinstance(problem, QuadraticProblem),
            "sense": linear.objective_sense.value, "offset": float(linear.objective_offset),
            "time_limit_s": backend.time_limit_s,
        }), encoding="utf-8")
        timeout = None if backend.time_limit_s is None else backend.time_limit_s + 30.0
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "solverpilot.backends._casadi_highs_worker", str(root)],
                capture_output=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return BackendSolveResult("solver_error", None, raw_statistics={
                "process_isolated": True, "error": "CasADi HiGHS worker exceeded controller timeout"})
        output = root / "result.json"
        if completed.returncode != 0 or not output.is_file():
            return BackendSolveResult("solver_error", None, raw_statistics={
                "process_isolated": True, "worker_returncode": completed.returncode,
                "error": completed.stderr.decode("utf-8", errors="replace")[-2000:]})
        result = json.loads(output.read_text(encoding="utf-8"))
        statistics = result["raw_statistics"] or {}
        statistics["process_isolated"] = True
        return BackendSolveResult(result["backend_status"],
            None if result["x"] is None else np.asarray(result["x"], dtype=np.float64),
            result["objective_reported"], statistics)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported worker result value: {type(value)!r}")


def main():
    from .casadi_conic import CasadiHighsBridgeBackend
    root = Path(sys.argv[1])
    options = json.loads((root / "options.json").read_text(encoding="utf-8"))
    with np.load(root / "vectors.npz", allow_pickle=False) as vectors:
        linear = LinearProblem.from_data(A=sparse.load_npz(root / "A.npz"), c=vectors["c"],
            variable_lower=vectors["lower"], variable_upper=vectors["upper"],
            constraint_lower=vectors["row_lower"], constraint_upper=vectors["row_upper"],
            domains=vectors["domains"], objective_sense=options["sense"],
            objective_offset=options["offset"])
    problem = linear
    if options["quadratic"]:
        from solverpilot.problem import ConvexityStatus
        problem = QuadraticProblem(linear, sparse.load_npz(root / "P.npz"), ConvexityStatus.CONFIRMED)
    backend = CasadiHighsBridgeBackend(time_limit_s=options["time_limit_s"])
    result = backend._solve_in_process(problem)
    (root / "result.json").write_text(json.dumps({
        "backend_status": result.backend_status, "x": result.x,
        "objective_reported": result.objective_reported, "raw_statistics": result.raw_statistics,
    }, default=_json_value), encoding="utf-8")


if __name__ == "__main__":
    main()
