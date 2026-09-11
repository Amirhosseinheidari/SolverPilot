"""Opt-in model export and checked LP replay in a temporary directory."""

from pathlib import Path
from tempfile import TemporaryDirectory
from solverpilot.applications import production_model
from solverpilot.runtime import solve_verified
from solverpilot.runtime.manifest import save_run, replay_run

problem = production_model([3.0, 2.0], [[1.0, 1.0]], [4.0]).compile().execution_ir
summary, result = solve_verified(problem, backend="scipy-highs-ds")
with TemporaryDirectory() as folder:
    path = Path(folder) / "run.json"
    save_run(path, problem, result, include_model=True)
    replayed = replay_run(path)
    assert abs(replayed.objective - result.objective) < 1e-7
print(summary.optimality, result.objective)
