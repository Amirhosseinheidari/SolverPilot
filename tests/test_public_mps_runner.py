from pathlib import Path
import importlib.util

import pytest

# Benchmark scripts are source files rather than installed package modules.
SPEC = importlib.util.spec_from_file_location(
    "m5_public_mps_runner", Path(__file__).parents[1] / "benchmarks" / "m5_public_mps_runner.py"
)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_manifest_runner_never_silently_drops_missing_or_bad_instances(tmp_path: Path):
    good = """NAME A
ROWS
 N OBJ
 G C
COLUMNS
 X OBJ 1 C 1
RHS
 R C 2
ENDATA
"""
    (tmp_path / "a.mps").write_text(good, encoding="ascii")
    (tmp_path / "bad.mps").write_text("not mps", encoding="ascii")

    payload = MOD.run_manifest(
        manifest_text="a.mps\nmissing.mps\nbad.mps\n",
        solu_text="=opt= a 2\n=unkn= missing\n=unkn= bad\n",
        data_dir=tmp_path,
        backend="scipy-highs-bridge",
        time_limit_s=None,
    )
    assert payload["instances_in_manifest"] == 3
    assert payload["state_counts"] == {"solved": 1, "missing": 1, "parse_error": 1}
    assert payload["reference_checks"] == 1
    assert payload["reference_matches"] == 1
    assert payload["reference_mismatches"] == 0
    assert payload["rows"][0]["reference_check"] == "objective_matches_optimum"


def test_reference_checks_best_known_and_infeasible(tmp_path: Path):
    max_model = """NAME MAXI
OBJSENSE
 MAX
ROWS
 N OBJ
 L C
COLUMNS
 X OBJ 1 C 1
RHS
 R C 3
ENDATA
"""
    infeasible = """NAME INF
ROWS
 N OBJ
 G LO
 L HI
COLUMNS
 X OBJ 1 LO 1
 X HI 1
RHS
 R LO 2 HI 1
ENDATA
"""
    (tmp_path / "maxi.mps").write_text(max_model, encoding="ascii")
    (tmp_path / "inf.mps").write_text(infeasible, encoding="ascii")
    payload = MOD.run_manifest(
        manifest_text="maxi.mps\ninf.mps\n",
        solu_text="=best= maxi 2.5\n=inf= inf\n",
        data_dir=tmp_path,
        backend="scipy-highs-bridge",
        time_limit_s=None,
    )
    checks = {r["instance"]: r["reference_check"] for r in payload["rows"]}
    assert checks == {"maxi": "meets_or_beats_best_known", "inf": "infeasible_matches"}
    assert payload["reference_mismatches"] == 0



def test_missing_solution_reference_check_is_inconclusive():
    from types import SimpleNamespace
    from solverpilot import ObjectiveSense, PublicStatus
    from solverpilot.evaluation import ObjectiveReference

    result = SimpleNamespace(status=PublicStatus.ERROR, objective=None)
    problem = SimpleNamespace(objective_sense=ObjectiveSense.MINIMIZE)
    ref = ObjectiveReference(name="x", status="optimal", objective=1.0)
    assert MOD._reference_check(result=result, problem=problem, reference=ref, atol=1e-6, rtol=1e-6) == "missing_solution"
