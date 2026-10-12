"""Admission and original-data oracle checks; no corpus download/native solve."""

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from solverpilot import LinearProblem
from solverpilot.runtime.result import OptimalityEvidence

DIRECTORY = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(DIRECTORY))
try:
    SPEC = importlib.util.spec_from_file_location("qualify_public_reuse", DIRECTORY / "qualify_public_reuse.py")
    bench = importlib.util.module_from_spec(SPEC)
    sys.modules[SPEC.name] = bench
    SPEC.loader.exec_module(bench)
finally:
    sys.path.pop(0)


def raw(offset=3):
    return LinearProblem.from_data(A=[[1, 1]], c=[-2, -1], variable_lower=[0, 0],
        variable_upper=[1, 1], constraint_lower=[-np.inf], constraint_upper=[1],
        objective_offset=offset)


@pytest.mark.parametrize("offset", [-3, 0, 3])
@pytest.mark.parametrize("step", bench.STEPS)
def test_affine_variants_preserve_analytic_optimum_and_offset(offset, step):
    original = raw(offset)
    transformed = bench.transformed_problem(original, step)
    alpha, scale, shift = bench.transform_parameters(2, step)
    y = scale * np.array([1., 0.]) + shift
    objective = alpha * scale * (offset - 2)
    assert bench.check_raw(original, step, y, objective, offset - 2)["passed"]
    assert transformed.c @ y + transformed.objective_offset == objective
    assert np.all(transformed.A @ y <= transformed.constraint_upper)
    np.testing.assert_array_equal(transformed.A.toarray(), original.A.toarray())


def test_inverse_checker_does_not_trust_transformed_builder(monkeypatch):
    monkeypatch.setattr(bench, "transform_parameters", lambda *args: (9, 9, np.zeros(2)))
    assert bench.check_raw(raw(), "objective_scale", [1, 0], 2, 1)["passed"]
    for y, obj in ((None, None), ([np.nan, 0], 1), ([2, 0], -1), ([0, 0], 3), ([1, 0], 999)):
        assert not bench.check_raw(raw(), "baseline", y, obj, 1)["passed"]
    feasible = bench.check_raw(raw(), "baseline", [0, 0], 3, 1)
    assert feasible["primal_feasible"] and feasible["objective_consistent"]
    assert not feasible["reference_agrees"]


@pytest.mark.parametrize("left,right", [("gfrd-pnc", "sierra"), ("scagr7", "scfxm3"),
    ("powell", "fffff800"), ("25fv47", "bp1"), ("maros", "maros-r7"),
    ("greenbea", "greenbeb"), ("stocfor1", "stocfor2")])
def test_conservative_aliases(left, right):
    assert bench.conservative_family(left) == bench.conservative_family(right)


def test_prior_uses_downloaded_rejects_but_not_never_seen_candidates(tmp_path):
    payload = {"names": ["AFIRO"], "selected": [{"name": "a", "data_hash": "a" * 64}],
        "exclusions": [{"name": "never-read", "reason": "structural_cap"},
                       {"name": "parse-fail", "reason": "pre_solve_import_failure"}]}
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(payload))
    models = tmp_path / "models"
    models.mkdir()
    (models / "netlib-rejected.mps").write_bytes(b"broken downloaded model")
    (models / "raw-reject.netlib").write_bytes(b"broken compressed model")
    prior = bench.collect_prior([path], [models])
    assert {"afiro", "a", "parse-fail", "rejected", "raw-reject"} <= set(prior["names"])
    assert "never-read" not in prior["names"]
    assert "a" * 64 in prior["hashes"]
    assert bench.sha(b"broken downloaded model") in prior["hashes"]
    assert len(prior["downloaded_models"]) == 2


def case(name, **updates):
    return {"name": name, "family": name, "n": 2, "m": 1, "nnz": 2,
            "reference": 1, **updates}


def test_admission_preserves_failures_and_checks_all_hash_coordinates():
    cases = [case("old"), case("related", family="old"), case("large", n=3001),
             case("rawdup"), case("mpsdup"), case("canonicaldup"), case("parsebad"),
             case("first"), case("samefamily", family="first"), case("second"), case("quota")]
    seen = []

    def loader(c):
        name = c["name"]
        seen.append(name)
        if name == "parsebad":
            raise ValueError("reader mismatch")
        out = {"n": 2, "m": 1, "nnz": 2,
               **{key: bench.sha((name+key).encode()) for key in ("raw_sha256", "mps_sha256", "data_hash")}}
        for name_match, key in (("rawdup", "raw_sha256"), ("mpsdup", "mps_sha256"), ("canonicaldup", "data_hash")):
            if name == name_match:
                out[key] = "x"
        return out

    selected, excluded = bench.admit(cases, {"names": ["old"], "families": ["old"], "hashes": ["x"]}, 2, loader)
    assert [c["name"] for c in selected] == ["first", "second"]
    assert len(excluded) == 9
    assert "quota" not in seen and "samefamily" not in seen
    assert sum(c["reason"] == "pre_solve_import_failure" for c in excluded) == 4


def test_metadata_order_is_outcome_independent():
    text = "A 3 2 4 100 B 1.0E+00\nB 4 5 6 120 BR -2.0D+01\n"
    cases = bench.metadata_cases(text)
    assert cases == bench.metadata_cases("\n".join(reversed(text.splitlines())))
    assert next(c for c in cases if c["name"] == "a")["m"] == 2


def protocol():
    return {"repeats": 1, "presolve": [False, True], "modes": ["cold", "reuse"],
            "steps": list(bench.STEPS), "max_wall_s": 100, "solve_limit_s": 1}


def cohort():
    return {"complete": True, "missing_admissions": 0, "selected": [case("tiny")]}


def rows():
    return [{"name": "tiny", "repeat": 0, "presolve": p, "mode": m, "step": s,
             "status": "optimal", "accepted": True, "checked_s": .01}
            for p in (False, True) for m in ("cold", "reuse") for s in bench.STEPS]


def test_accounting_fails_closed_on_missing_duplicate_unexpected_and_failed_rows():
    assert bench.summarize(rows(), cohort(), protocol())["passed"]
    mutations = [rows()[:-1], rows() + [rows()[0]],
                 rows()[:-1] + [{**rows()[-1], "accepted": False, "status": "limit"}],
                 rows() + [{**rows()[0], "name": "extra"}]]
    for observed in mutations:
        summary = bench.summarize(observed, cohort(), protocol())
        assert not summary["passed"]
        assert summary["failed_or_unobserved"] >= 0
    assert not bench.summarize(rows(), {**cohort(), "complete": False, "missing_admissions": 1}, protocol())["passed"]
    assert not bench.summarize([], {"complete": True, "missing_admissions": 0, "selected": []}, protocol())["passed"]


def test_frozen_protocol_changes_rejected(tmp_path):
    for name in ("cohort.json", "protocol.json"):
        (tmp_path / name).write_text("{}")
    bench.write_json(tmp_path / "freeze.json", {"schema": bench.SCHEMA,
        "files": {name: bench.sha((tmp_path / name).read_bytes()) for name in ("cohort.json", "protocol.json")}})
    (tmp_path / "protocol.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="changed"):
        bench.load_frozen(tmp_path)


@dataclass
class Timings:
    solve_s: float = 0


@pytest.mark.parametrize("fail", [False, True, "cleanup", "budget"])
def test_run_retains_every_outcome_without_native_solves(tmp_path, monkeypatch, fail):
    import solverpilot
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "freeze.json").write_text("{}")
    frozen_protocol = protocol()
    if fail == "budget":
        frozen_protocol["max_wall_s"] = 0
    monkeypatch.setattr(bench, "load_frozen", lambda _: (frozen_protocol, cohort(), [(case("tiny"), raw())]))
    monkeypatch.setattr(bench, "evidence_bundle", lambda *a, **k: {"mock": True})
    monkeypatch.setattr(bench, "verify_evidence_bundle", lambda _: None)
    constructions = []

    class Backend:
        def __init__(self, **kw):
            constructions.append(kw)
            self.reuse = kw["reuse"]

        def close(self):
            if fail == "cleanup":
                raise RuntimeError("cleanup failure")

    def execute(p, b):
        if fail is True:
            raise ValueError("injected solve failure")
        x = np.array([p.variable_upper[0], p.variable_lower[1]])
        return SimpleNamespace(status=SimpleNamespace(value="VALID_OPTIMAL"), backend_status="optimal",
            x=x, objective=float(p.c @ x + p.objective_offset), validation=SimpleNamespace(valid=True),
            optimality_evidence=OptimalityEvidence(True, True, True, True, False),
            trace=SimpleNamespace(timings=Timings(), reuse_applied=b.reuse, reuse_mode="mock"), raw_statistics={})

    monkeypatch.setattr(bench, "HighspyNativeBackend", Backend)
    monkeypatch.setattr(solverpilot, "execute", execute)
    output = tmp_path / "results"
    assert bench.run(SimpleNamespace(prepared=prepared, output=output)) == (2 if fail else 0)
    result = json.loads((output / "result.json").read_text())
    assert result["summary"]["observed"] == 16
    assert result["summary"]["accepted"] == (0 if fail is True or fail == "budget" else 16)
    assert len((output / "rows.jsonl").read_text().splitlines()) == 16
    assert len(constructions) == (0 if fail == "budget" else 10)  # 8 cold + 2 retained.
    assert len(result["teardown_errors"]) == (10 if fail == "cleanup" else 0)
