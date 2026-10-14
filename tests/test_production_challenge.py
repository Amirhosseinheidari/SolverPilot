import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("production_challenge",
    Path(__file__).resolve().parents[1] / "benchmarks/challenge_production_evidence.py")
challenge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(challenge)


def test_seeded_challenge_accepts_controls_rejects_mutations_and_names_blind_spots():
    result = challenge.run_challenge(seed=61, sizes=(3,), scales=(.01, 1000.))
    assert result["summary"]["passed"]
    assert result["summary"]["positive_controls"] > 0
    assert result["summary"]["invalid_cases"] > 0
    assert not result["independent_holdout"] and not result["human_pilot_completed"]
    assert all(row["accepted"] and not row["in_scope"] for row in result["scope_demonstrations"])


def test_challenge_fails_if_formulation_detector_accepts_everything(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(challenge, "check_production_formulation", lambda *a: SimpleNamespace(matches=True))
    result = challenge.run_challenge(seed=62, sizes=(2,), scales=(1.,))
    assert not result["summary"]["passed"]
    assert result["summary"]["false_acceptances"] > 0


def test_challenge_fails_if_evidence_loader_accepts_everything(monkeypatch):
    monkeypatch.setattr(challenge, "verify_evidence_bundle", lambda value: value)
    result = challenge.run_challenge(seed=63, sizes=(2,), scales=(1.,))
    assert not result["summary"]["passed"]
    assert result["summary"]["false_acceptances"] >= 8


def test_challenge_detects_missing_maximum_check_even_with_resource_checks(monkeypatch):
    from dataclasses import replace
    import numpy as np
    original = challenge.check_production_candidate
    def ignores_maximum(contract, quantities, objective, **kwargs):
        return original(replace(contract, maximum=np.full(contract.profit.shape, np.inf)),
                        quantities, objective, **kwargs)
    monkeypatch.setattr(challenge, "check_production_candidate", ignores_maximum)
    result = challenge.run_challenge(seed=64, sizes=(2,), scales=(1.,))
    assert not result["summary"]["passed"]
    failures = [row["name"] for row in result["rows"] if not row["passed"]]
    assert any("isolated_maximum_overrun" in name for name in failures)


def test_errors_cannot_count_as_successful_rejections():
    rows = []
    def broken():
        raise RuntimeError("validator broke")
    challenge.observe(rows, "fixture", "broken", False, broken)
    assert not challenge.summarize(rows)["passed"]
    assert challenge.summarize(rows)["unexpected_errors"] == 1


@pytest.mark.parametrize("options", [{"sizes": ()}, {"sizes": (1,)}, {"scales": (float('nan'),)},
                                    {"seed": True}, {"seed": -1}])
def test_invalid_protocol_rejected(options):
    with pytest.raises(ValueError):
        challenge.run_challenge(**options)
