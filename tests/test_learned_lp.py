from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from solverpilot import LinearProblem, ProductionEvidence, EvidenceClass, plan_production_solve
from solverpilot.experimental.learned_lp import (
    LPObservation, LPSelector, LPRouteDecision, fit_lp_selector, lp_features,
    decide_lp_backend, evaluate_lp_selector, _choose,
)
from solverpilot.benchmark.splits import SplitRecord, validate_split_records


def rows(split="train", prefix="train"):
    return [LPObservation(f"{prefix}{i}", f"{prefix}g{i}",
            hashlib.sha256(f"{prefix}{i}".encode()).hexdigest(), split, "env",
            (float(i % 2),) + (0.0,) * 7,
            {"a": ((1., True), (1., True)),
             "b": (((.1 if i % 2 else 2.), True),) * 2}) for i in range(12)]


def fit(data=None):
    return fit_lp_selector(rows() if data is None else data, candidates=("a", "b"),
                           cutoff_s=3, protocol_sha256="a" * 64)


def decisions(model, data, overhead=0.001):
    return {r.instance: LPRouteDecision(_choose(r.features, model, "env", {"a", "b"})[0],
                                       "test", overhead) for r in data}


def test_learns_cost_sensitive_switch_and_training_baseline():
    m = fit()
    assert m.baseline == "a"
    assert m.left == "a" and m.right == "b"
    assert m.feature_index == 0 and m.threshold == .5
    heldout = rows("validation", "val")
    report = evaluate_lp_selector(m, heldout, decisions=decisions(m, heldout))
    assert report["policy_to_baseline"] < .56
    assert report["research_gate_passed"]
    assert not report["production_authorized"]


def test_all_failed_repeats_are_penalized_not_dropped():
    data = [replace(r, samples={"a": ((1., True),) * 2,
                               "b": ((.01, True), (.01, False))}) for r in rows()]
    assert fit(data).baseline == "a"


def test_verified_over_cutoff_is_still_penalized():
    data = [replace(r, samples={"a": ((4., True),) * 2,
                               "b": ((2., True),) * 2}) for r in rows()]
    assert fit(data).baseline == "b"


@pytest.mark.parametrize("split", ["validation", "test", "typo"])
def test_fit_rejects_other_splits(split):
    with pytest.raises(ValueError):
        fit(rows(split))


@pytest.mark.parametrize("change", [
    {"environment_id": "another"}, {"features": (float("nan"),) * 8},
    {"samples": {"a": ((1., True),)}}, {"samples": {"a": ((1., True),), "b": ((1., True),) * 2}},
    {"samples": {"a": ((-1., True),) * 2, "b": ((1., True),) * 2}},
    {"data_hash": "bad"}, {"group": ""},
])
def test_malformed_observations_rejected(change):
    data = rows()
    data[0] = replace(data[0], **change)
    with pytest.raises(ValueError):
        fit(data)


def test_duplicate_data_under_new_name_rejected():
    data = rows()
    data[1] = replace(data[1], data_hash=data[0].data_hash)
    with pytest.raises(ValueError, match="duplicate"):
        fit(data)


@pytest.mark.parametrize("field", ["instance", "group", "data_hash"])
def test_heldout_leakage_rejected(field):
    m = fit()
    data = rows("test", "held")
    data[0] = replace(data[0], **{field: getattr(rows()[0], field)})
    with pytest.raises(ValueError, match="leakage"):
        evaluate_lp_selector(m, data, decisions=decisions(m, data))


def test_model_json_roundtrip_and_tamper(tmp_path):
    m = fit()
    path = tmp_path / "selector.json"
    m.save(path)
    assert LPSelector.load(path) == m
    p = json.loads(path.read_text())
    p["threshold"] = 200
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError, match="digest"):
        LPSelector.load(path)


def test_abstention_on_environment_support_availability():
    m = fit()
    assert _choose((1.,) + (0.,) * 7, m, "other", {"a", "b"}) == ("a", "environment_mismatch")
    assert _choose((2.,) + (0.,) * 7, m, "env", {"a", "b"}) == ("a", "outside_training_support")
    assert _choose((1.,) + (0.,) * 7, m, "env", {"a"}) == ("a", "candidate_unavailable")
    assert _choose((2.,) + (0.,) * 7, m, "env", {"b"})[0] is None


def test_real_sparse_features_and_timed_shadow():
    p = LinearProblem.from_data(A=[[1., 1.]], c=[1., 2.], variable_lower=[0., 0.],
                               variable_upper=[1., 1.], constraint_lower=[1.], constraint_upper=[np.inf])
    features = lp_features(p)
    assert len(features) == 8 and features[3] == 1
    d = decide_lp_backend(p, fit(), environment_id="env", available=["a", "b"])
    assert d.candidate == "a" and d.overhead_s >= 0
    with pytest.raises(ValueError, match="continuous"):
        lp_features(object())


def test_overhead_can_veto_apparent_gain():
    m = fit()
    data = rows("validation", "val")
    r = evaluate_lp_selector(m, data, decisions=decisions(m, data, overhead=1))
    assert not r["research_gate_passed"]


def test_missing_or_posthoc_decision_rejected():
    m = fit()
    data = rows("test", "held")
    with pytest.raises(ValueError, match="coverage"):
        evaluate_lp_selector(m, data, decisions={})
    ds = decisions(m, data)
    ds[data[1].instance] = LPRouteDecision("a", "changed", 0)
    with pytest.raises(ValueError, match="frozen"):
        evaluate_lp_selector(m, data, decisions=ds)


def evidence():
    return ProductionEvidence(EvidenceClass.COMPARATIVE_HELDOUT, "test", True, True, True, True,
                              ("scipy-highs-ds", "scipy-highs-ipm"), True, True, True, True,
                              True, True, True)


@pytest.mark.parametrize("field", ["corpus_integrity_passed", "outcome_accounting_passed",
                                   "independent_validation_passed", "reference_crosscheck_passed"])
def test_production_performance_requires_correctness_evidence(field):
    assert evidence().supports_performance_ranking
    assert not replace(evidence(), **{field: False}).supports_performance_ranking


def test_production_override_must_be_benchmarked():
    from solverpilot.runtime import default_registry
    p = LinearProblem.from_data(A=[[1.]], c=[1.], variable_lower=[0.], variable_upper=[1.],
                               constraint_lower=[0.], constraint_upper=[1.])
    d = plan_production_solve(p, default_registry(), evidence=evidence(),
                             performance_override={"lp": "scipy-highs-bridge"})
    assert "absent from comparative evidence" in d.rejected_performance_override_reason
    assert not d.auto_performance_ranking_enabled


def test_invalid_direct_split_record_is_rejected():
    assert not validate_split_records([SplitRecord("i", "typo", "g")]).ok


def test_unused_evidence_does_not_report_executed_performance_ranking():
    from solverpilot.runtime import default_registry
    p = LinearProblem.from_data(A=[[1.]], c=[1.], variable_lower=[0.], variable_upper=[1.],
                               constraint_lower=[0.], constraint_upper=[1.])
    d = plan_production_solve(p, default_registry(), evidence=evidence())
    assert not d.auto_performance_ranking_enabled


def test_train_order_does_not_change_model():
    assert fit(list(reversed(rows()))) == fit()


@pytest.mark.parametrize("kwargs", [{"cutoff_s": float("inf")}, {"min_leaf_groups": 0},
                                   {"switch_margin_s": -1}, {"candidates": ("a", "a")}])
def test_invalid_training_parameters(kwargs):
    options = dict(candidates=("a", "b"), cutoff_s=3, protocol_sha256="a" * 64)
    options.update(kwargs)
    with pytest.raises(ValueError):
        fit_lp_selector(rows(), **options)


def test_lost_verified_repeat_rejects_promotion():
    m = fit()
    data = rows("test", "held")
    data[1] = replace(data[1], samples={"a": ((1., True),) * 2,
                                      "b": ((.1, True), (.1, False))})
    report = evaluate_lp_selector(m, data, decisions=decisions(m, data))
    assert not report["gates"]["no_lost_verified_solves"]
    assert not report["research_gate_passed"]


def test_insufficient_group_support_abstains():
    m = fit([replace(r, group="one-group") for r in rows()])
    assert m.feature_index == -1
