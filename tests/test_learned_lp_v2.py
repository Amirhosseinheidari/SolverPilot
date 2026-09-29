from dataclasses import FrozenInstanceError, replace
import hashlib
import json

import numpy as np
import pytest
from scipy import sparse

from solverpilot import LinearProblem
from solverpilot.experimental.learned_lp import LPObservation, LPSelector
from solverpilot.experimental.learned_lp_v2 import (
    FEATURE_NAMES_V2, LPLeafGainGuardV2, LPNodeV2, LPRouteDecisionV2, LPSelectorV2,
    calibrate_lp_guard_v2, choose_lp_v2, decide_lp_backend_v2,
    evaluate_lp_selector_v2, fit_lp_selector_v2, lp_features_v2, validate_lp_splits_v2,
)


def observations(split="train", prefix=None, count=12, production=False):
    prefix = split if prefix is None else prefix
    result = []
    for i in range(count):
        samples = {"a": ((1., True),) * 2, "b": (((.1 if i % 2 else 2.), True),) * 2}
        if production:
            samples["production"] = ((2.5, True),) * 2
        result.append(LPObservation(f"{prefix}{i}", f"{prefix}-family{i}",
            hashlib.sha256(f"{prefix}{i}".encode()).hexdigest(), split, "env",
            (float(i % 2),) + (0.,) * (len(FEATURE_NAMES_V2) - 1), samples))
    return result


def fit(rows=None, **kwargs):
    return fit_lp_selector_v2(observations() if rows is None else rows,
        candidates=("a", "b"), cutoff_s=3., protocol_sha256="a" * 64,
        min_leaf_families=2, **kwargs)


def decisions(model, rows, overhead=.001):
    result = {}
    for row in rows:
        candidate, reason, leaf = choose_lp_v2(row.features, model, "env", model.candidates)
        result[row.instance] = LPRouteDecisionV2(candidate, reason, overhead, leaf)
    return result


def evaluate(model, rows, **kwargs):
    options = dict(decisions=decisions(model, rows), min_families=2, bootstrap_draws=100)
    options.update(kwargs)
    return evaluate_lp_selector_v2(model, rows, **options)


def test_sparse_versioned_features_include_finite_log_coefficient_range(monkeypatch):
    p = LinearProblem.from_data(A=sparse.csr_matrix([[1e-300, -1e300, 0.]]), c=[1., 0., 1.],
        variable_lower=[-np.inf, 1., 0.], variable_upper=[np.inf, 1., 2.],
        constraint_lower=[0.], constraint_upper=[np.inf])
    monkeypatch.setattr(sparse.csr_matrix, "toarray", lambda *a, **k: pytest.fail("densification"))
    values = lp_features_v2(p)
    assert len(values) == 13
    assert values[10] == pytest.approx(1 / 3)
    assert values[11] == pytest.approx(1 / 3)
    assert values[12] == pytest.approx(600 * np.log(10))
    assert all(np.isfinite(values))
    with pytest.raises(ValueError, match="continuous"):
        lp_features_v2(object())


def test_empty_matrix_coefficient_range_and_timed_shadow():
    p = LinearProblem.from_data(A=sparse.csr_matrix((0, 2)), c=[1., 0.],
        variable_lower=[0., 0.], variable_upper=[1., 1.],
        constraint_lower=[], constraint_upper=[])
    assert lp_features_v2(p)[12] == 0
    d = decide_lp_backend_v2(p, fit(), environment_id="env", available=("a", "b"))
    assert d.overhead_s >= 0 and d.reason == "outside_training_support" and d.leaf_id is None


def test_family_weighted_tree_is_deterministic_and_frozen():
    model = fit()
    assert model.baseline == "a" and len(model.nodes) == 3
    assert fit(list(reversed(observations()))) == model
    assert choose_lp_v2(observations()[1].features, model, "env", model.candidates)[0] == "b"
    with pytest.raises(FrozenInstanceError):
        model.baseline = "b"
    copied = replace(model, candidates=list(model.candidates), nodes=list(model.nodes), lower=list(model.lower))
    assert isinstance(copied.candidates, tuple) and isinstance(copied.nodes, tuple)
    assert isinstance(copied.lower, tuple)


def test_family_size_cannot_dominate_training_baseline():
    rows = observations(count=53)
    rows = [replace(r, group="large" if i < 50 else f"small{i}", features=(0.,) * 13,
        samples={"a": ((10. if i < 50 else 1., True),) * 2,
                 "b": ((1. if i < 50 else 10., True),) * 2}) for i, r in enumerate(rows)]
    model = fit_lp_selector_v2(rows, candidates=("a", "b"), cutoff_s=20.,
                              protocol_sha256="b" * 64, max_depth=0)
    assert model.baseline == "a"


def test_depth_two_finds_second_split_without_exceeding_bound():
    rows = observations(count=16)
    rows = [replace(r, features=(float(i // 4),) + (0.,) * 12,
        samples={"a": ((.1 if i // 4 in (0, 3) else 2., True),) * 2,
                 "b": ((2. if i // 4 in (0, 3) else .1, True),) * 2})
            for i, r in enumerate(rows)]
    model = fit(rows)
    assert len(model.nodes) == 5
    assert [choose_lp_v2(r.features, model, "env", model.candidates)[0] for r in rows] == [
        "a" if i // 4 in (0, 3) else "b" for i in range(16)]


@pytest.mark.parametrize("field", ["instance", "group", "data_hash"])
def test_train_calibration_test_leakage_and_guard_calibration_rejected(field):
    train, cal, test = observations(), observations("calibration", production=True), observations("test", production=True)
    validate_lp_splits_v2(train, cal, test)
    cal[0] = replace(cal[0], **{field: getattr(train[0], field)})
    with pytest.raises(ValueError, match="leakage"):
        validate_lp_splits_v2(train, cal, test)
    with pytest.raises(ValueError, match="leakage"):
        calibrate_lp_guard_v2(fit(train), cal, min_families=2)


def test_seed_variants_stay_with_their_family():
    train, cal, test = observations(), observations("calibration"), observations("test")
    test[0] = replace(test[0], group=train[0].group)
    with pytest.raises(ValueError, match="family"):
        validate_lp_splits_v2(train, cal, test)


def test_calibration_certifies_specific_leaves_and_never_missing_leaf():
    model = fit()
    rows = observations("calibration", production=True)
    guard = calibrate_lp_guard_v2(model, rows, overhead_limit_s=.02, min_families=2)
    assert len(guard.certified_leaves) == 2
    for leaf in guard.certified_leaves:
        assert guard.permits_bound_candidate(leaf.candidate, leaf.leaf_id)
        assert guard.permits(model, leaf.candidate, leaf_id=leaf.leaf_id, elapsed_s=.001, cutoff_s=3.)
        assert not guard.permits(model, leaf.candidate, elapsed_s=.001, cutoff_s=3.)
        assert not guard.permits(model, leaf.candidate, leaf_id=leaf.leaf_id, elapsed_s=.03, cutoff_s=3.)
    assert not guard.permits_bound_candidate("b", True)
    assert tuple(name for name, _ in guard.candidate_gains) == ("a", "b")


@pytest.mark.parametrize("kind", ["failure", "late", "loss", "too_few"])
def test_bad_calibration_blocks_only_affected_leaf(kind):
    model = fit()
    rows = observations("calibration", production=True)
    bad_leaf = choose_lp_v2(rows[1].features, model, "env", model.candidates)[2]
    if kind == "too_few":
        rows = rows[:2] + rows[2::2]
    else:
        samples = dict(rows[1].samples)
        samples["b"] = {"failure": ((.01, False), (.01, True)),
                        "late": ((2.999, True),) * 2, "loss": ((2.7, True),) * 2}[kind]
        rows[1] = replace(rows[1], samples=samples)
    guard = calibrate_lp_guard_v2(model, rows, overhead_limit_s=.01, min_families=2)
    assert not guard.permits_bound_candidate("b", bad_leaf)
    assert any(leaf.candidate == "a" for leaf in guard.certified_leaves)


def test_cold_and_fixed_amortization_charge_setup_before_cutoff():
    model = fit()
    rows = observations("test", production=True)
    report = evaluate(model, rows, session_setup_s=3., reuse_count=100)
    assert report["scopes"]["cold"]["verified_repeats"] == 0
    assert report["scopes"]["cold"]["policy_family_mean_par10_s"] == 30
    assert report["scopes"]["amortized"]["verified_repeats"] == 24
    assert report["scopes"]["amortized"]["session_setup_charge_s"] == .03
    assert not report["production_authorized"] and not report["automatic_production_routing_enabled"]


def test_routing_overhead_crossing_deadline_loses_success_and_pays_par10():
    model = fit()
    rows = observations("test", production=True)
    rows = [replace(r, samples={**r.samples, "b": ((2.999, True),) * 2}) for r in rows]
    report = evaluate(model, rows, decisions=decisions(model, rows, .002))
    cold = report["scopes"]["cold"]
    assert cold["verified_repeats"] == 12
    assert not cold["comparisons"]["production"]["gates"]["no_lost_verified_solves"]
    assert cold["policy_family_mean_par10_s"] == pytest.approx((1.002 + 30) / 2)


def test_guard_fallback_still_pays_routing_and_setup_costs():
    model = fit()
    guard = calibrate_lp_guard_v2(model, observations("calibration", production=True),
                                  overhead_limit_s=.01, min_families=2)
    rows = observations("test", production=True)
    report = evaluate(model, rows, guard=guard, decisions=decisions(model, rows, .02), session_setup_s=.1)
    assert report["routed_candidates"] == {"production": 12}
    assert report["scopes"]["cold"]["policy_family_mean_par10_s"] == pytest.approx(2.62)


def test_out_of_support_fallback_still_pays_measured_work():
    model = fit()
    rows = [replace(r, features=(2.,) + r.features[1:])
            for r in observations("test", production=True)]
    report = evaluate(model, rows, session_setup_s=.1)
    assert report["routed_candidates"] == {"production": 12}
    assert report["scopes"]["cold"]["policy_family_mean_par10_s"] == pytest.approx(2.601)


def test_independent_baseline_setup_and_default_coverage_are_accounted():
    model = fit()
    rows = [replace(r, samples={**r.samples, "default": ((.5, True),) * 2})
            for r in observations("test", production=True)]
    report = evaluate(model, rows, baselines=("production", "default"),
                      baseline_setup_s={"default": 3.}, reuse_count=100)
    assert report["scopes"]["cold"]["comparisons"]["default"]["baseline_verified_repeats"] == 0
    assert report["scopes"]["amortized"]["comparisons"]["default"]["baseline_verified_repeats"] == 24


def test_failed_repeats_remain_in_family_evaluation():
    model = fit()
    rows = observations("test", production=True)
    rows[1] = replace(rows[1], samples={**rows[1].samples, "b": ((.1, False), (.1, True))})
    report = evaluate(model, rows)
    comparison = report["scopes"]["cold"]["comparisons"]["production"]
    assert not comparison["gates"]["no_lost_verified_solves"]
    assert report["scopes"]["cold"]["verified_repeats"] == 23


@pytest.mark.parametrize("field", ["instance", "group", "data_hash"])
def test_calibration_identities_are_excluded_from_evaluation(field):
    model = fit()
    cal = observations("calibration", production=True)
    guard = calibrate_lp_guard_v2(model, cal, min_families=2)
    rows = observations("test", production=True)
    rows[0] = replace(rows[0], **{field: getattr(cal[0], field)})
    with pytest.raises(ValueError, match="leakage"):
        evaluate(model, rows, guard=guard)


def test_model_guard_roundtrip_tampering_and_v1_schema_rejection(tmp_path):
    model = fit()
    guard = calibrate_lp_guard_v2(model, observations("calibration", production=True), min_families=2)
    for obj, loader, filename in ((model, LPSelectorV2.load, "model.json"),
                                  (guard, LPLeafGainGuardV2.load, "guard.json")):
        path = tmp_path / filename
        obj.save(path)
        assert loader(path) == obj
        payload = json.loads(path.read_text())
        payload["cutoff_s"] = 30.
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="digest"):
            loader(path)
    model.save(tmp_path / "v2.json")
    with pytest.raises(ValueError, match="schema"):
        LPSelector.load(tmp_path / "v2.json")


@pytest.mark.parametrize("change", [
    {"split": "test"}, {"features": (0.,) * 8}, {"features": (float("nan"),) * 13},
    {"features": (True,) + (0.,) * 12}, {"samples": {"a": ((1., True),) * 2}},
    {"samples": {"a": ((1., True),), "b": ((1., True),) * 2}},
    {"samples": {"a": ((-1., True),) * 2, "b": ((1., True),) * 2}},
    {"samples": {"a": ((1., 1),) * 2, "b": ((1., True),) * 2}},
])
def test_malformed_observations_rejected(change):
    rows = observations()
    rows[0] = replace(rows[0], **change)
    with pytest.raises(ValueError):
        fit(rows)


def test_decisions_cannot_be_missing_or_reassigned_after_outcomes():
    model, rows = fit(), observations("test", production=True)
    with pytest.raises(ValueError, match="coverage"):
        evaluate(model, rows, decisions={})
    chosen = decisions(model, rows)
    chosen[rows[1].instance] = replace(chosen[rows[1].instance], candidate="a")
    with pytest.raises(ValueError, match="frozen"):
        evaluate(model, rows, decisions=chosen)


@pytest.mark.parametrize("options", [{"reuse_count": 0}, {"reuse_count": True},
    {"session_setup_s": -1}, {"session_setup_s": float("nan")},
    {"baseline_setup_s": {"missing": 0}}, {"bootstrap_draws": 1}])
def test_invalid_evaluation_setup_and_protocol_values_rejected(options):
    with pytest.raises(ValueError):
        evaluate(fit(), observations("test", production=True), **options)


def test_invalid_tree_topology_rejected():
    model = fit()
    for nodes in ((LPNodeV2(0, .5, 0, 0, "a"),),
                  (LPNodeV2(-1, 0., -1, -1, "a"), LPNodeV2(-1, 0., -1, -1, "b"))):
        with pytest.raises(ValueError):
            replace(model, nodes=nodes)


def test_abstention_has_no_leaf_authority():
    model = fit()
    features = observations()[1].features
    assert choose_lp_v2(features, model, "changed", model.candidates)[1:] == ("environment_mismatch", None)
    assert choose_lp_v2((2.,) + features[1:], model, "env", model.candidates)[1:] == ("outside_training_support", None)
    assert choose_lp_v2(features, model, "env", ("a",))[1:] == ("candidate_unavailable", None)
