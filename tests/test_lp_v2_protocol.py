"""Protocol integrity/accounting tests; no real backend is executed."""
import importlib.util
import json
from pathlib import Path

import pytest


_SPEC = importlib.util.spec_from_file_location("qualify_lp_v2",
    Path(__file__).resolve().parents[1]/"benchmarks"/"qualify_lp_v2.py")
protocol = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(protocol)


@pytest.mark.parametrize('split', protocol.SPLITS)
def test_paired_repeats_reverse_order_without_rotating_again(split):
    for index in range(5):
        first = protocol.strategy_order(split, index, 0)
        assert protocol.strategy_order(split, index, 1) == list(reversed(first))
        assert protocol.strategy_order(split, index, 2) == first


@pytest.fixture
def demo(tmp_path):
    path = protocol.generate_development_demo(tmp_path/"corpus")
    return path, json.loads(path.read_text())


def test_demo_has_real_distinct_template_families_and_preregistered_call_cap(demo):
    path, manifest = demo
    config, cases, problems, consumed, calls = protocol.validate_manifest(manifest, path.parent)
    assert [len(cases[split]) for split in protocol.SPLITS] == [4, 2, 2]
    assert config["repeats"] == 2 and config["reuse_count"] == 20
    assert calls == 52 and consumed == {}
    assert len({case["family"] for split in cases.values() for case in split}) == 8
    assert len({problem.structural_hash for problem in problems.values()}) == 8
    assert protocol.strategies("test") == ("v2", *protocol.CANDIDATES, "production", "default")


@pytest.mark.parametrize("key", protocol.IDENTITIES)
def test_all_cross_split_identity_overlaps_rejected_before_parsing(demo, key):
    path, manifest = demo
    manifest["splits"]["test"][0][key] = manifest["splits"]["train"][0][key]
    def never_load(*args):
        pytest.fail("identity overlap must be found before input parsing or execution")
    with pytest.raises(ValueError, match="overlap"):
        protocol.validate_manifest(manifest, path.parent, loader=never_load)


def test_family_and_name_case_whitespace_aliases_cannot_cross_splits(demo):
    path, manifest = demo
    manifest["splits"]["calibration"][0]["family"] = "  " + manifest["splits"]["train"][0]["family"].upper() + "  "
    with pytest.raises(ValueError, match="family"):
        protocol.validate_manifest(manifest, path.parent)


@pytest.mark.parametrize("key", protocol.IDENTITIES)
def test_consumed_identity_protects_calibration_and_test(demo, key):
    path, manifest = demo
    value = manifest["splits"]["calibration"][0][key]
    manifest["consumed"] = {protocol.CONSUMED_KEYS[key]: [value]}
    with pytest.raises(ValueError, match="consumed"):
        protocol.validate_manifest(manifest, path.parent)


def test_consumed_training_is_explicitly_allowed_and_external_list_is_hashed(demo):
    path, manifest = demo
    consumed_path = path.parent/"prior.json"
    protocol.write_json(consumed_path, {"names": [manifest["splits"]["train"][0]["name"]]})
    manifest["consumed_lists"] = [consumed_path.name]
    _, _, _, hashes, _ = protocol.validate_manifest(manifest, path.parent)
    assert hashes == {str(consumed_path): protocol.file_sha(consumed_path)}


@pytest.mark.parametrize("change,match", [
    ({"repeats": 1}, "repeats"), ({"reuse_count": 0}, "reuse_count"),
    ({"cutoff_s": float("nan")}, "cutoff_s"), ({"max_depth": True}, "max_depth"),
    ({"unregistered_option": 3}, "unknown"),
])
def test_invalid_protocol_options_rejected_before_runs(demo, change, match):
    path, manifest = demo
    manifest["protocol"].update(change)
    with pytest.raises(ValueError, match=match):
        protocol.validate_manifest(manifest, path.parent)


def test_total_api_call_limit_counts_every_repeat_candidate_and_baseline(demo):
    path, manifest = demo
    with pytest.raises(ValueError, match="52 API calls"):
        protocol.validate_manifest(manifest, path.parent, max_calls=51)


@pytest.mark.parametrize("identity", ["raw_sha256", "data_hash"])
def test_manifest_hashes_are_checked_against_real_bytes_and_parsed_models(demo, identity):
    path, manifest = demo
    manifest["splits"]["test"][0][identity] = "f"*64
    with pytest.raises(ValueError, match="hash mismatch|SHA256 mismatch"):
        protocol.validate_manifest(manifest, path.parent)


def test_cold_and_amortized_cost_recompute_lateness_without_discarding_verified_rows():
    row = dict(name="case", family="family", strategy="v2", verified=True,
               api_wall_s=.9, preparation_s=.2)
    cold = protocol.score_outcome(row, cutoff_s=1., reuse_count=10, scope="cold")
    warm = protocol.score_outcome(row, cutoff_s=1., reuse_count=10, scope="amortized")
    assert cold == {"charged_s": 1.1, "success": False, "par10_s": 10.}
    assert warm["success"] and warm["charged_s"] == pytest.approx(.92)
    summary = protocol.summarize([row, {**row, "verified": False}], cutoff_s=1., reuse_count=10)
    assert summary["outcome_count"] == 2
    assert summary["cost_scopes"]["cold"]["v2"]["failed_or_late"] == 2
    assert summary["cost_scopes"]["amortized"]["v2"]["mean_par10_s"] == pytest.approx(5.46)
    assert summary["production_promotion_supported"] is False


@pytest.mark.parametrize("wall,verified", [(float("nan"), True), (-1., True), (2., True), (.1, False)])
def test_failed_invalid_or_late_results_always_pay_par10(wall, verified):
    result = protocol.score_outcome({"api_wall_s": wall, "verified": verified},
        cutoff_s=1., reuse_count=10, scope="cold")
    assert not result["success"] and result["par10_s"] == 10.


def test_duplicate_and_missing_outcomes_cannot_pass_accounting():
    cases = [{"name": "case"}]
    rows = [{"name": "case", "strategy": strategy, "repeat": repeat}
            for strategy in protocol.strategies("test") for repeat in range(2)]
    protocol.validate_outcome_coverage(rows, cases, 2, "test")
    for invalid in (rows[:-1], rows+[rows[0]]):
        with pytest.raises(ValueError, match="discarded"):
            protocol.validate_outcome_coverage(invalid, cases, 2, "test")


def test_disagreeing_verified_objectives_remain_visible_and_cannot_score_success():
    rows = [dict(name="case", strategy=s, objective=value, verified=True, api_wall_s=.1)
            for s, value in [("production", 1.), ("v2", 2.)]]
    protocol.check_objectives(rows, [{"name": "case"}])
    assert all(row["verified"] and not row["objective_consistent"] for row in rows)
    assert all(not protocol.score_outcome(row, cutoff_s=1., reuse_count=2, scope="cold")["success"] for row in rows)


def test_frozen_artifact_and_implementation_changes_are_rejected(tmp_path, monkeypatch):
    path = tmp_path/"model.json"
    path.write_text("model")
    expected = {str(path): protocol.file_sha(path)}
    monkeypatch.setattr(protocol, "source_hashes", lambda: {"source.py": "old"})
    protocol.verify_frozen({"source.py": "old"}, expected)
    path.write_text("changed model")
    with pytest.raises(ValueError, match="artifact changed"):
        protocol.verify_frozen({"source.py": "old"}, expected)
    with pytest.raises(ValueError, match="implementation changed"):
        protocol.verify_frozen({"source.py": "new"}, {})


class FakeArtifact:
    def __init__(self, value):
        self.value = value

    def save(self, path):
        protocol.write_json(path, self.value)


class FakeRuntime:
    environment = {"fake": True}
    environment_id = "fake-environment"

    def __init__(self, output, *, corrupt_model=False):
        self.output = output
        self.events = []
        self.corrupt_model = corrupt_model

    def features(self, problem):
        return (0.,)*13

    def fit(self, rows, config, protocol_hash):
        assert all(row.split == "train" for row in rows)
        assert len(rows) == 4 and len(self.events) == 16
        assert all(set(row.samples) == set(protocol.CANDIDATES) for row in rows)
        self.events.append("fit")
        return FakeArtifact({"training_names": [row.instance for row in rows]})

    def calibrate(self, model, rows, config):
        assert (self.output/"model-freeze.json").exists()
        assert all(row.split == "calibration" for row in rows)
        assert len(rows) == 2 and len(self.events) == 33
        assert all(set(row.samples) == {*protocol.CANDIDATES, "production"} for row in rows)
        self.events.append("calibrate")
        if self.corrupt_model:
            (self.output/"model.json").write_text("modified after model freeze")
        return FakeArtifact({"calibration_names": [row.instance for row in rows]})

    def prepare(self, model_path, guard_path, config):
        assert (self.output/"guard-freeze.json").exists()
        self.events.append("prepare")
        return object()

    def execute(self, problem, strategy, config, prepared):
        assert (self.output/"protocol.json").exists()
        self.events.append(strategy)
        if len(self.events) == 2:
            raise RuntimeError("deliberate fake backend failure")
        if strategy == "v2":
            assert prepared is not None and "calibrate" in self.events
        return {"status": "fake", "objective": 0., "verified": True}


def test_full_fake_workflow_freezes_before_separate_calibration_and_single_test(demo, tmp_path, monkeypatch):
    path, _ = demo
    output = tmp_path/"run"
    runtime = FakeRuntime(output)
    monkeypatch.setattr(protocol, "source_hashes", lambda: {"fake.py": "fixed"})
    report = protocol.run_protocol(path, output, runtime=runtime)
    assert report["total_api_calls"] == 52 and report["test_evaluations"] == 1
    assert report["automatic_production_routing_enabled"] is False
    train = json.loads((output/"train-outcomes.json").read_text())
    test = json.loads((output/"test-outcomes.json").read_text())
    assert len(train) == 16 and sum(row["status"] == "exception" for row in train) == 1
    assert len(test) == 20 and sum(row["strategy"] == "v2" for row in test) == 4
    assert {row["strategy"] for row in test} == {"v2", *protocol.CANDIDATES, *protocol.BASELINES}
    assert all(row["preparation_s"] >= 0 for row in test)
    with pytest.raises(FileExistsError):
        protocol.run_protocol(path, output, runtime=runtime)


def test_modified_model_after_calibration_prevents_all_test_solves(demo, tmp_path, monkeypatch):
    path, _ = demo
    output = tmp_path/"run"
    runtime = FakeRuntime(output, corrupt_model=True)
    monkeypatch.setattr(protocol, "source_hashes", lambda: {"fake.py": "fixed"})
    with pytest.raises(ValueError, match="artifact changed"):
        protocol.run_protocol(path, output, runtime=runtime)
    assert "prepare" not in runtime.events and "v2" not in runtime.events
    assert not (output/"test-outcomes.json").exists()
