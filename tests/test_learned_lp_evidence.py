"""Replay committed measurements; no new timing, native dependency or tuning."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from solverpilot.experimental.learned_lp import (
    LPObservation, LPRouteDecision, LPSelector, fit_lp_selector, evaluate_lp_selector,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/learned-lp-local"


def read(name):
    return json.loads((EVIDENCE / name).read_text())


def observations(split):
    return [LPObservation(**r) for r in read(f"{split}.json")]


def test_captured_file_integrity_and_protocol_binding():
    for name, digest in read("manifest.json")["files"].items():
        assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == digest
    model = LPSelector.load(EVIDENCE / "model.json")
    assert model.protocol_sha256 == hashlib.sha256((EVIDENCE / "protocol.json").read_bytes()).hexdigest()


def test_model_is_reproducible_from_training_only():
    saved = LPSelector.load(EVIDENCE / "model.json")
    fitted = fit_lp_selector(observations("train"), candidates=saved.candidates,
                             cutoff_s=saved.cutoff_s, protocol_sha256=saved.protocol_sha256)
    assert asdict(fitted) == asdict(saved)


@pytest.mark.parametrize("split", ["validation", "test"])
def test_reports_replay_without_native_solves(split):
    model = LPSelector.load(EVIDENCE / "model.json")
    choices = {k: LPRouteDecision(**v) for k, v in read(f"{split}-decisions.json").items()}
    replay = evaluate_lp_selector(model, observations(split), decisions=choices)
    saved = read(f"{split}-report.json")
    # Preserve the frozen historical artifact. Its cohorts have no deadline
    # crossing, so the corrected per-repeat accounting changes only the label.
    assert replay.pop('cost_definition') == (
        'add measured decision cost to each repeat before cutoff; unsuccessful/late = 10*cutoff')
    assert saved.pop('cost_definition') == (
        'mean of all repeats; unsuccessful/late repeat = 10*cutoff; plus measured decision cost')
    assert replay == saved
    assert saved["research_gate_passed"] is True
    assert saved["production_authorized"] is False


def test_all_declared_outcomes_and_split_groups_are_accounted_for():
    protocol = read("protocol.json")
    rows = read("observations.json")
    cases = protocol["cases"]
    assert len(cases) == len({r["data_hash"] for r in cases}) == 60
    actual = {(r["instance"], r["candidate"], r["repeat"]) for r in rows}
    expected = {(r["instance"], c, rep) for r in cases for c in protocol["candidates"]
                for rep in range(protocol["repeats"])}
    assert len(rows) == len(actual) == 360 and actual == expected
    groups = [{r["group"] for r in cases if r["split"] == split}
              for split in ("train", "validation", "test")]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    for split in ("train", "validation", "test"):
        for observation in observations(split):
            for candidate, samples in observation.samples.items():
                raw = sorted((r for r in rows if r["instance"] == observation.instance
                              and r["candidate"] == candidate), key=lambda r: r["repeat"])
                assert samples == [[r["wall_s"], r["verified"]] for r in raw]
    assert read("summary.json")["test_consumed"] is True
