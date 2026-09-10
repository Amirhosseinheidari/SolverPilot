from __future__ import annotations

import json
from pathlib import Path

from solverpilot import builtin_backend_candidates, default_registry

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "docs/history/frozen-m30/BACKEND-CONTRACT-M30.json").read_text())


def test_backend_contract_covers_every_builtin_candidate_exactly_once():
    groups = CONTRACT["stable_backend_ids"]
    flattened = [name for group in groups.values() for name in group]
    assert len(flattened) == len(set(flattened))
    runtime = [backend.manifest.name for backend in builtin_backend_candidates()]
    assert set(flattened) == set(runtime)


def test_verification_only_contract_matches_backend_metadata():
    verification = set(CONTRACT["stable_backend_ids"]["verification_only"])
    for backend in builtin_backend_candidates():
        is_verification = backend.manifest.metadata.get("verification_only") is True
        assert (backend.manifest.name in verification) is is_verification


def test_default_registry_never_contains_verification_only_backend_ids():
    verification = set(CONTRACT["stable_backend_ids"]["verification_only"])
    assert verification.isdisjoint(default_registry().names())


def test_base_runtime_backend_ids_are_available_with_base_dependencies():
    base = set(CONTRACT["stable_backend_ids"]["base_runtime"])
    available = set(default_registry().names())
    assert base.issubset(available)


def test_learned_lp_performance_routing_is_frozen_off():
    assert CONTRACT["production_baseline_policy"]["learned_lp_performance_routing"] is False
