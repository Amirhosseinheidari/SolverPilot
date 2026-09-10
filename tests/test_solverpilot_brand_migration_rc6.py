from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_rc6_brand_migration_is_preserved_as_historical_evidence():
    frozen = ROOT / "docs/history/frozen-rc6"
    assert (frozen / "pyproject-rc6.toml").is_file()
    api = json.loads((frozen / "PUBLIC-API-RC6.json").read_text())
    backend = json.loads((frozen / "BACKEND-CONTRACT-RC6.json").read_text())
    assert api["package_version"] == "0.0.36rc6"
    assert api["schema"] == "solverpilot.public_api.v1"
    assert backend["package_version"] == "0.0.36rc6"
    assert (frozen / "SOLVERPILOT-BRAND-MIGRATION-RC6.md").is_file()


def test_rc7_deliberately_removes_the_prepublic_legacy_namespace():
    assert not (ROOT / "src/optimind").exists()
    assert (ROOT / "src/solverpilot").is_dir()
