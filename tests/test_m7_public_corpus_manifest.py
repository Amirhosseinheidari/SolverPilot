from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_m7_public_corpus_manifest_is_auditable_and_results_passed():
    manifest_path = ROOT / "benchmarks" / "public-corpus-m7.json"
    payload = json.loads(manifest_path.read_text())
    assert payload["schema_version"] == "1.0"
    assert len(payload["instances"]) == 3

    for instance in payload["instances"]:
        assert len(instance["input_sha256"]) == 64
        assert instance["source_url"].startswith("https://")
        result_path = ROOT / "benchmarks" / instance["result_file"]
        assert result_path.exists()
        result = json.loads(result_path.read_text())
        assert result["input_sha256"] == instance["input_sha256"]

        if instance["name"] == "AFIRO":
            assert all(row["abs_error"] < 1e-7 for row in result["results"])
            cross = json.loads(
                (ROOT / "benchmarks" / instance["native_crosscheck_file"]).read_text()
            )
            assert cross["canonical_validation_valid"] is True
            assert cross["objective_abs_diff"] < 1e-10
        elif instance["name"] == "p0033":
            assert result["canonical_validation_valid"] is True
            assert result["canonical_published_abs_diff"] < 1e-7
            assert result["canonical_native_abs_diff"] < 1e-10
        elif instance["name"] == "exmip1":
            assert result["canonical_validation_valid"] is True
            assert result["independent_equation_model_success"] is True
            assert result["canonical_vs_equation_abs_diff"] < 1e-10
            assert result["canonical_vs_native_abs_diff"] < 1e-10
