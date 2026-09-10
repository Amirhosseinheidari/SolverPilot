from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .model import DatasetSpec


MIPLIB2017_BENCHMARK_V2 = DatasetSpec(
    name="miplib2017-benchmark-v2",
    kind="miplib2017-mps",
    archive_url="https://miplib.zib.de/downloads/benchmark.zip",
    manifest_url="https://miplib.zib.de/downloads/benchmark-v2.test",
    reference_url="https://miplib.zib.de/downloads/miplib2017-v36.solu",
    archive_filename="benchmark.zip",
    archive_sha256="c756eefd544d83b31809306b45d3549a1a5b9378e6aa78b68738b1a3b6a418fa",
    manifest_filename="benchmark-v2.test",
    reference_filename="miplib2017-v36.solu",
    provenance_level="PINNED_ARCHIVE_ONLY",
    metadata={
        "benchmark_instances": 240,
        "benchmark_version": 2,
        "solution_file_version": 36,
        "solution_file_release": "2026-01-26",
        "source": "Zuse Institute Berlin / MIPLIB 2017",
    },
)


QPLIB_ARCHIVE = DatasetSpec(
    name="qplib-current",
    kind="qplib",
    archive_url="https://qplib.zib.de/qplib.zip",
    archive_filename="qplib.zip",
    instance_suffixes=(".qplib",),
    provenance_level="UNPINNED_EXTERNAL",
    metadata={
        "continuous_instances": 134,
        "discrete_instances": 319,
        "continuous_convex_instances": 32,
        "solverpilot_supported_subset_count": None,
        "solverpilot_scope_note": "32 is the official continuous-convex count, not the supported count; current IR also requires linear constraints and must parse/filter before benchmarking",
        "license": "CC-BY-4.0",
        "latest_site_update": "2026-08-07",
        "archive_url_verified_from_official_site": True,
        "runner_support": "acquisition-only in M12; QPLIB parser/filter not yet implemented",
        "source": "Zuse Institute Berlin / QPLIB",
    },
)


BUILTIN_DATASETS = {
    MIPLIB2017_BENCHMARK_V2.name: MIPLIB2017_BENCHMARK_V2,
    QPLIB_ARCHIVE.name: QPLIB_ARCHIVE,
}


def get_builtin_dataset(name: str) -> DatasetSpec:
    try:
        return BUILTIN_DATASETS[name]
    except KeyError as exc:
        raise KeyError(f"unknown built-in dataset: {name}") from exc


def save_dataset_spec(spec: DatasetSpec, path: str | Path) -> None:
    payload = asdict(spec)
    payload["instance_suffixes"] = list(spec.instance_suffixes)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_dataset_spec(path: str | Path) -> DatasetSpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload["instance_suffixes"] = tuple(payload.get("instance_suffixes", (".mps.gz", ".mps")))
    return DatasetSpec(**payload)
