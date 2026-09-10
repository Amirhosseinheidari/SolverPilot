from __future__ import annotations

import json
from pathlib import Path

import pytest

from solverpilot.benchmark.environment import benchmark_environment_fingerprint, thread_environment
from solverpilot.benchmark.model import RunSpec
from solverpilot.benchmark.runner import _load_completed, _protocol_id, _task_order
from solverpilot.benchmark.slurm import render_slurm_array
from solverpilot.benchmark.specs import MIPLIB2017_BENCHMARK_V2, QPLIB_ARCHIVE
from solverpilot.benchmark.summary import summarize_rows
from solverpilot.benchmark.policy import evaluate_policy_map
from solverpilot.backends import BackendRegistry, ScipyHighsLPBackend


def _row(*, run_id: str, env: str, backend: str = "a", instance: str = "i", wall: float = 1.0):
    return {
        "run_id": run_id,
        "protocol_id": "p",
        "environment_id": env,
        "instance_sha256": "a" * 64,
        "instance": instance,
        "backend": backend,
        "repetition": 0,
        "state": "solved",
        "validated": True,
        "reference_check": "not_checkable", "public_status": "valid_optimal",
        "optimality_evidence": {"independently_verified_optimal": True},
        "wall_s": wall,
        "worker_solve_wall_s": wall,
        "trace": {"solve_s": wall, "total_s": wall},
    }


def test_thread_environment_covers_common_pools():
    env = thread_environment(2)
    assert env["OMP_NUM_THREADS"] == "2"
    assert env["OPENBLAS_NUM_THREADS"] == "2"
    assert env["MKL_NUM_THREADS"] == "2"
    assert env["BLIS_NUM_THREADS"] == "2"
    assert thread_environment(None) == {}
    with pytest.raises(ValueError):
        thread_environment(0)


def test_environment_fingerprint_ignores_hostname_but_tracks_cpu_and_threads():
    base = {
        "python": "3.13", "platform": "linux", "machine": "x86_64", "hostname": "node-a",
        "cpu_model": "CPU-X", "logical_cpus": 8, "memory_bytes": 10, "packages": {"scipy": "1"},
    }
    other_host = {**base, "hostname": "node-b"}
    a = benchmark_environment_fingerprint(base, thread_env_limit=1, solver_threads=None)
    b = benchmark_environment_fingerprint(other_host, thread_env_limit=1, solver_threads=None)
    assert a == b
    assert a != benchmark_environment_fingerprint({**base, "cpu_model": "CPU-Y"}, thread_env_limit=1, solver_threads=None)
    assert a != benchmark_environment_fingerprint(base, thread_env_limit=2, solver_threads=None)


def test_runspec_validates_thread_controls(tmp_path):
    kw = dict(dataset_dir=tmp_path, manifest=tmp_path/"m", reference=None, backends=("x",), output_jsonl=tmp_path/"o")
    with pytest.raises(ValueError):
        RunSpec(**kw, thread_env_limit=0)
    with pytest.raises(ValueError):
        RunSpec(**kw, solver_threads=0)


def test_protocol_identity_changes_with_thread_policy(tmp_path):
    manifest = tmp_path / "m.test"; manifest.write_text("a.mps\n")
    registry = BackendRegistry(); registry.register(ScipyHighsLPBackend(method="highs-ds"))
    base = dict(dataset_dir=tmp_path, manifest=manifest, reference=None, backends=("scipy-highs-ds",), output_jsonl=tmp_path/"o")
    assert _protocol_id(RunSpec(**base, thread_env_limit=1), registry) != _protocol_id(RunSpec(**base, thread_env_limit=2), registry)
    assert _protocol_id(RunSpec(**base, solver_threads=None), registry) != _protocol_id(RunSpec(**base, solver_threads=1), registry)


def test_backend_execution_order_is_deterministic_but_interleaved():
    one = _task_order("a.mps", backends=("a", "b", "c"), repetitions=3, seed=4)
    two = _task_order("a.mps", backends=("a", "b", "c"), repetitions=3, seed=4)
    assert one == two
    assert sorted(one) == sorted((b, r) for b in ("a", "b", "c") for r in range(3))
    assert one != [(b, r) for b in ("a", "b", "c") for r in range(3)]


def test_summary_rejects_heterogeneous_timing_by_default():
    rows = [_row(run_id="1", env="cpu-a"), _row(run_id="2", env="cpu-b", backend="b")]
    with pytest.raises(ValueError, match="multiple benchmark environments"):
        summarize_rows(rows, cutoff_s=5)
    out = summarize_rows(rows, cutoff_s=5, allow_mixed_environments=True)
    assert out["mixed_environments_allowed"] is True
    assert set(out["environment_ids"]) == {"cpu-a", "cpu-b"}


def test_policy_evaluation_rejects_mixed_environments():
    rows = [_row(run_id="1", env="cpu-a", backend="a"), _row(run_id="2", env="cpu-b", backend="b")]
    policy = {"i": {"backend": "a", "overhead_s": 0.0}}
    with pytest.raises(ValueError, match="multiple benchmark environments"):
        evaluate_policy_map(rows, policy, cutoff_s=5)


def test_resume_repairs_only_truncated_tail(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text('{"run_id":"a","complete":true}\n{"run_id":"b"')
    assert _load_completed(path) == {"a"}
    assert path.read_text() == '{"run_id":"a","complete":true}\n'


def test_resume_rejects_middle_corruption(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text('{"run_id":"a","complete":true}\nnot-json\n{"run_id":"b","complete":true}\n')
    with pytest.raises(ValueError, match="invalid JSONL"):
        _load_completed(path)


def test_slurm_carries_thread_policy():
    text = render_slurm_array(
        spec_path="", dataset_dir="d", manifest="m", reference=None, output_dir="o",
        backends=("b",), shards=4, repetitions=2, time_limit_s=10,
        thread_env_limit=1, solver_threads=1,
    )
    assert "--thread-env-limit 1" in text
    assert "--solver-threads 1" in text


def test_dataset_specs_do_not_overclaim_support():
    assert MIPLIB2017_BENCHMARK_V2.metadata["benchmark_instances"] == 240
    assert MIPLIB2017_BENCHMARK_V2.provenance_level == "PINNED_ARCHIVE_ONLY"
    assert "solution_records" not in MIPLIB2017_BENCHMARK_V2.metadata
    assert QPLIB_ARCHIVE.metadata["continuous_convex_instances"] == 32
    assert QPLIB_ARCHIVE.provenance_level == "UNPINNED_EXTERNAL"
    assert QPLIB_ARCHIVE.metadata["solverpilot_supported_subset_count"] is None
    assert "not the supported count" in QPLIB_ARCHIVE.metadata["solverpilot_scope_note"]

def test_reference_free_feasible_candidate_does_not_count_as_exact_solve():
    feasible = _row(run_id="1", env="e", backend="local", wall=0.01)
    feasible["public_status"] = "valid_feasible"
    optimal = _row(run_id="2", env="e", backend="proof", wall=0.02)
    optimal["public_status"] = "valid_optimal"
    out = summarize_rows([feasible, optimal], cutoff_s=1.0, par_penalty=10.0)
    assert out["solver_stats"]["local"]["reference_ok_rows"] == 0
    assert out["solver_stats"]["proof"]["reference_ok_rows"] == 1
    assert out["sbs_solver"] == "proof"


def test_dataset_spec_metadata_is_deeply_immutable():
    with pytest.raises(TypeError):
        MIPLIB2017_BENCHMARK_V2.metadata["benchmark_instances"] = 999
