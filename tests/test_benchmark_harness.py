from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from solverpilot.benchmark import (
    DatasetSpec,
    RunSpec,
    acquire_dataset,
    benchmark_registry,
    create_result_bundle,
    run_benchmark,
    summarize_jsonl,
    verify_acquired_dataset,
)
from solverpilot.benchmark.acquire import sha256_file
from solverpilot.benchmark.runner import _stable_shard_index


BASIC = """NAME          BASIC
ROWS
 N  COST
 G  DEMAND
 L  CAP
COLUMNS
    X1        COST       1        DEMAND     1
    X1        CAP        1
    X2        COST       2        DEMAND     1
    X2        CAP        2
RHS
    RHS1      DEMAND     3        CAP        8
BOUNDS
 UP BND1      X1         5
ENDATA
"""


def _tiny_dataset(tmp_path: Path, names=("a.mps",)):
    data = tmp_path / "data"; data.mkdir()
    for name in names:
        (data/name).write_text(BASIC)
    manifest = tmp_path / "bench.test"; manifest.write_text("\n".join(names)+"\n")
    reference = tmp_path / "bench.solu"; reference.write_text("\n".join(f"=opt= {Path(n).stem} 3" for n in names)+"\n")
    return data, manifest, reference


def test_local_zip_acquisition_and_verification(tmp_path):
    src = tmp_path / "src.zip"
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr("x.qplib", "payload")
    spec = DatasetSpec(name="local", kind="qplib", archive_url=src.as_uri(), archive_filename="x.zip", archive_sha256=sha256_file(src), metadata={"continuous_instances":1,"discrete_instances":0})
    target = tmp_path / "target"
    record = acquire_dataset(spec, target)
    assert record.extracted_files == 1
    report = verify_acquired_dataset(target)
    assert report.ok
    assert report.instances_present == 1


def test_acquisition_rejects_bad_sha(tmp_path):
    src=tmp_path/"x.zip"
    with zipfile.ZipFile(src,"w") as zf: zf.writestr("x","x")
    spec=DatasetSpec(name="bad",kind="x",archive_url=src.as_uri(),archive_filename="x.zip",archive_sha256="0"*64)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        acquire_dataset(spec,tmp_path/"out")


def test_acquisition_rejects_archive_traversal(tmp_path):
    src=tmp_path/"x.zip"
    with zipfile.ZipFile(src,"w") as zf: zf.writestr("../escape","x")
    spec=DatasetSpec(name="bad",kind="x",archive_url=src.as_uri(),archive_filename="x.zip",archive_sha256=sha256_file(src))
    with pytest.raises(ValueError, match="escapes target"):
        acquire_dataset(spec,tmp_path/"out")


def test_sharding_is_deterministic_and_covers_names():
    names=[f"i{i}.mps" for i in range(50)]
    buckets=[[n for n in names if _stable_shard_index(n,4)==k] for k in range(4)]
    assert sorted(x for b in buckets for x in b)==sorted(names)
    assert all(set(buckets[i]).isdisjoint(buckets[j]) for i in range(4) for j in range(i+1,4))


def test_run_benchmark_is_rectangular_and_records_environment(tmp_path):
    data, manifest, reference = _tiny_dataset(tmp_path)
    output=tmp_path/"r.jsonl"
    backends=("scipy-highs-ds","scipy-highs-ipm")
    spec=RunSpec(dataset_dir=data,manifest=manifest,reference=reference,backends=backends,output_jsonl=output,repetitions=2,hard_timeout_s=10)
    run=run_benchmark(spec,registry=benchmark_registry(backends))
    rows=[json.loads(x) for x in output.read_text().splitlines()]
    assert len(rows)==4
    assert all(r["state"]=="solved" and r["validated"] for r in rows)
    assert all(r["reference_check"]=="objective_matches_optimum" for r in rows)
    assert len({r["environment_id"] for r in rows})==1
    assert run["environment_id"]==rows[0]["environment_id"]
    assert rows[0]["thread_policy"]["thread_env_limit"]==1
    assert "runtime_overhead_s" in rows[0]["trace"]


def test_run_resume_skips_completed_rows(tmp_path):
    data, manifest, reference = _tiny_dataset(tmp_path)
    output=tmp_path/"r.jsonl"; backends=("scipy-highs-ds",)
    spec=RunSpec(dataset_dir=data,manifest=manifest,reference=reference,backends=backends,output_jsonl=output,repetitions=1,hard_timeout_s=10)
    first=run_benchmark(spec,registry=benchmark_registry(backends)); before=output.read_text()
    second=run_benchmark(spec,registry=benchmark_registry(backends))
    assert output.read_text()==before
    assert first["counters"]["completed"]==1
    assert second["counters"]["skipped_resume"]==1


def test_instance_content_change_creates_new_run_id(tmp_path):
    data, manifest, reference = _tiny_dataset(tmp_path)
    output=tmp_path/"r.jsonl"; backends=("scipy-highs-ds",)
    spec=RunSpec(dataset_dir=data,manifest=manifest,reference=reference,backends=backends,output_jsonl=output,repetitions=1,hard_timeout_s=10)
    run_benchmark(spec,registry=benchmark_registry(backends))
    first=json.loads(output.read_text().splitlines()[0])["run_id"]
    (data/"a.mps").write_text(BASIC.replace("DEMAND     3", "DEMAND     4"))
    run_benchmark(spec,registry=benchmark_registry(backends))
    rows=[json.loads(x) for x in output.read_text().splitlines()]
    assert len(rows)==2 and rows[-1]["run_id"]!=first


def test_hard_timeout_is_controller_enforced(tmp_path):
    data, manifest, reference = _tiny_dataset(tmp_path)
    output=tmp_path/"r.jsonl"; backends=("scipy-highs-ds",)
    spec=RunSpec(dataset_dir=data,manifest=manifest,reference=reference,backends=backends,output_jsonl=output,hard_timeout_s=1e-6)
    run_benchmark(spec,registry=benchmark_registry(backends))
    row=json.loads(output.read_text().strip())
    assert row["state"]=="hard_timeout"


def test_summary_computes_sbs_vbs_and_excludes_auto_from_sbs(tmp_path):
    data, manifest, reference = _tiny_dataset(tmp_path, names=("a.mps","b.mps"))
    output=tmp_path/"r.jsonl"; backends=("scipy-highs-ds","scipy-highs-ipm","@auto")
    spec=RunSpec(dataset_dir=data,manifest=manifest,reference=reference,backends=backends,output_jsonl=output,repetitions=1,hard_timeout_s=10)
    run_benchmark(spec,registry=benchmark_registry(backends))
    summary=summarize_jsonl([output],cutoff_s=5,cost_field="worker_solve_wall_s",bootstrap_draws=100)
    assert summary["sbs_solver"] in {"scipy-highs-ds","scipy-highs-ipm"}
    assert summary["vbs_cost"] <= summary["sbs_cost"]
    assert summary["auto_policy_metrics"] is not None


def test_solver_threads_request_fails_explicitly_if_backend_cannot_enforce(tmp_path):
    data, manifest, reference = _tiny_dataset(tmp_path)
    output=tmp_path/"r.jsonl"; backends=("scipy-highs-ds",)
    spec=RunSpec(dataset_dir=data,manifest=manifest,reference=reference,backends=backends,output_jsonl=output,solver_threads=1,hard_timeout_s=10)
    run_benchmark(spec,registry=benchmark_registry(backends))
    row=json.loads(output.read_text().strip())
    assert row["state"]=="worker_error"
    assert "thread" in (row.get("error") or "").lower()


def test_result_bundle_contains_hash_manifest(tmp_path):
    a=tmp_path/"a.json"; a.write_text("{}")
    z=tmp_path/"bundle.zip"
    result=create_result_bundle(z,files=[a],metadata={"x":1})
    assert len(result["sha256"])==64
    with zipfile.ZipFile(z) as zf:
        manifest=json.loads(zf.read("bundle-manifest.json"))
        assert manifest["metadata"]=={"x":1}
        assert manifest["entries"][0]["name"]=="a.json"


def test_acquisition_force_replaces_extracted_snapshot_without_stale_files(tmp_path):
    src = tmp_path / "x.zip"
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr("keep.qplib", "v1")
        zf.writestr("stale.qplib", "old")
    spec = DatasetSpec(
        name="replace", kind="qplib", archive_url=src.as_uri(),
        archive_filename="x.zip", archive_sha256=sha256_file(src),
    )
    target = tmp_path / "target"
    acquire_dataset(spec, target)
    assert (target / "instances" / "stale.qplib").exists()

    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr("keep.qplib", "v2")
    spec2 = DatasetSpec(
        name="replace", kind="qplib", archive_url=src.as_uri(),
        archive_filename="x.zip", archive_sha256=sha256_file(src),
    )
    acquire_dataset(spec2, target, force=True)
    assert (target / "instances" / "keep.qplib").read_text() == "v2"
    assert not (target / "instances" / "stale.qplib").exists()


def test_unpinned_acquisition_is_hash_locked_but_not_claimed_verified(tmp_path):
    src = tmp_path / "x.zip"
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr("x", "payload")
    spec = DatasetSpec(name="unpinned", kind="x", archive_url=src.as_uri(), archive_filename="x.zip")
    record = acquire_dataset(spec, tmp_path / "target", extract=False)
    assert record.files[0].sha256
    assert record.files[0].expected_sha256 is None
    assert record.files[0].verified is False


def test_relative_acquisition_lock_verifies_from_different_working_directory(tmp_path, monkeypatch):
    src = tmp_path / "x.zip"
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr("x.qplib", "payload")
    spec = DatasetSpec(
        name="relative", kind="qplib", archive_url=src.as_uri(),
        archive_filename="x.zip", archive_sha256=sha256_file(src),
    )
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    acquire_dataset(spec, "dataset")
    monkeypatch.chdir(tmp_path)
    report = verify_acquired_dataset(work / "dataset")
    assert report.ok


def test_builtin_miplib_archive_is_pinned_to_verified_m22_hash():
    from solverpilot.benchmark.specs import MIPLIB2017_BENCHMARK_V2
    assert MIPLIB2017_BENCHMARK_V2.archive_sha256 == "c756eefd544d83b31809306b45d3549a1a5b9378e6aa78b68738b1a3b6a418fa"


def test_pinned_external_dataset_can_attain_scientific_provenance(tmp_path):
    src=tmp_path/'src.zip'
    with zipfile.ZipFile(src,'w') as zf: zf.writestr('x.qplib','payload')
    spec=DatasetSpec(name='local-pinned',kind='qplib',archive_url=src.as_uri(),archive_filename='x.zip',archive_sha256=sha256_file(src),provenance_level='PINNED_EXTERNAL',metadata={'continuous_instances':1,'discrete_instances':0})
    target=tmp_path/'target'; record=acquire_dataset(spec,target)
    assert record.all_remote_inputs_pinned is True and record.unpinned_roles == ()
    assert (target/'acquisition-lock.sha256').exists()
    report=verify_acquired_dataset(target); assert report.ok and report.scientific_provenance_verified is True


def test_unpinned_dataset_never_becomes_scientifically_verified_by_self_hash(tmp_path):
    src=tmp_path/'src.zip'
    with zipfile.ZipFile(src,'w') as zf: zf.writestr('x.qplib','payload')
    spec=DatasetSpec(name='external-unpinned',kind='qplib',archive_url=src.as_uri(),archive_filename='x.zip',provenance_level='UNPINNED_EXTERNAL',metadata={'continuous_instances':1,'discrete_instances':0})
    target=tmp_path/'target'; acquire_dataset(spec,target); report=verify_acquired_dataset(target)
    assert report.ok and not report.all_remote_inputs_pinned and report.unpinned_roles == ('archive',) and not report.scientific_provenance_verified


def test_acquisition_lock_sidecar_detects_lock_tampering(tmp_path):
    src=tmp_path/'src.zip'
    with zipfile.ZipFile(src,'w') as zf: zf.writestr('x.qplib','payload')
    spec=DatasetSpec(name='pinned',kind='qplib',archive_url=src.as_uri(),archive_filename='x.zip',archive_sha256=sha256_file(src),provenance_level='PINNED_EXTERNAL',metadata={'continuous_instances':1,'discrete_instances':0})
    target=tmp_path/'target'; acquire_dataset(spec,target); lock=target/'acquisition-lock.json'; payload=json.loads(lock.read_text()); payload['metadata']['continuous_instances']=2; lock.write_text(json.dumps(payload,indent=2,sort_keys=True))
    report=verify_acquired_dataset(target); assert not report.ok and not report.lock_integrity_ok


def test_benchmark_row_checksum_detects_post_run_tampering(tmp_path):
    data,manifest,reference=_tiny_dataset(tmp_path); output=tmp_path/'r.jsonl'; backends=('scipy-highs-ds',)
    run_benchmark(RunSpec(dataset_dir=data,manifest=manifest,reference=reference,backends=backends,output_jsonl=output,repetitions=1,hard_timeout_s=10),registry=benchmark_registry(backends))
    row=json.loads(output.read_text().strip()); assert len(row['row_sha256'])==64; row['wall_s']=float(row['wall_s'])+100.0; output.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='row_sha256 mismatch'): summarize_jsonl([output],cutoff_s=5)


def test_summary_labels_vbs_as_non_deployable_oracle(tmp_path):
    data,manifest,reference=_tiny_dataset(tmp_path,names=('a.mps','b.mps')); output=tmp_path/'r.jsonl'; backends=('scipy-highs-ds','scipy-highs-ipm')
    run_benchmark(RunSpec(dataset_dir=data,manifest=manifest,reference=reference,backends=backends,output_jsonl=output,repetitions=1,hard_timeout_s=10),registry=benchmark_registry(backends))
    summary=summarize_jsonl([output],cutoff_s=5,cost_field='worker_solve_wall_s',bootstrap_draws=50)
    assert summary['vbs']['is_oracle'] is True and summary['vbs']['deployable'] is False and summary['experiment_identities']


def test_experiment_identity_is_recomputed_from_provenance_fields(tmp_path):
    from solverpilot.benchmark.integrity import attach_row_integrity
    data,manifest,reference=_tiny_dataset(tmp_path,names=('a.mps','b.mps')); output=tmp_path/'r.jsonl'; backends=('scipy-highs-ds',)
    run_benchmark(RunSpec(dataset_dir=data,manifest=manifest,reference=reference,backends=backends,output_jsonl=output,repetitions=1,hard_timeout_s=10),registry=benchmark_registry(backends))
    rows=[json.loads(line) for line in output.read_text().splitlines()]; rows[0]['thread_policy']['thread_env_limit']=7; rows[0]=attach_row_integrity(rows[0]); output.write_text('\n'.join(json.dumps(r,sort_keys=True) for r in rows)+'\n')
    with pytest.raises(ValueError,match='experiment_identity does not match'): summarize_jsonl([output],cutoff_s=5)
