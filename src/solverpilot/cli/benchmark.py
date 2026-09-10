from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from solverpilot.backends import probe_backends
from solverpilot.runtime.auto import builtin_backend_candidates
from solverpilot.benchmark import (
    BUILTIN_DATASETS,
    RunSpec,
    acquire_dataset,
    benchmark_registry,
    capture_environment,
    create_result_bundle,
    get_builtin_dataset,
    load_dataset_spec,
    run_benchmark,
    save_dataset_spec,
    summarize_jsonl,
    write_slurm_array,
    verify_acquired_dataset,
    evaluate_policy_jsonl,
    validate_split_file,
    generate_group_split, save_split_records,
    load_campaign_spec, miplib2017_campaign_template, run_campaign, save_campaign_spec, validate_campaign,
    scientific_report_payload, write_scientific_report,
)


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def cmd_doctor(args) -> int:
    candidates = builtin_backend_candidates()
    reports = probe_backends(candidates)
    payload = {
        "environment": capture_environment(packages=(
            "solverpilot", "numpy", "scipy", "highspy", "osqp", "pyscipopt", "nlopt", "casadi", "benchopt"
        )),
        "backends": [asdict(r) for r in reports],
    }
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    _print(payload)
    return 0


def cmd_list_datasets(args) -> int:
    _print({name: asdict(spec) for name, spec in BUILTIN_DATASETS.items()})
    return 0


def cmd_write_spec(args) -> int:
    spec = get_builtin_dataset(args.dataset)
    save_dataset_spec(spec, args.output)
    _print({"written": str(args.output), "dataset": spec.name})
    return 0


def cmd_acquire(args) -> int:
    spec = get_builtin_dataset(args.dataset) if args.dataset else load_dataset_spec(args.spec)
    record = acquire_dataset(spec, args.target, extract=not args.no_extract, timeout_s=args.timeout_s, retries=args.retries, force=args.force)
    _print(asdict(record))
    return 0


def cmd_verify_dataset(args) -> int:
    report = verify_acquired_dataset(args.target)
    _print(report.to_dict())
    return 0 if report.ok else 3


def cmd_run(args) -> int:
    backends = tuple(args.backend)
    if args.thread_env_uncontrolled:
        args.thread_env_limit = None
    registry = benchmark_registry(backends)
    spec = RunSpec(
        dataset_dir=Path(args.dataset_dir),
        manifest=Path(args.manifest),
        reference=None if args.reference is None else Path(args.reference),
        backends=backends,
        output_jsonl=Path(args.output),
        repetitions=args.repetitions,
        time_limit_s=args.time_limit_s,
        hard_timeout_s=args.hard_timeout_s,
        objective_atol=args.objective_atol,
        objective_rtol=args.objective_rtol,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        resume=not args.no_resume,
        seed=args.seed,
        thread_env_limit=args.thread_env_limit,
        solver_threads=args.solver_threads,
        worker_python_mode=args.worker_python_mode,
    )
    summary = run_benchmark(spec, registry=registry)
    summary_path = Path(args.output).with_suffix(Path(args.output).suffix + ".run.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _print(summary)
    return 0


def cmd_summarize(args) -> int:
    payload = summarize_jsonl(
        args.input, cutoff_s=args.cutoff_s, par_penalty=args.par_penalty, cost_field=args.cost_field,
        bootstrap_draws=args.bootstrap_draws, bootstrap_seed=args.bootstrap_seed,
        allow_mixed_environments=args.allow_mixed_environments,
    )
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _print(payload)
    return 0


def cmd_evaluate_policy(args) -> int:
    payload = evaluate_policy_jsonl(
        args.input, args.policy, cutoff_s=args.cutoff_s, par_penalty=args.par_penalty,
        cost_field=args.cost_field, bootstrap_draws=args.bootstrap_draws, bootstrap_seed=args.bootstrap_seed,
        allow_mixed_environments=args.allow_mixed_environments,
        allow_unmeasured_overhead=args.allow_unmeasured_overhead,
    )
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _print(payload)
    return 0


def cmd_slurm(args) -> int:
    text = write_slurm_array(
        args.output,
        spec_path="",
        dataset_dir=args.dataset_dir,
        manifest=args.manifest,
        reference=args.reference,
        output_dir=args.output_dir,
        backends=tuple(args.backend),
        shards=args.shards,
        repetitions=args.repetitions,
        time_limit_s=args.time_limit_s,
        job_name=args.job_name,
        cpus_per_task=args.cpus_per_task,
        memory=args.memory,
        walltime=args.walltime,
        thread_env_limit=args.thread_env_limit,
        solver_threads=args.solver_threads,
        worker_python_mode=args.worker_python_mode,
    )
    print(text)
    return 0



def cmd_validate_split(args) -> int:
    expected = None
    if args.manifest:
        from solverpilot.evaluation import parse_test_manifest
        expected = [Path(x).name.removesuffix(".gz").removesuffix(".mps") for x in parse_test_manifest(Path(args.manifest).read_text(encoding="utf-8"))]
    report = validate_split_file(args.input, expected_instances=expected)
    _print(report.to_dict())
    return 0 if report.ok else 4


def cmd_write_miplib_campaign(args) -> int:
    spec = miplib2017_campaign_template(tuple(args.backend), repetitions=args.repetitions, cutoff_s=args.cutoff_s, hard_timeout_s=args.hard_timeout_s, seed=args.seed, solver_threads=args.solver_threads)
    save_campaign_spec(spec, args.output)
    _print({"written": str(args.output), "campaign_id": spec.campaign_id, "campaign_sha256": spec.sha256})
    return 0

def cmd_validate_campaign(args) -> int:
    spec=load_campaign_spec(args.campaign); report=validate_campaign(spec,dataset_target=args.dataset_target,split_path=args.split); _print(report.to_dict()); return 0 if report.ok else 5

def cmd_run_campaign(args) -> int:
    spec=load_campaign_spec(args.campaign); payload=run_campaign(spec,dataset_target=args.dataset_target,output_jsonl=args.output,split_path=args.split,allow_unverified_dataset=args.allow_unverified_dataset); _print(payload); return 0

def cmd_generate_split(args) -> int:
    payload=json.loads(Path(args.groups).read_text(encoding="utf-8"))
    if not isinstance(payload,dict): raise ValueError("groups JSON must be an object mapping instance to group/null")
    records=generate_group_split({str(k):None if v is None else str(v) for k,v in payload.items()},train_fraction=args.train_fraction,validation_fraction=args.validation_fraction,test_fraction=args.test_fraction,seed=args.seed)
    report=save_split_records(records,args.output,seed=args.seed); _print(report.to_dict()); return 0

def cmd_scientific_report(args) -> int:
    campaign=load_campaign_spec(args.campaign); summary=json.loads(Path(args.summary).read_text(encoding="utf-8")); campaign_run=None if args.campaign_run is None else json.loads(Path(args.campaign_run).read_text(encoding="utf-8"))
    payload=scientific_report_payload(campaign,summary,campaign_run=campaign_run,limitations=tuple(args.limitation or ())); write_scientific_report(payload,json_path=args.output_json,markdown_path=args.output_md); _print({"json":str(args.output_json),"markdown":str(args.output_md),"deployable_scientific_evidence":payload["deployable_scientific_evidence"]}); return 0

def cmd_bundle(args) -> int:
    payload = create_result_bundle(args.output, files=args.file, metadata={"label": args.label} if args.label else {})
    _print(payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="solverpilot-bench", description="Portable, auditable optimization benchmark harness")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("doctor", help="capture environment and backend health")
    sp.add_argument("--output")
    sp.set_defaults(func=cmd_doctor)

    sp = sub.add_parser("list-datasets", help="list built-in public dataset specs")
    sp.set_defaults(func=cmd_list_datasets)

    sp = sub.add_parser("write-spec", help="write a built-in dataset spec to JSON")
    sp.add_argument("dataset", choices=sorted(BUILTIN_DATASETS))
    sp.add_argument("--output", required=True, type=Path)
    sp.set_defaults(func=cmd_write_spec)

    sp = sub.add_parser("acquire", help="download and hash-lock a public dataset")
    group = sp.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", choices=sorted(BUILTIN_DATASETS))
    group.add_argument("--spec")
    sp.add_argument("--target", required=True)
    sp.add_argument("--timeout-s", type=float, default=60.0)
    sp.add_argument("--retries", type=int, default=3)
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--no-extract", action="store_true")
    sp.set_defaults(func=cmd_acquire)

    sp = sub.add_parser("verify-dataset", help="re-hash acquisition files and verify manifest/reference/instances")
    sp.add_argument("--target", required=True)
    sp.set_defaults(func=cmd_verify_dataset)

    sp = sub.add_parser("run", help="run a resumable rectangular MPS benchmark matrix")
    sp.add_argument("--dataset-dir", required=True)
    sp.add_argument("--manifest", required=True)
    sp.add_argument("--reference")
    sp.add_argument("--backend", action="append", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument("--repetitions", type=int, default=1)
    sp.add_argument("--time-limit-s", type=float)
    sp.add_argument("--hard-timeout-s", type=float, help="controller-enforced process timeout; enables solver isolation")
    sp.add_argument("--objective-atol", type=float, default=1e-6)
    sp.add_argument("--objective-rtol", type=float, default=1e-7)
    sp.add_argument("--shard-index", type=int, default=0)
    sp.add_argument("--shard-count", type=int, default=1)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--thread-env-limit", type=int, default=1, help="set common BLAS/OpenMP thread env vars in isolated workers; use 0 only via --thread-env-uncontrolled")
    sp.add_argument("--thread-env-uncontrolled", action="store_true", help="do not set common C/BLAS thread env vars")
    sp.add_argument("--solver-threads", type=int, help="require backend-level solver thread control; unsupported backends fail explicitly")
    sp.add_argument("--worker-python-mode", choices=("normal", "no_site"), default="no_site", help="isolated worker startup mode; no_site disables sitecustomize and uses an explicit import path")
    sp.add_argument("--no-resume", action="store_true")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("summarize", help="merge JSONL shards and compute PAR/SBS/VBS/performance profiles")
    sp.add_argument("--input", action="append", required=True)
    sp.add_argument("--cutoff-s", type=float, required=True)
    sp.add_argument("--par-penalty", type=float, default=10.0)
    sp.add_argument("--cost-field", default="wall_s", choices=["wall_s", "worker_solve_wall_s", "trace.solve_s", "trace.total_s"])
    sp.add_argument("--bootstrap-draws", type=int, default=10000)
    sp.add_argument("--bootstrap-seed", type=int, default=0)
    sp.add_argument("--allow-mixed-environments", action="store_true", help="override timing comparability guard for heterogeneous environments")
    sp.add_argument("--output")
    sp.set_defaults(func=cmd_summarize)

    sp = sub.add_parser("evaluate-policy", help="evaluate an external frozen selector policy against direct solver rows")
    sp.add_argument("--input", action="append", required=True)
    sp.add_argument("--policy", required=True)
    sp.add_argument("--cutoff-s", type=float, required=True)
    sp.add_argument("--par-penalty", type=float, default=10.0)
    sp.add_argument("--cost-field", default="wall_s", choices=["wall_s", "worker_solve_wall_s", "trace.solve_s", "trace.total_s"])
    sp.add_argument("--bootstrap-draws", type=int, default=10000)
    sp.add_argument("--bootstrap-seed", type=int, default=0)
    sp.add_argument("--allow-mixed-environments", action="store_true", help="override timing comparability guard for heterogeneous environments")
    sp.add_argument("--allow-unmeasured-overhead", action="store_true", help="treat missing policy decision overhead as zero for non-deployable analysis; default is fail-closed")
    sp.add_argument("--output")
    sp.set_defaults(func=cmd_evaluate_policy)

    sp = sub.add_parser("emit-slurm", help="emit a deterministic SLURM array launcher")
    sp.add_argument("--dataset-dir", required=True)
    sp.add_argument("--manifest", required=True)
    sp.add_argument("--reference")
    sp.add_argument("--backend", action="append", required=True)
    sp.add_argument("--output-dir", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument("--shards", type=int, required=True)
    sp.add_argument("--repetitions", type=int, default=1)
    sp.add_argument("--time-limit-s", type=float)
    sp.add_argument("--job-name", default="solverpilot-bench")
    sp.add_argument("--cpus-per-task", type=int, default=1)
    sp.add_argument("--memory", default="8G")
    sp.add_argument("--walltime", default="02:00:00")
    sp.add_argument("--thread-env-limit", type=int, default=1)
    sp.add_argument("--solver-threads", type=int)
    sp.add_argument("--worker-python-mode", choices=("normal", "no_site"), default="no_site")
    sp.set_defaults(func=cmd_slurm)

    sp = sub.add_parser("validate-split", help="validate train/validation/test split coverage and group leakage")
    sp.add_argument("--input", required=True)
    sp.add_argument("--manifest", help="optional benchmark manifest requiring exact instance coverage")
    sp.set_defaults(func=cmd_validate_split)

    sp = sub.add_parser("write-miplib-campaign", help="write the frozen MIPLIB-2017 benchmark-v2 campaign protocol")
    sp.add_argument("--backend", action="append", required=True); sp.add_argument("--output", required=True, type=Path); sp.add_argument("--repetitions", type=int, default=1); sp.add_argument("--cutoff-s", type=float, default=3600.0); sp.add_argument("--hard-timeout-s", type=float); sp.add_argument("--seed", type=int, default=0); sp.add_argument("--solver-threads", type=int, default=1); sp.set_defaults(func=cmd_write_miplib_campaign)

    sp = sub.add_parser("validate-campaign", help="validate campaign identity, split binding and dataset scientific provenance")
    sp.add_argument("--campaign", required=True); sp.add_argument("--dataset-target"); sp.add_argument("--split"); sp.set_defaults(func=cmd_validate_campaign)

    sp = sub.add_parser("run-campaign", help="run a validated benchmark campaign")
    sp.add_argument("--campaign", required=True); sp.add_argument("--dataset-target", required=True); sp.add_argument("--output", required=True); sp.add_argument("--split"); sp.add_argument("--allow-unverified-dataset", action="store_true"); sp.set_defaults(func=cmd_run_campaign)

    sp = sub.add_parser("generate-split", help="create a deterministic group-safe train/validation/test split")
    sp.add_argument("--groups", required=True); sp.add_argument("--output", required=True); sp.add_argument("--train-fraction", type=float, default=0.60); sp.add_argument("--validation-fraction", type=float, default=0.20); sp.add_argument("--test-fraction", type=float, default=0.20); sp.add_argument("--seed", type=int, default=0); sp.set_defaults(func=cmd_generate_split)

    sp = sub.add_parser("scientific-report", help="render a claim-bounded scientific benchmark report")
    sp.add_argument("--campaign", required=True); sp.add_argument("--summary", required=True); sp.add_argument("--campaign-run"); sp.add_argument("--limitation", action="append"); sp.add_argument("--output-json", required=True); sp.add_argument("--output-md", required=True); sp.set_defaults(func=cmd_scientific_report)

    sp = sub.add_parser("bundle", help="bundle result files with SHA-256 manifest")
    sp.add_argument("--file", action="append", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument("--label")
    sp.set_defaults(func=cmd_bundle)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code = args.func(args)
    except Exception as exc:
        print(json.dumps({"error_type": type(exc).__name__, "error": str(exc)}, indent=2), file=sys.stderr)
        raise SystemExit(2) from exc
    raise SystemExit(code)


if __name__ == "__main__":
    main()
