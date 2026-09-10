from __future__ import annotations

import json

from solverpilot.cli.benchmark import main


def test_benchmark_cli_list_datasets(capsys):
    try:
        main(["list-datasets"])
    except SystemExit as exc:
        assert exc.code == 0
    payload=json.loads(capsys.readouterr().out)
    assert "miplib2017-benchmark-v2" in payload
    assert "qplib-current" in payload


def test_benchmark_cli_doctor(capsys):
    try:
        main(["doctor"])
    except SystemExit as exc:
        assert exc.code == 0
    payload=json.loads(capsys.readouterr().out)
    assert "environment" in payload and "backends" in payload


def test_benchmark_cli_emit_slurm_includes_thread_policy(tmp_path, capsys):
    out=tmp_path/"job.sh"
    argv=["emit-slurm","--dataset-dir","d","--manifest","m","--backend","scipy-highs-ds","--output-dir","o","--output",str(out),"--shards","2","--thread-env-limit","1","--solver-threads","1"]
    try:
        main(argv)
    except SystemExit as exc:
        assert exc.code == 0
    text=out.read_text()
    assert "--thread-env-limit 1" in text
    assert "--solver-threads 1" in text
