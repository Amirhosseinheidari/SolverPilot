import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

from solverpilot.experimental.learned_lp import LPSelector
from solverpilot.experimental.lp_environment import bind_lp_environment

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/"docs/evidence/learned-lp-local"


def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/"benchmarks"/(name+".py"))
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_environment_binding_preserves_actual_identity_and_small_memory_drift():
    m=LPSelector.load(EVIDENCE/"model.json")
    env=json.loads((EVIDENCE/"protocol.json").read_text())["environment"]
    current=copy.deepcopy(env); current["memory_bytes"]+=8192
    result=bind_lp_environment(m,EVIDENCE/"protocol.json",current)
    assert result["compatible"] and result["reason"]=="bounded_guest_memory_reporting_drift"
    assert result["actual_environment_id"]!=result["decision_environment_id"]
    assert current["memory_bytes"]==env["memory_bytes"]+8192


@pytest.mark.parametrize("field,value",[("cpu_model","different"),("python","different"),
    ("memory_bytes",1024*1024),("packages",{}),("env_threads",{}),("platform","different")])
def test_material_environment_change_rejected(field,value):
    m=LPSelector.load(EVIDENCE/"model.json")
    env=json.loads((EVIDENCE/"protocol.json").read_text())["environment"]
    env[field]=value
    binding=bind_lp_environment(m,EVIDENCE/"protocol.json",env)
    assert not binding["compatible"]
    assert binding["decision_environment_id"]!=m.environment_id


def test_training_protocol_tamper_rejected(tmp_path):
    m=LPSelector.load(EVIDENCE/"model.json")
    path=tmp_path/"protocol.json";path.write_text('{}')
    with pytest.raises(ValueError,match="digest"):
        bind_lp_environment(m,path,{})


def fake_rows():
    mod=module("qualify_public_lp")
    cases=[dict(file=f"i{i}",group=f"g{i}",reference=0.) for i in range(24)]
    rows=[dict(instance=c["file"],strategy=s,repeat=r,verified=True,
               wall_s=1.,api_wall_s=.1,objective=0.,decision={"reason":"learned_stump"})
          for c in cases for s in mod.STRATEGIES for r in range(mod.REPEATS)]
    return mod,cases,rows


def test_failed_repeats_cannot_be_hidden_and_default_is_compared():
    mod,cases,rows=fake_rows()
    rows[0]["verified"]=False
    report=mod.summarize(rows,cases)
    assert report["mean_par10_s"]["learned"]>report["mean_par10_s"]["default"]
    assert not report["gates"]["default_no_lost_verified_solves"]
    assert not report["public_gate_passed"]


def test_duplicate_or_missing_outcomes_rejected():
    mod,cases,rows=fake_rows()
    for malformed in (rows[:-1], rows+[rows[0]]):
        with pytest.raises(ValueError,match="outcomes"):
            mod.summarize(malformed,cases)


def test_verified_objective_mismatch_blocks_gate():
    mod,cases,rows=fake_rows();rows[0]["objective"]=1.
    report=mod.summarize(rows,cases)
    assert report["objective_mismatches"] and not report["public_gate_passed"]


@pytest.mark.skipif(os.name!='posix',reason='POSIX process-group controller')
def test_controller_deadline_is_a_recorded_failure():
    mod=module("qualify_public_lp")
    row=mod.run_process([sys.executable,'-c','import time;time.sleep(10)'],dict(os.environ),.1)
    assert row['status']=='controller_timeout' and not row['verified']
