"""Frozen, serial public-corpus evaluation of the already trained local selector."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import perf_counter

import numpy as np
from solverpilot.benchmark.environment import capture_environment, thread_environment


STRATEGIES = ("learned","default","production","highs-simplex","highs-ipm","ortools-pdlp")
REPEATS = 2
CUTOFF = 2.0
DEADLINE = 12.0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")


def run_process(command, env, deadline):
    """Kill the whole POSIX group, including isolated PDLP children, on deadline."""
    start = perf_counter()
    proc = subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                            text=True,start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=deadline)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        return {"status":"controller_timeout","verified":False,"wall_s":perf_counter()-start}
    wall = perf_counter()-start
    if proc.returncode:
        return {"status":"worker_error","verified":False,"wall_s":wall,"stderr":stderr[-2000:]}
    try:
        result = json.loads(stdout.strip().splitlines()[-1])
        return {**result,"wall_s":wall}
    except (ValueError,IndexError):
        return {"status":"invalid_worker_output","verified":False,"wall_s":wall,"stdout":stdout[-1000:]}


def summarize(rows, cases):
    """Predeclared independent-optimality objective, all repeats and paired groups."""
    expected = {(c["file"],s,r) for c in cases for s in STRATEGIES for r in range(REPEATS)}
    actual = [(r["instance"],r["strategy"],r["repeat"]) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("incomplete or duplicate public outcomes")
    mismatches = []
    for c in cases:
        group = [r for r in rows if r["instance"]==c["file"] and r.get("verified")]
        vals = [float(r["objective"]) for r in group]
        ref = c.get("reference")
        if vals and any(abs(v-vals[0]) > 1e-6*max(1,abs(vals[0])) for v in vals):
            mismatches.append(c["file"]+":cross_solver")
        if ref is not None and any(abs(v-ref) > 1e-6*max(1,abs(ref)) for v in vals):
            mismatches.append(c["file"]+":published_reference")
    def success(r):
        return (r.get("verified") is True and r["wall_s"] <= DEADLINE
                and r.get("api_wall_s",float("inf")) <= CUTOFF)
    costs, successes = {}, {}
    for s in STRATEGIES:
        costs[s], successes[s] = [], []
        for c in cases:
            rr = [r for r in rows if r["instance"]==c["file"] and r["strategy"]==s]
            costs[s].append(float(np.mean([r["wall_s"] if success(r) else 10*DEADLINE for r in rr])))
            successes[s].append(sum(success(r) for r in rr))
    policy = np.asarray(costs["learned"])
    comparisons = {}
    groups = sorted({c["group"] for c in cases})
    indices = [np.asarray([i for i,c in enumerate(cases) if c["group"]==g]) for g in groups]
    for s in STRATEGIES[1:]:
        baseline = np.asarray(costs[s]); rng=np.random.default_rng(271828)
        boot=[]
        for _ in range(2000):
            idx=np.concatenate([indices[j] for j in rng.integers(0,len(groups),len(groups))])
            boot.append(float(policy[idx].sum()/baseline[idx].sum()))
        comparisons[s]={"ratio":float(policy.sum()/baseline.sum()),
            "bootstrap_95pct":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
            "p90_ratio":float(np.quantile(policy/baseline,.9)),
            "no_lost_verified_solves":all(a>=b for a,b in zip(successes["learned"],successes[s]))}
    gates={"complete_outcomes":True,"no_verified_objective_mismatch":not mismatches,
           "at_least_20_groups":len(groups)>=20,
           "some_in_support_decisions":any(r.get("decision",{}).get("reason")=="learned_stump" for r in rows)}
    for s in ("default","production"):
        x=comparisons[s]
        gates.update({f"{s}_gain_at_least_3_percent":x["ratio"]<=.97,
                      f"{s}_bootstrap_upper_below_one":x["bootstrap_95pct"][1]<1,
                      f"{s}_p90_at_most_1_25":x["p90_ratio"]<=1.25,
                      f"{s}_no_lost_verified_solves":x["no_lost_verified_solves"]})
    return {"instances":len(cases),"groups":len(groups),"outcomes":len(rows),
        "cost":"mean across all repeats; independently verified within 2s API / 12s controller deadline costs full process wall; otherwise PAR10=120s",
        "mean_par10_s":{s:float(np.mean(v)) for s,v in costs.items()},
        "verified_within_budget":{s:sum(v) for s,v in successes.items()},
        "independently_verified_total":{s:sum(r.get('verified') is True for r in rows if r['strategy']==s) for s in STRATEGIES},
        "comparisons":comparisons,"decision_reasons":dict(Counter(r.get("decision",{}).get("reason","missing") for r in rows if r["strategy"]=="learned")),
        "objective_mismatches":mismatches,"gates":gates,"public_gate_passed":all(gates.values()),
        "automatic_production_routing_enabled":False}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus",type=Path,required=True)
    ap.add_argument("--model",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    if os.name != "posix" or not hasattr(os,"sched_setaffinity"):
        raise SystemExit("this qualification runner requires Linux process groups and CPU affinity")
    args.output.mkdir(parents=True,exist_ok=False)
    cohort=json.loads((args.corpus/"cohort.json").read_text()); cases=cohort["selected"]
    if len(cases)!=24 or len({c["data_hash"] for c in cases})!=24:
        raise ValueError("expected 24 distinct public models")
    worker=Path(__file__).with_name("public_lp_worker.py")
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ); env.update(thread_environment(1))
    # Restrict actual compute to one logical CPU, including all descendant threads.
    cpu=min(os.sched_getaffinity(0)); os.sched_setaffinity(0,{cpu})
    for c in cases:
        if sha(args.corpus/c["file"]) != c["mps_sha256"]:
            raise ValueError("corpus bytes changed")
    protocol={"schema":"solverpilot.public-lp-qualification.v1","captured_at":datetime.now(timezone.utc).isoformat(),
        "environment":capture_environment(packages=("numpy","scipy","highspy","ortools")),
        "cohort_sha256":sha(args.corpus/"cohort.json"),"model_file_sha256":sha(args.model),
        "model_payload_sha256":json.loads(args.model.read_text())["sha256"],
        "source_hashes":{str(p.relative_to(root)):sha(p) for p in
            (Path(__file__).resolve(),worker,root/"src/solverpilot/experimental/learned_lp.py",
             root/"src/solverpilot/experimental/lp_environment.py")},
        "strategies":STRATEGIES,"repeats":REPEATS,"api_budget_s":CUTOFF,"controller_deadline_s":DEADLINE,
        "affinity_cpu":cpu,"parallel_solves":1,"retuning_allowed":False,
        "gate":"24 fresh models; >=20 groups; some learned switches; zero certificate objective mismatch; >=3% gain, bootstrap upper<1, p90<=1.25 and no lost solves against both default and production",
        "scope":"frozen synthetic-trained policy; public OOD challenge; real isolated routed calls; no retraining or promotion from a result alone"}
    write(args.output/"protocol.json",protocol)
    write(args.output/"cohort.json",cohort)
    rows=[]
    for i,c in enumerate(cases):
        for repeat in range(REPEATS):
            shift=i%len(STRATEGIES); order=list(STRATEGIES[shift:]+STRATEGIES[:shift])
            if repeat:order.reverse()
            for s in order:
                cmd=[sys.executable,str(worker),"--mps",str(args.corpus/c["file"]),"--strategy",s,
                     "--model",str(args.model),"--cutoff",str(CUTOFF)]
                r=run_process(cmd,env,DEADLINE)
                if "data_hash" in r and r["data_hash"]!=c["data_hash"]:
                    raise RuntimeError("worker model identity mismatch")
                rows.append({"instance":c["file"],"source":c["source"],"group":c["group"],
                             "strategy":s,"repeat":repeat,**r})
                write(args.output/"observations.json",rows)
        print(json.dumps({"completed":c["file"],"outcomes":len(rows)}),flush=True)
    report=summarize(rows,cases); write(args.output/"summary.json",report)
    stress=[]
    for case in ("infeasible","unbounded","ill_scaled","deadline"):
        for s in STRATEGIES:
            cmd=[sys.executable,str(worker),"--stress",case,"--strategy",s,"--model",str(args.model),
                 "--cutoff",str(.0001 if case=="deadline" else CUTOFF)]
            stress.append({"case":case,"strategy":s,**run_process(cmd,env,DEADLINE)})
            write(args.output/"stress.json",stress)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
