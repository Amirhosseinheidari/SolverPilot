from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from scipy import sparse

from solverpilot import LinearProblem, QuadraticProblem
from solverpilot.backends import BundledHighsCAPIBackend, BundledOSQPCAPIBackend
from solverpilot.runtime import execute

OUT = Path(__file__).parent / "results" / "m13-direct-capi-reoptimization.json"
LP_SIZES = {"small": (50, 120, 0.04), "medium": (100, 260, 0.035), "large": (180, 480, 0.025)}
QP_SIZES = (20, 60, 120)
SEEDS = (15001, 15002, 15003)
REPEATS = 3
STEPS = 10


def lp_sequence(seed: int, m: int, n: int, density: float):
    rng=np.random.default_rng(seed)
    A=sparse.random(m,n,density=density,random_state=rng,data_rvs=lambda k:rng.uniform(0.2,2.0,size=k),format='csr')
    A=A.tolil()
    for j in range(n):
        if len(A[:,j].nonzero()[0])==0: A[int(rng.integers(0,m)),j]=float(rng.uniform(0.2,2.0))
    A=A.tocsr(); A.sum_duplicates(); A.eliminate_zeros(); A.sort_indices()
    x0=rng.uniform(0.1,1.0,size=n); c0=rng.normal(size=n); problems=[]
    for step in range(STEPS):
        xf=np.maximum(0.01,x0*(1+rng.normal(scale=0.02,size=n))); b=np.asarray(A@xf).ravel(); c=c0+rng.normal(scale=0.02,size=n)
        problems.append(LinearProblem.from_data(A=A,c=c,variable_lower=np.zeros(n),variable_upper=np.full(n,np.inf),constraint_lower=b,constraint_upper=b))
    return problems


def qp_sequence(seed: int, n: int):
    rng=np.random.default_rng(seed)
    diag=rng.uniform(0.5,2.0,size=n)
    P=sparse.diags(diag,format='csr')
    # Fixed sparse difference-ish constraints plus bounds represented canonically by the problem itself.
    m=max(2,n//3)
    A=sparse.random(m,n,density=min(0.15,5/max(n,1)),random_state=rng,data_rvs=lambda k:rng.uniform(-1.0,1.0,size=k),format='csr')
    A.sum_duplicates(); A.eliminate_zeros(); A.sort_indices()
    x0=rng.uniform(-0.3,0.3,size=n); problems=[]
    for step in range(STEPS):
        xref=x0+rng.normal(scale=0.01,size=n)
        q=-(diag*xref)
        center=np.asarray(A@xref).ravel()
        cl=center-1.0; cu=center+1.0
        # Every third step changes P values but preserves exact sparsity pattern.
        if step and step%3==0:
            scale=1.0+0.01*step
            P_step=sparse.diags(diag*scale,format='csr')
        else:
            P_step=P
        problems.append(QuadraticProblem.from_data(P=P_step,q=q,A=A,variable_lower=np.full(n,-2.0),variable_upper=np.full(n,2.0),constraint_lower=cl,constraint_upper=cu))
    return problems


def run_lp_stateful(problems):
    b=BundledHighsCAPIBackend(solver='simplex',presolve=False,threads=1); rows=[]
    try:
        for p in problems:
            t0=perf_counter(); r=execute(p,b); wall=perf_counter()-t0
            if not (r.validation and r.validation.valid): raise AssertionError(r.status)
            rows.append({"wall_s":wall,"objective":r.objective,"reuse":bool(r.trace.reuse_applied),"mode":r.trace.reuse_mode,"iterations":int(r.raw_statistics['simplex_iteration_count'])})
    finally:b.close()
    return rows


def run_lp_cold(problems):
    rows=[]
    for p in problems:
        b=BundledHighsCAPIBackend(solver='simplex',presolve=False,threads=1)
        try:
            t0=perf_counter(); r=execute(p,b); wall=perf_counter()-t0
            if not (r.validation and r.validation.valid): raise AssertionError(r.status)
            rows.append({"wall_s":wall,"objective":r.objective,"iterations":int(r.raw_statistics['simplex_iteration_count'])})
        finally:b.close()
    return rows


def run_qp_stateful(problems):
    b=BundledOSQPCAPIBackend(); rows=[]
    try:
        for p in problems:
            t0=perf_counter(); r=execute(p,b); wall=perf_counter()-t0
            if not (r.validation and r.validation.valid): raise AssertionError(r.status)
            rows.append({"wall_s":wall,"objective":r.objective,"reuse":bool(r.trace.reuse_applied),"mode":r.trace.reuse_mode,"iter":int(r.raw_statistics['iter']),"matrix_values_updated":bool(r.raw_statistics['matrix_values_updated'])})
    finally:b.close()
    return rows


def run_qp_cold(problems):
    rows=[]
    for p in problems:
        b=BundledOSQPCAPIBackend()
        try:
            t0=perf_counter(); r=execute(p,b); wall=perf_counter()-t0
            if not (r.validation and r.validation.valid): raise AssertionError(r.status)
            rows.append({"wall_s":wall,"objective":r.objective,"iter":int(r.raw_statistics['iter'])})
        finally:b.close()
    return rows


def main():
    lp_runs=[]; qp_runs=[]; started=perf_counter()
    for label,(m,n,density) in LP_SIZES.items():
        for seed in SEEDS:
            problems=lp_sequence(seed,m,n,density)
            for rep in range(REPEATS):
                if (seed+rep)%2: cold=run_lp_cold(problems); state=run_lp_stateful(problems)
                else: state=run_lp_stateful(problems); cold=run_lp_cold(problems)
                maxdiff=max(abs(a['objective']-b['objective']) for a,b in zip(state,cold,strict=True))
                sw=sum(x['wall_s'] for x in state); cw=sum(x['wall_s'] for x in cold)
                lp_runs.append({"size":label,"seed":seed,"repeat":rep,"stateful_wall_s":sw,"cold_wall_s":cw,"ratio":sw/cw,"max_objective_abs_diff":maxdiff,"reuse_steps":sum(x['reuse'] for x in state),"stateful_iterations":sum(x['iterations'] for x in state),"cold_iterations":sum(x['iterations'] for x in cold)})
    for n in QP_SIZES:
        for seed in SEEDS:
            problems=qp_sequence(seed+1000,n)
            for rep in range(REPEATS):
                if (seed+rep)%2: cold=run_qp_cold(problems); state=run_qp_stateful(problems)
                else: state=run_qp_stateful(problems); cold=run_qp_cold(problems)
                maxdiff=max(abs(a['objective']-b['objective']) for a,b in zip(state,cold,strict=True))
                sw=sum(x['wall_s'] for x in state); cw=sum(x['wall_s'] for x in cold)
                qp_runs.append({"n":n,"seed":seed,"repeat":rep,"stateful_wall_s":sw,"cold_wall_s":cw,"ratio":sw/cw,"max_objective_abs_diff":maxdiff,"reuse_steps":sum(x['reuse'] for x in state),"matrix_update_steps":sum(x['matrix_values_updated'] for x in state)})
    lp_summary={}
    for label in LP_SIZES:
        sub=[r for r in lp_runs if r['size']==label]
        lp_summary[label]={"runs":len(sub),"median_ratio":median(r['ratio'] for r in sub),"all_objectives_agree":all(r['max_objective_abs_diff']<=1e-7 for r in sub),"all_expected_reuse":all(r['reuse_steps']==STEPS-1 for r in sub),"median_iteration_ratio":median(r['stateful_iterations']/max(1,r['cold_iterations']) for r in sub)}
    qp_summary={}
    for n in QP_SIZES:
        sub=[r for r in qp_runs if r['n']==n]
        qp_summary[str(n)]={"runs":len(sub),"median_ratio":median(r['ratio'] for r in sub),"all_objectives_agree":all(r['max_objective_abs_diff']<=1e-7 for r in sub),"all_expected_reuse":all(r['reuse_steps']==STEPS-1 for r in sub),"matrix_update_steps_observed":sorted(set(r['matrix_update_steps'] for r in sub))}
    payload={"benchmark":"M13 direct C-API stateful reoptimization","scope":{"highs":"persistent model + numerical update + explicit basis","osqp":"persistent workspace + vector/same-sparsity matrix updates + automatic warm-start; no factorization-reuse claim"},"steps":STEPS,"repeats":REPEATS,"lp":lp_summary,"qp":qp_summary,"lp_runs":lp_runs,"qp_runs":qp_runs,"wall_s":perf_counter()-started}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print(json.dumps({"lp":lp_summary,"qp":qp_summary,"wall_s":payload['wall_s']},indent=2))

if __name__=='__main__':main()
