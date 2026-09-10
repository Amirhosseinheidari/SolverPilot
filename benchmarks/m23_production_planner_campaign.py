from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np

from solverpilot import (
    LinearProblem, QuadraticProblem, VariableDomain, SolveIntent,
    plan_production_solve, solve_production,
)
from solverpilot.experimental import M22_OFFICIAL_EVIDENCE, production_evidence_from_m22_gate
from solverpilot.runtime import default_registry


def percentile(xs, p):
    return float(np.percentile(np.asarray(xs, dtype=float), p)) if xs else None


def make_lp(rng, n=8, m=5):
    x0=rng.uniform(0.1,1.0,size=n)
    A=rng.normal(size=(m,n))
    slack=rng.uniform(0.5,2.0,size=m)
    upper=A@x0+slack
    return LinearProblem.from_data(
        A=A,c=rng.normal(size=n),variable_lower=np.zeros(n),variable_upper=np.full(n,3.0),
        constraint_lower=np.full(m,-np.inf),constraint_upper=upper,
    )


def make_milp(rng,n=7,m=4):
    A=rng.integers(-3,4,size=(m,n)).astype(float)
    x0=rng.integers(0,2,size=n).astype(float)
    upper=A@x0+rng.integers(1,5,size=m)
    return LinearProblem.from_data(
        A=A,c=rng.integers(-4,5,size=n).astype(float),variable_lower=np.zeros(n),variable_upper=np.ones(n),
        constraint_lower=np.full(m,-np.inf),constraint_upper=upper,
        domains=[VariableDomain.BINARY]*n,
    )


def brute_milp(problem):
    n=problem.n_variables
    best=None
    for mask in range(1<<n):
        x=np.array([(mask>>j)&1 for j in range(n)],dtype=float)
        ax=problem.A@x
        if np.any(ax < problem.constraint_lower-1e-9) or np.any(ax > problem.constraint_upper+1e-9):
            continue
        obj=float(problem.c@x)+problem.objective_offset
        if best is None or obj<best: best=obj
    return best


def make_qp(rng,n=6):
    target=rng.uniform(0.2,2.0,size=n)
    # 0.5*x'2I*x - 2t'x => ||x-t||^2 - const; unique optimum target.
    return QuadraticProblem.from_data(
        P=np.eye(n)*2.0,q=-2.0*target,A=np.empty((0,n)),
        variable_lower=np.zeros(n),variable_upper=np.full(n,3.0),
        constraint_lower=np.empty(0),constraint_upper=np.empty(0),
    ),target


def audit_m22(gate_dir: Path):
    gate=json.loads((gate_dir/'M22-FINALIZATION-GATE.json').read_text())
    ref=json.loads((gate_dir/'M22-REFERENCE-CROSSCHECK.json').read_text())
    parsed=production_evidence_from_m22_gate(gate, source=str(gate_dir/'M22-FINALIZATION-GATE.json'))
    assert gate['gate_passed'] is True
    assert gate['integrity']['miplib_sha256_match'] is True
    assert gate['integrity']['qplib_sha256_match'] is True
    assert gate['integrity']['pace_official_sha1_match'] is True
    assert gate['campaign_summary']['miplib']['reference_mismatches']==0
    assert gate['campaign_summary']['qplib']['reference_mismatches']==0
    assert gate['campaign_summary']['pace']['invalid_solutions']==0
    return {
        'gate_passed':True,
        'official_integrity':True,
        'fixed_host':gate['fixed_host'],
        'repetitions':gate['protocol']['repetitions'],
        'performance_superiority_claim':gate['protocol']['performance_superiority_claim'],
        'miplib_reference_crosscheck':ref['miplib']['reference_match'],
        'qplib_reference_crosscheck':ref['qplib']['meets_or_beats_best_known'],
        'm23_interpretation':'corpus_validated correctness/robustness evidence; not comparative held-out solver-ranking evidence',
        'parsed_evidence_class':parsed.evidence_class.value,
        'performance_auto_enable':parsed.supports_performance_ranking,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--m22-gate-dir',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=23092026)
    ap.add_argument('--planning-cases',type=int,default=900)
    ap.add_argument('--solve-cases-per-family',type=int,default=30)
    a=ap.parse_args()
    rng=np.random.default_rng(a.seed)
    reg=default_registry()
    m22=audit_m22(a.m22_gate_dir)

    planning_times=[]; routing_counts={}; planning_failures=[]
    families=[]
    for i in range(a.planning_cases):
        family=('lp','milp','qp')[i%3]
        if family=='lp': p=make_lp(rng)
        elif family=='milp': p=make_milp(rng)
        else: p,_=make_qp(rng)
        t=perf_counter()
        try:
            d=plan_production_solve(p,reg,intent=SolveIntent.BALANCED)
        except Exception as exc:
            planning_failures.append({'case':i,'family':family,'error':type(exc).__name__,'reason':str(exc)})
            continue
        planning_times.append(perf_counter()-t)
        routing_counts[d.plan.selected_backend]=routing_counts.get(d.plan.selected_backend,0)+1
        families.append((family,d.plan.selected_backend,d.auto_performance_ranking_enabled))

    solve_rows=[]; solve_failures=[]
    for family in ('lp','milp','qp'):
        for i in range(a.solve_cases_per_family):
            if family=='lp':
                p=make_lp(rng)
                reference=None
            elif family=='milp':
                p=make_milp(rng)
                reference=brute_milp(p)
            else:
                p,target=make_qp(rng)
                reference=target
            t=perf_counter()
            try:
                result,decision=solve_production(p,intent=SolveIntent.BALANCED)
                wall=perf_counter()-t
                valid=bool(result.validation and result.validation.valid)
                row={'family':family,'case':i,'backend':decision.plan.selected_backend,'status':result.status.value,'valid':valid,'wall_s':wall,'objective':result.objective}
                if family=='milp':
                    row['bruteforce_objective']=reference
                    row['objective_match']=reference is not None and result.objective is not None and abs(float(result.objective)-float(reference))<=1e-7
                elif family=='qp':
                    row['target_linf_error']=float(np.max(np.abs(result.x-reference))) if result.x is not None else math.inf
                    row['target_match']=row['target_linf_error']<=2e-5
                solve_rows.append(row)
                if not valid or (family=='milp' and not row['objective_match']) or (family=='qp' and not row['target_match']):
                    solve_failures.append(row)
            except Exception as exc:
                solve_failures.append({'family':family,'case':i,'error':type(exc).__name__,'reason':str(exc)})

    # Explicit guard: M22 evidence must reject a speed override.
    guard=plan_production_solve(
        make_lp(rng),reg,performance_override={'lp':'scipy-highs-ipm'},performance_policy='allow_comparative'
    )
    assert guard.plan.selected_backend=='scipy-highs-ds'
    assert guard.rejected_performance_override_reason

    payload={
      'schema':'optimind.m23.production_planner_campaign.v1',
      'seed':a.seed,
      'm22_evidence_audit':m22,
      'planning':{
        'cases':a.planning_cases,'completed':len(planning_times),'failures':planning_failures,
        'routing_counts':routing_counts,'performance_auto_enabled_count':sum(int(x[2]) for x in families),
        'median_plan_s':statistics.median(planning_times) if planning_times else None,
        'p95_plan_s':percentile(planning_times,95),'max_plan_s':max(planning_times) if planning_times else None,
      },
      'solve_validation':{
        'cases_per_family':a.solve_cases_per_family,'total':3*a.solve_cases_per_family,
        'completed':len(solve_rows),'failures':solve_failures,
        'valid_count':sum(r.get('valid') is True for r in solve_rows),
        'milp_bruteforce_matches':sum(r.get('objective_match') is True for r in solve_rows if r['family']=='milp'),
        'qp_target_matches':sum(r.get('target_match') is True for r in solve_rows if r['family']=='qp'),
        'rows':solve_rows,
      },
      'performance_override_guard':{
        'requested':'scipy-highs-ipm','selected':guard.plan.selected_backend,
        'rejected_reason':guard.rejected_performance_override_reason,
      },
      'claims_boundary':[
        'M23 production planner does not claim learned or empirical solver-speed selection from M22 evidence.',
        'M22 is classified as corpus-validation evidence because it did not compare multiple backends on held-out instances with feature-cost accounting.',
        'Performance overrides require comparative held-out evidence and must still pass capability, intent, availability and health gates.',
      ],
      'passed':not planning_failures and not solve_failures and m22['performance_auto_enable'] is False,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(payload,indent=2,sort_keys=True))
    print(json.dumps({k:v for k,v in payload.items() if k not in ('solve_validation',)},indent=2,default=str)[:7000])
    print('solve failures',len(solve_failures),'passed',payload['passed'])

if __name__=='__main__': main()
