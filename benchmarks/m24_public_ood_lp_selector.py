from __future__ import annotations

import argparse, gzip, hashlib, json, math, os, platform, statistics, sys, zipfile
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy

from solverpilot import LinearProblem, VariableDomain, execute, inspect_problem, parse_mps
from solverpilot.backends import ScipyHighsLPBackend
from solverpilot.validate import PublicStatus

RULE_THRESHOLD = 1.5  # frozen in M4/M5 before public OOD evaluation
RULE_NAME = "ipm_if_m_over_n_ge_1.50"
CUTOFF_S = 0.75
TARGET_INSTANCES = 24
MAX_COMPRESSED_BYTES = 256 * 1024
MAX_N = 2000
MAX_M = 2000
MAX_NNZ = 80_000
ROUNDS = 3


def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()


def _filename_key(name: str) -> str:
    return hashlib.sha256(name.encode('utf-8')).hexdigest()


def _relax(p: LinearProblem, name: str) -> LinearProblem:
    return LinearProblem.from_data(
        A=p.A, c=p.c,
        variable_lower=p.variable_lower, variable_upper=p.variable_upper,
        constraint_lower=p.constraint_lower, constraint_upper=p.constraint_upper,
        domains=[VariableDomain.CONTINUOUS] * p.n_variables,
        objective_sense=p.objective_sense, objective_offset=p.objective_offset,
        name=f"{name}-lp-relaxation",
        metadata={**p.metadata, "m24_source_instance":name, "m24_relaxation":True},
    )


def _run(p: LinearProblem, method: str):
    b=ScipyHighsLPBackend(method=method,time_limit_s=CUTOFF_S)
    t0=perf_counter(); r=execute(p,b); wall=perf_counter()-t0
    terminal = r.status in {PublicStatus.VALID_OPTIMAL, PublicStatus.INFEASIBLE, PublicStatus.UNBOUNDED}
    valid = r.validation is not None and r.validation.valid if r.x is not None else terminal
    return r, wall, terminal, bool(valid)


def _cost(sample):
    return min(float(sample['wall_s']), CUTOFF_S) if sample['terminal'] else CUTOFF_S


def _bootstrap(policy, sbs, draws=10000):
    rng=np.random.default_rng(240024)
    p=np.asarray(policy,float); s=np.asarray(sbs,float); n=len(p)
    vals=np.empty(draws)
    for i in range(draws):
        idx=rng.integers(0,n,size=n); vals[i]=p[idx].mean()/s[idx].mean()
    q=np.quantile(vals,[.025,.5,.975])
    return {'point':float(p.mean()/s.mean()),'low':float(q[0]),'median':float(q[1]),'high':float(q[2]),'draws':draws}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--miplib-zip',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--m22-integrity',type=Path)
    args=ap.parse_args()
    names=[x.strip() for x in args.manifest.read_text(encoding='utf-8').splitlines() if x.strip() and not x.lstrip().startswith('#')]
    zip_sha=_sha256(args.miplib_zip)
    integrity_verified=False; integrity_source=None
    if args.m22_integrity is not None:
        integ=json.loads(args.m22_integrity.read_text(encoding='utf-8'))
        mip=(integ.get('archives') or {}).get('miplib') if isinstance(integ,dict) else None
        integrity_verified=bool(integ.get('all_integrity_pass')) and isinstance(mip,dict) and mip.get('sha256_match') is True and mip.get('sha256')==zip_sha
        integrity_source=str(args.m22_integrity.resolve())
    with zipfile.ZipFile(args.miplib_zip) as z:
        info={Path(i.filename).name:i for i in z.infolist() if i.filename.endswith('.mps.gz')}
        pre=[n for n in names if n in info and info[n].compress_size <= MAX_COMPRESSED_BYTES]
        pre=sorted(pre,key=_filename_key)
        parse_exclusions=[]; structural_exclusions=[]; cohort=[]
        for name in pre:
            if len(cohort)>=TARGET_INSTANCES: break
            i=info[name]
            try:
                raw=z.read(i); text=gzip.decompress(raw).decode('utf-8',errors='strict'); original=parse_mps(text); p=_relax(original,Path(name).name.removesuffix('.mps.gz'))
            except Exception as exc:
                parse_exclusions.append({'instance':name,'error':f'{type(exc).__name__}: {exc}'})
                continue
            if p.n_variables>MAX_N or p.n_constraints>MAX_M or p.nnz>MAX_NNZ:
                structural_exclusions.append({'instance':name,'n':p.n_variables,'m':p.n_constraints,'nnz':p.nnz})
                continue
            cohort.append((name,p))

    # warm outside measurements
    warm=LinearProblem.from_data(A=[[1.,1.]],c=[1.,2.],variable_lower=[0.,0.],variable_upper=[1.,1.],constraint_lower=[1.],constraint_upper=[np.inf])
    for m in ('highs-ds','highs-ipm'): _run(warm,m)

    rows=[]; started=perf_counter()
    for idx,(name,p) in enumerate(cohort):
        fp_samples=[]; samples={'highs-ds':[],'highs-ipm':[]}
        for round_idx in range(ROUNDS):
            t0=perf_counter(); fp=inspect_problem(p); fp_samples.append(perf_counter()-t0)
            order=['highs-ds','highs-ipm']
            if (idx+round_idx)%2: order.reverse()
            for method in order:
                r,wall,terminal,valid=_run(p,method)
                samples[method].append({'wall_s':wall,'status':r.status.value,'objective':None if r.objective is None else float(r.objective),'terminal':terminal,'valid':valid})
        ds=samples['highs-ds']; ip=samples['highs-ipm']
        ds_cost=float(statistics.median(_cost(x) for x in ds)); ip_cost=float(statistics.median(_cost(x) for x in ip))
        inspect_cost=float(statistics.median(fp_samples))
        ds_opt=[x['objective'] for x in ds if x['status']=='valid_optimal' and x['objective'] is not None]
        ip_opt=[x['objective'] for x in ip if x['status']=='valid_optimal' and x['objective'] is not None]
        obj_diff=None
        if ds_opt and ip_opt: obj_diff=abs(float(statistics.median(ds_opt))-float(statistics.median(ip_opt)))
        ratio=p.n_constraints/max(p.n_variables,1)
        choice='highs-ipm' if ratio>=RULE_THRESHOLD else 'highs-ds'
        rows.append({
            'instance':name,'n':p.n_variables,'m':p.n_constraints,'nnz':p.nnz,'m_over_n':ratio,
            'inspect_median_s':inspect_cost,'ds_median_cost_s':ds_cost,'ipm_median_cost_s':ip_cost,
            'policy_choice':choice,'policy_cost_s':(ip_cost if choice.endswith('ipm') else ds_cost)+inspect_cost,
            'ds_terminal_rounds':sum(x['terminal'] for x in ds),'ipm_terminal_rounds':sum(x['terminal'] for x in ip),
            'ds_valid_rounds':sum(x['valid'] for x in ds),'ipm_valid_rounds':sum(x['valid'] for x in ip),
            'objective_abs_diff':obj_diff,'samples':samples,
        })

    ds=np.array([r['ds_median_cost_s'] for r in rows]); ip=np.array([r['ipm_median_cost_s'] for r in rows])
    means={'scipy-highs-ds':float(ds.mean()),'scipy-highs-ipm':float(ip.mean())}
    sbs_name=min(means,key=means.get); sbs=ds if sbs_name.endswith('ds') else ip
    vbs=np.minimum(ds,ip); policy=np.array([r['policy_cost_s'] for r in rows])
    gap=float((sbs.mean()-policy.mean())/(sbs.mean()-vbs.mean())) if sbs.mean()>vbs.mean()+1e-15 else 0.0
    boot=_bootstrap(policy,sbs)
    objective_mismatches=sum(r['objective_abs_diff'] is not None and r['objective_abs_diff']>1e-6*max(1.0,abs(next((x['objective'] for x in r['samples']['highs-ds'] if x['objective'] is not None),0.0))) for r in rows)
    ds_terminal=sum(r['ds_terminal_rounds'] for r in rows); ip_terminal=sum(r['ipm_terminal_rounds'] for r in rows); total=len(rows)*ROUNDS
    gates={
        'minimum_instances':len(rows)>=24,
        'ds_terminal_rate_ge_0_90': ds_terminal/max(total,1)>=0.90,
        'ipm_terminal_rate_ge_0_90': ip_terminal/max(total,1)>=0.90,
        'objective_mismatches_zero':objective_mismatches==0,
        'policy_beats_sbs_point':boot['point']<1.0,
        'bootstrap_upper_below_1':boot['high']<1.0,
        'gap_closure_ge_0_50':gap>=0.50,
    }
    promote=all(gates.values())
    payload={
        'schema':'optimind.m24.public_ood_lp_selector.v1','source':'MIPLIB 2017 benchmark-v2 derived continuous LP relaxations',
        'miplib_zip_sha256':zip_sha,'manifest_sha256':_sha256(args.manifest),'official_corpus_integrity_passed':integrity_verified,'integrity_source':integrity_source,
        'pre_registered_rule':{'name':RULE_NAME,'threshold':RULE_THRESHOLD,'origin':'M4/M5 synthetic LOFO/stability; frozen before M24 public OOD run'},
        'selection_protocol':{'outcome_independent':True,'compressed_size_cap_bytes':MAX_COMPRESSED_BYTES,'hash_order':'sha256(filename)','target_instances':TARGET_INSTANCES,'max_n':MAX_N,'max_m':MAX_M,'max_nnz':MAX_NNZ},
        'instances':len(rows),'parse_exclusions':parse_exclusions,'structural_exclusions':structural_exclusions,'rounds':ROUNDS,'cutoff_s':CUTOFF_S,
        'metrics':{'sbs_backend':sbs_name,'sbs_mean_s':float(sbs.mean()),'vbs_mean_s':float(vbs.mean()),'policy_mean_s':float(policy.mean()),'gap_closure':gap,'policy_to_sbs_bootstrap':boot,'ds_wins':int(np.sum(ds<ip)),'ipm_wins':int(np.sum(ip<ds)),'ties':int(np.sum(ds==ip)),'ds_terminal_rate':ds_terminal/max(total,1),'ipm_terminal_rate':ip_terminal/max(total,1),'objective_mismatches':objective_mismatches},
        'promotion_gates':gates,'selector_promoted':promote,
        'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),'cpu_count':os.cpu_count()},
        'wall_s':perf_counter()-started,'rows':rows,
        'claims_boundary':['public/OOD evidence uses continuous LP relaxations derived from official MIPLIB 2017 MILP models','this is within-HiGHS DS-vs-IPM algorithm selection, not cross-solver superiority','feature inspection cost is included in deployable policy cost','promotion requires all predeclared gates; failure leaves M23 conservative baseline unchanged'],
    }
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps({'instances':len(rows),'metrics':payload['metrics'],'promotion_gates':gates,'selector_promoted':promote,'parse_exclusions':len(parse_exclusions),'structural_exclusions':len(structural_exclusions),'wall_s':payload['wall_s']},indent=2))

if __name__=='__main__': main()
