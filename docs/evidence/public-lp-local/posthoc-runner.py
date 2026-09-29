"""Post-outcome diagnostics only; no model fitting or performance qualification."""
from collections.abc import Mapping
from dataclasses import asdict
import json
import math
from pathlib import Path
import numpy as np
from solverpilot import read_mps,LinearProblem,solve,solve_production,SolveBudget
from solverpilot.backends import PDLPBackend

base=Path('/mnt/d/ARTICLE/solverpilot')
rows=json.loads((base/'public-lp-stage4-evidence/observations.json').read_text())
selected=[]
for label,predicate in [
    ('mip_primal_valid_certificate_unverified',lambda r:r['source']=='miplib' and r['strategy']=='default' and r.get('primal_valid') and not r.get('verified')),
    ('netlib_primal_valid_certificate_unverified',lambda r:r['source']=='netlib' and r['strategy']=='production' and r.get('primal_valid') and not r.get('verified')),
    ('learned_candidate_rejected',lambda r:r['strategy']=='learned' and r['status']=='invalid_solution'),
]:
    row=next((r for r in rows if predicate(r)),None)
    if row:selected.append((label,row))
def clean(v):
    if isinstance(v,Mapping):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,float) and not math.isfinite(v):return str(v)
    return v
out=[]
for label,row in selected:
    p=read_mps(base/'public-lp-stage4-corpus'/row['instance'])
    p=LinearProblem.from_data(A=p.A,c=p.c,variable_lower=p.variable_lower,variable_upper=p.variable_upper,
        constraint_lower=p.constraint_lower,constraint_upper=p.constraint_upper,
        objective_sense=p.objective_sense,objective_offset=p.objective_offset)
    if row['strategy']=='learned':r=solve(p,backend=PDLPBackend(time_limit_s=2,threads=1))
    elif row['strategy']=='production':r,_=solve_production(p,budget=SolveBudget(wall_time_s=2))
    else:r=solve(p,budget=SolveBudget(wall_time_s=2))
    out.append(clean({'label':label,'instance':row['instance'],'original_strategy':row['strategy'],
        'posthoc':True,'budget_s':2,'status':r.status.value,'backend_status':r.backend_status,
        'data_hash':p.data_hash,'infinite_upper_bounds':int(np.isposinf(p.variable_upper).sum()),
        'infinite_lower_bounds':int(np.isneginf(p.variable_lower).sum()),
        'validation':None if r.validation is None else asdict(r.validation),
        'optimality_check':r.raw_statistics.get('optimality_check'),
        'trust':r.raw_statistics.get('solverpilot_trust')}))
print(json.dumps(out,indent=2,allow_nan=False))
(base/'public-lp-stage4-evidence/posthoc-diagnostics.json').write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
