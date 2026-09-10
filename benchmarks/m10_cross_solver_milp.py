from __future__ import annotations
import itertools,json,os,platform,sys
from pathlib import Path
from statistics import median
from time import perf_counter
import numpy as np, scipy, casadi
from solverpilot import LinearProblem,ObjectiveSense,VariableDomain,read_mps
from solverpilot.backends import ScipyHighsBackend,CasadiHighsBridgeBackend,CasadiCBCBridgeBackend
from solverpilot.runtime import execute
OUT=Path(__file__).parent/'results'/'m10-cross-solver-milp.json'
ROUNDS=3; CASES=100

def brute(p):
    best=None; bestx=None
    for bits in itertools.product((0.,1.),repeat=p.n_variables):
        x=np.array(bits); ax=p.A@x
        if np.any(ax < p.constraint_lower-1e-10) or np.any(ax > p.constraint_upper+1e-10): continue
        v=float(p.c@x+p.objective_offset)
        if best is None or (p.objective_sense is ObjectiveSense.MAXIMIZE and v>best) or (p.objective_sense is ObjectiveSense.MINIMIZE and v<best):best=v;bestx=x
    return best,bestx

def make(seed):
    rng=np.random.default_rng(seed); n=int(rng.integers(6,13)); m=int(rng.integers(3,8))
    A=rng.integers(0,8,size=(m,n)).astype(float); c=rng.integers(1,30,size=n).astype(float)
    # feasible packing model with nontrivial capacities
    caps=np.maximum(1,np.floor(A.sum(axis=1)*rng.uniform(.25,.55,size=m)))
    p=LinearProblem.from_data(A=A,c=c,variable_lower=np.zeros(n),variable_upper=np.ones(n),constraint_lower=np.full(m,-np.inf),constraint_upper=caps,domains=[VariableDomain.BINARY]*n,objective_sense=ObjectiveSense.MAXIMIZE,metadata={'seed':seed})
    return p

def run_problem(p,backends,rounds=ROUNDS):
    times={k:[] for k in backends}; last={}
    for rr in range(rounds):
      names=list(backends); names=names[rr%len(names):]+names[:rr%len(names)]
      for name in names:
        t=perf_counter(); r=execute(p,backends[name]); times[name].append(perf_counter()-t); last[name]=r
    return {name:{'status':last[name].status.value,'valid':bool(last[name].validation and last[name].validation.valid),'objective':last[name].objective,'median_s':float(median(times[name])),'samples_s':times[name]} for name in backends}

def main():
    backends={'scipy_highs':ScipyHighsBackend(),'casadi_highs':CasadiHighsBridgeBackend(),'cbc':CasadiCBCBridgeBackend()}
    rows=[];start=perf_counter()
    for i in range(CASES):
      p=make(120000+i); ref,_=brute(p); sols=run_problem(p,backends)
      row={'seed':120000+i,'n':p.n_variables,'m':p.n_constraints,'reference_objective':ref,'solvers':sols}
      for v in sols.values(): v['objective_abs_error']=None if v['objective'] is None or ref is None else float(abs(v['objective']-ref))
      rows.append(row)
    public_path=Path(__file__).parent/'public'/'p0033.mps'
    if public_path.exists():
      pp=read_mps(public_path); psol=run_problem(pp,backends,rounds=3)
      public={'state':'executed','name':'p0033','source':'COIN-OR Data-Sample / Crowder-Johnson-Padberg','sha256':'83e66c91887524b9f4db408c50cd8fe4ff2154374c7e28ac28fc3fd18aa49db3','reference_objective':3089.0,'solvers':psol}
      for v in psol.values():v['objective_abs_error']=None if v['objective'] is None else float(abs(v['objective']-3089.0))
      public_calls=3*len(backends)
    else:
      public={'state':'not_executed_missing_third_party_instance','name':'p0033','source':'COIN-OR Data-Sample / Crowder-Johnson-Padberg','expected_sha256':'83e66c91887524b9f4db408c50cd8fe4ff2154374c7e28ac28fc3fd18aa49db3','reference_objective':3089.0,'path':str(public_path)}
      public_calls=0
    failures={name:sum(not row['solvers'][name]['valid'] for row in rows) for name in backends}
    maxerr={name:max(float(row['solvers'][name]['objective_abs_error'] or 0) for row in rows) for name in backends}
    wins={name:0 for name in backends};wins['ties']=0
    for row in rows:
      vals={n:row['solvers'][n]['median_s'] for n in backends}; mv=min(vals.values()); ws=[n for n,v in vals.items() if abs(v-mv)<=1e-9]
      wins[ws[0] if len(ws)==1 else 'ties']+=1
    payload={'benchmark':'M10 cross-solver binary MILP exact-bruteforce + public p0033','synthetic_instances':len(rows),'rounds':ROUNDS,'direct_solver_calls':len(rows)*ROUNDS*len(backends)+public_calls,'failures':failures,'max_objective_abs_error_vs_bruteforce':maxerr,'timing_wins':wins,'public_p0033':public,'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'casadi':casadi.__version__,'platform':platform.platform(),'cpu_count':os.cpu_count()},'wall_s':perf_counter()-start,'rows':rows}
    OUT.write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps({k:payload[k] for k in ['synthetic_instances','direct_solver_calls','failures','max_objective_abs_error_vs_bruteforce','timing_wins','public_p0033','wall_s']},indent=2))
if __name__=='__main__':main()
