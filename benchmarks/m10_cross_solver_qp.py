from __future__ import annotations
import json, os, platform, sys
from pathlib import Path
from statistics import median
from time import perf_counter
import numpy as np, scipy, nlopt, casadi
from solverpilot.backends import CasadiOSQPBridgeBackend,CasadiHighsBridgeBackend,NLoptNativeBackend,ScipySLSQPQPBackend
from solverpilot.runtime import execute, execute_portfolio
from m9_cross_library_qp import make_case

OUT=Path(__file__).parent/'results'/'m10-cross-solver-qp.json'
SIZES=(5,20,50); FAMILIES=('interior','equality','upper_active','box_active'); SEEDS=4; ROUNDS=3

def main():
    backends={
      'osqp':CasadiOSQPBridgeBackend(enable_primal_warm_start=False),
      'highs':CasadiHighsBridgeBackend(),
      'scipy_slsqp':ScipySLSQPQPBackend(maxiter=2000,ftol=1e-11),
      'nlopt_slsqp':NLoptNativeBackend(max_evals=4000,ftol_rel=1e-11,xtol_rel=1e-11,constraint_tol=1e-10),
    }
    warm,_,_=make_case(5,'interior',991)
    for b in backends.values(): execute(warm,b)
    rows=[]; start=perf_counter()
    for n in SIZES:
      for fi,fam in enumerate(FAMILIES):
       for ls in range(SEEDS):
        seed=101000+n*100+fi*10+ls; p,xs,os_=make_case(n,fam,seed)
        times={k:[] for k in backends}; last={}
        names=list(backends)
        for rr in range(ROUNDS):
          order=names[rr%len(names):]+names[:rr%len(names)]
          for name in order:
            t=perf_counter(); r=execute(p,backends[name]); times[name].append(perf_counter()-t); last[name]=r
        row={'n':n,'family':fam,'seed':seed,'reference_objective':os_}
        for name in names:
          r=last[name]
          row[f'{name}_status']=r.status.value
          row[f'{name}_valid']=bool(r.validation and r.validation.valid)
          row[f'{name}_objective_abs_error']=None if r.objective is None else float(abs(r.objective-os_))
          row[f'{name}_x_inf_error']=None if r.x is None else float(np.max(np.abs(r.x-xs)))
          row[f'{name}_median_s']=float(median(times[name])); row[f'{name}_samples_s']=times[name]
        pf=execute_portfolio(p,list(backends.values()),stop_on_valid_optimal=False)
        row['portfolio_valid']=bool(pf.chosen and pf.chosen.validation and pf.chosen.validation.valid)
        row['portfolio_attempts']=len(pf.attempts); row['portfolio_chosen_backend']=pf.chosen_backend
        rows.append(row)
    names=list(backends)
    failures={n:sum(not r[f'{n}_valid'] for r in rows) for n in names}
    max_obj={n:max(float(r[f'{n}_objective_abs_error'] or 0) for r in rows) for n in names}
    max_x={n:max(float(r[f'{n}_x_inf_error'] or 0) for r in rows) for n in names}
    wins={n:0 for n in names}; wins['ties']=0
    for r in rows:
      vals={n:float(r[f'{n}_median_s']) for n in names}; mv=min(vals.values()); winners=[n for n,v in vals.items() if abs(v-mv)<=1e-9]
      if len(winners)==1:wins[winners[0]]+=1
      else:wins['ties']+=1
    payload={'benchmark':'M10 cross-solver convex QP','instances':len(rows),'rounds':ROUNDS,'direct_solver_calls':len(rows)*ROUNDS*len(names),'portfolio_executions':len(rows),'portfolio_backend_attempts':len(rows)*len(names),'failures':failures,'max_objective_abs_error':max_obj,'max_x_inf_error':max_x,'timing_wins':wins,'portfolio_valid_count':sum(r['portfolio_valid'] for r in rows),'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'nlopt':nlopt.__version__,'casadi':casadi.__version__,'platform':platform.platform(),'cpu_count':os.cpu_count()},'wall_s':perf_counter()-start,'rows':rows}
    OUT.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print(json.dumps({k:payload[k] for k in ['instances','direct_solver_calls','failures','max_objective_abs_error','timing_wins','portfolio_valid_count','wall_s']},indent=2))
if __name__=='__main__':main()
