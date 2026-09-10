from __future__ import annotations
import argparse,json,pickle
from pathlib import Path
from time import perf_counter
from solverpilot import execute
from solverpilot.backends import ScipyHighsLPBackend
from solverpilot.validate import PublicStatus

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--problem',type=Path,required=True); ap.add_argument('--method',required=True,choices=('highs-ds','highs-ipm')); ap.add_argument('--cutoff-s',type=float,default=1.0); args=ap.parse_args()
 p=pickle.loads(args.problem.read_bytes()); b=ScipyHighsLPBackend(method=args.method,time_limit_s=args.cutoff_s)
 t0=perf_counter(); r=execute(p,b); solve_s=perf_counter()-t0
 terminal=r.status in {PublicStatus.VALID_OPTIMAL,PublicStatus.INFEASIBLE,PublicStatus.UNBOUNDED}; valid=(r.validation is not None and r.validation.valid) if r.x is not None else terminal
 print(json.dumps({'solve_s':solve_s,'status':r.status.value,'objective':None if r.objective is None else float(r.objective),'terminal':terminal,'valid':bool(valid)}))
if __name__=='__main__': main()
