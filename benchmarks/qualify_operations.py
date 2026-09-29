"""Bounded local/CI soak; observed RSS is not a proof of absence of memory leaks."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
from solverpilot import LinearProblem, QuadraticProblem
from solverpilot.runtime.batch_stream import BatchExecutor
from solverpilot.runtime.deadline import solve_with_deadline
from solverpilot.benchmark.environment import capture_environment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    if args.output.exists(): raise SystemExit('refusing to overwrite evidence')
    lp=LinearProblem.from_data(A=[[1.,1.]],c=[1.,2.],variable_lower=[0.,0.],variable_upper=[2.,2.],
                              constraint_lower=[1.],constraint_upper=[np.inf])
    qp=QuadraticProblem.from_data(P=np.eye(2),q=[-1.,-1.],A=[[1.,1.]],
        variable_lower=[0.,0.],variable_upper=[2.,2.],constraint_lower=[0.],constraint_upper=[1.])
    process=psutil.Process(); observations=[];pids=[]
    for name, problem in (('scipy-highs-ds',lp),('osqp-native',qp)):
        with BatchExecutor(backend=name,max_workers=1,max_pending=1,timeout_s=30) as executor:
            for i,item in enumerate(executor.iter(problem for _ in range(35))):
                workers=[p for p in process.children(recursive=True) if p.is_running()]
                pids.extend(p.pid for p in workers)
                observations.append({'backend':name,'iteration':i,'verified':item.independently_verified_optimal,
                    'valid':item.validation_valid,'elapsed_s':item.elapsed_s,
                    'parent_rss':process.memory_info().rss,
                    'children_rss':sum(p.memory_info().rss for p in workers)})
    deadlines=[]
    for budget in (.00001,.1,5.):
        start=perf_counter();item=solve_with_deadline(lp,timeout_s=budget,backend='scipy-highs-ds')
        deadlines.append({'budget_s':budget,'call_wall_s':perf_counter()-start,'status':item.status,
                          'verified':item.independently_verified_optimal})
    # multiprocessing's resource tracker can persist legitimately. Check owned
    # solver workers, not all unrelated processes in the session.
    remaining=[p.pid for p in process.children(recursive=True)
               if p.is_running() and 'spawn_main' in ' '.join(p.cmdline())]
    summary={}
    for name in ('scipy-highs-ds','osqp-native'):
        rows=[r for r in observations if r['backend']==name and r['iteration']>=5]
        rss=[r['parent_rss']+r['children_rss'] for r in rows]
        summary[name]={'calls':35,'verified':sum(r['verified'] for r in observations if r['backend']==name),
                      'post_warmup_rss_range_bytes':max(rss)-min(rss),'last_minus_first_rss_bytes':rss[-1]-rss[0]}
    passed=all(r['valid'] and r['verified'] for r in observations) and not remaining
    payload={'scope':__doc__,'environment':capture_environment(packages=('numpy','scipy','osqp','psutil')),
             'observations':observations,'deadline_observations':deadlines,'summary':summary,
             'remaining_owned_workers':remaining,'functional_gate_passed':passed,
             'memory_leak_absence_claimed':False,'native_factorization_reuse_claimed':False}
    args.output.write_bytes(json.dumps(payload,indent=2,allow_nan=False).encode())
    print(json.dumps({'summary':summary,'functional_gate_passed':passed,'deadlines':deadlines}))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
