from __future__ import annotations
import itertools,json,time
from pathlib import Path
import numpy as np
import solverpilot as om

rng=np.random.default_rng(20260907)
rows=[]; passed=0; t0=time.perf_counter()
for case in range(300):
    n=int(rng.integers(2,5)); ub=int(rng.integers(2,5)); m=om.CPModel(f'r{case}'); xs=[m.int_var(0,ub,f'x{i}') for i in range(n)]
    specs=[]
    # 1-3 linear constraints
    for _ in range(int(rng.integers(1,4))):
        coeff=rng.integers(-2,3,size=n); coeff[np.all(coeff==0) if False else 0]=coeff[0]
        if not np.any(coeff): coeff[0]=1
        rhs=int(rng.integers(-2,ub*n+2)); sense='le' if rng.random()<.5 else 'ge'
        expr=sum(int(coeff[i])*xs[i] for i in range(n))
        m.add(expr<=rhs if sense=='le' else expr>=rhs); specs.append(('lin',tuple(map(int,coeff)),sense,rhs))
    if n<=ub+1 and rng.random()<.45:
        m.add_all_different(xs); specs.append(('alldiff',))
    # occasional table on first two vars
    if rng.random()<.3:
        allowed=[]
        for a in range(ub+1):
            for b in range(ub+1):
                if (a+b+case)%2==0: allowed.append((a,b))
        m.add_allowed_assignments(xs[:2],allowed); specs.append(('table',tuple(allowed)))
    # occasional element with dedicated index+target (add them to oracle vars)
    extra=[]
    if rng.random()<.3:
        idx=m.int_var(0,2,'idx'); vals=(2,5,7); tgt=m.int_var(0,8,'tgt'); m.add_element(idx,vals,tgt); specs.append(('element',n,n+1,vals)); extra=[idx,tgt]
    allvars=xs+extra
    coeff_obj=[int(v) for v in rng.integers(-3,4,size=len(allvars))]
    obj=sum(coeff_obj[i]*allvars[i] for i in range(len(allvars))); m.minimize(obj)
    r=m.solve(om.ReferenceCPBackend(max_states=1_000_000))
    # independent oracle directly from specs; no SolverPilot validator.
    best=None; besta=None
    domains=[v.domain for v in allvars]
    for vals in itertools.product(*domains):
        a=list(map(int,vals)); ok=True
        for sp in specs:
            if sp[0]=='lin':
                value=sum(sp[1][i]*a[i] for i in range(n)); ok &= value<=sp[3] if sp[2]=='le' else value>=sp[3]
            elif sp[0]=='alldiff': ok &= len(set(a[:n]))==n
            elif sp[0]=='table': ok &= tuple(a[:2]) in set(sp[1])
            elif sp[0]=='element': ok &= a[sp[2]]==sp[3][a[sp[1]]]
            if not ok: break
        if not ok: continue
        val=sum(coeff_obj[i]*a[i] for i in range(len(a)))
        if best is None or val<best: best=val; besta=a
    match=(best is None and r.status=='infeasible' and r.optimality_proven) or (best is not None and r.optimality_proven and r.validation.valid and r.objective==best)
    passed+=int(match); rows.append({'case':case,'match':match,'oracle':best,'solverpilot':r.objective,'status':r.status,'states':r.raw_statistics.get('states_total')})
out={'schema':'solverpilot.p9.cp-random-oracle.v1','seed':20260907,'cases':300,'passed':passed,'wall_s':time.perf_counter()-t0,'rows':rows}
Path('benchmarks/results/p9-cp-random-oracle.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps({'cases':300,'passed':passed,'max_states':max(r['states'] for r in rows),'wall_s':out['wall_s']},indent=2))
