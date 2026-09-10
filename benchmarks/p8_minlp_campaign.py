import json, time
from pathlib import Path
import numpy as np
import solverpilot as om
from solverpilot.minlp import solve_binary_enumeration, solve_outer_approximation
rng=np.random.default_rng(20260906)
rows=[]; ok=0
for case in range(100):
    k=int(rng.integers(1,4)); m=om.Model(f'p8rnd{case}')
    x=m.variable(lower=-2,upper=3,name='x'); z=m.binary(k,name='z')
    target=float(rng.uniform(-1,2)); costs=rng.uniform(-0.2,0.3,size=k)
    # PSD rank-1 quadratic objective over (x,z): square of affine + affine.
    affine=x-target
    for j in range(k): affine=affine+float(rng.uniform(-0.8,0.8))*z[j]
    obj=affine**2 + sum(float(costs[j])*z[j] for j in range(k))
    m.minimize(obj)
    # Add one convex nonlinear upper constraint, deliberately loose enough for all assignments.
    center=x-float(rng.uniform(-0.5,1.5))
    for j in range(k): center=center+float(rng.uniform(-0.3,0.3))*z[j]
    m.add(center**2 <= 16.0)
    c=m.compile(); p=c.execution_ir
    t=time.perf_counter(); er=solve_binary_enumeration(p); et=time.perf_counter()-t
    t=time.perf_counter(); oa=solve_outer_approximation(p,max_iter=50); ot=time.perf_counter()-t
    match=bool(er.globally_proven and oa.globally_proven and abs(er.objective-oa.objective)<=2e-5 and oa.lower_bound<=oa.objective+2e-5)
    ok+=match
    rows.append({'case':case,'k':k,'enumeration_objective':er.objective,'oa_objective':oa.objective,'oa_lb':oa.lower_bound,'oa_gap':oa.gap,'enum_proven':er.globally_proven,'oa_proven':oa.globally_proven,'match':match,'enum_s':et,'oa_s':ot,'oa_iterations':len(oa.iterations)})
out={'schema':'solverpilot.p8.random-minlp-campaign.v1','seed':20260906,'cases':100,'passed':ok,'rows':rows}
Path('benchmarks/results/p8-random-minlp-campaign.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'passed':ok,'cases':100,'max_obj_diff':max(abs(r['enumeration_objective']-r['oa_objective']) for r in rows if r['enumeration_objective'] is not None and r['oa_objective'] is not None),'max_iterations':max(r['oa_iterations'] for r in rows)},indent=2))
