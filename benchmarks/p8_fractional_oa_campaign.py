import json,time
from pathlib import Path
import numpy as np, solverpilot as om
from solverpilot.minlp import solve_binary_enumeration, solve_outer_approximation
rng=np.random.default_rng(80939); rows=[]; ok=0
for case in range(60):
    k=int(rng.integers(1,4)); m=om.Model(f'frac{case}'); x=m.variable(lower=-2,upper=2); z=m.binary(k)
    obj=(x-float(rng.uniform(-.5,.5)))**2
    targets=rng.uniform(.15,.85,size=k)
    for j in range(k): obj=obj+(z[j]-float(targets[j]))**2
    m.minimize(obj)
    p=m.compile().execution_ir
    t=time.perf_counter(); e=solve_binary_enumeration(p); et=time.perf_counter()-t
    t=time.perf_counter(); o=solve_outer_approximation(p,max_iter=50); ot=time.perf_counter()-t
    match=bool(e.globally_proven and o.globally_proven and abs(e.objective-o.objective)<=2e-5)
    ok+=match; rows.append({'case':case,'k':k,'enum':e.objective,'oa':o.objective,'lb':o.lower_bound,'gap':o.gap,'bound_inversion':o.raw_statistics.get('bound_inversion',0.0),'proven':o.globally_proven,'iterations':len(o.iterations),'match':match,'enum_s':et,'oa_s':ot})
out={'schema':'solverpilot.p8.fractional-oa.v1','seed':80939,'cases':60,'passed':ok,'rows':rows}
Path('benchmarks/results/p8-fractional-oa-campaign.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'passed':ok,'cases':60,'max_iter':max(r['iterations'] for r in rows),'mean_iter':sum(r['iterations'] for r in rows)/60,'median_enum_s':float(np.median([r['enum_s'] for r in rows])),'median_oa_s':float(np.median([r['oa_s'] for r in rows])),'median_oa_over_enum':float(np.median([r['oa_s']/r['enum_s'] for r in rows])),'fails':[r for r in rows if not r['match']][:3]},indent=2))
