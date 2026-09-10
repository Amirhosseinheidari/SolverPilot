from __future__ import annotations
import json
from pathlib import Path
from time import perf_counter_ns
import numpy as np
import solverpilot as om


def median_ms(values): return float(np.median(values))/1e6

def run(path: Path):
    rows=[]
    for k in [1,10,50,100]:
        m=om.Model(); xs=[]
        for i in range(k):
            x=m.variable(2,lower=-10,upper=10); xs.append(x); m.soc(1.0,x)
        obj=m.constant(0.0)
        for x in xs: obj=obj+x[0]
        m.minimize(obj)
        times=[]
        for _ in range(7):
            m.clear_compile_cache(); t=perf_counter_ns(); m.compile(use_cache=False); times.append(perf_counter_ns()-t)
        rows.append({"cones":k,"median_compile_ms":median_ms(times)})
    # snapshot hit and parameter relower
    m=om.Model(); r=m.parameter(value=1.0); x=m.variable(3); m.soc(r,x); m.minimize(x[0]); m.compile()
    hits=[]; relowers=[]
    for _ in range(20):
        t=perf_counter_ns(); m.compile(); hits.append(perf_counter_ns()-t)
    for val in np.linspace(1.1,3.0,20):
        r.value=float(val); t=perf_counter_ns(); m.compile(); relowers.append(perf_counter_ns()-t)
    out={"schema":"solverpilot.p6.compile-overhead.v1","cone_compile":rows,"snapshot_hit_median_ms":median_ms(hits),"parameter_full_relower_median_ms":median_ms(relowers),"claim":"overhead_characterization_only"}
    path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':
    import sys; run(Path(sys.argv[1]))
