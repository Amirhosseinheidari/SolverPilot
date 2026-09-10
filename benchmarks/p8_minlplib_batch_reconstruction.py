import json,time
from pathlib import Path
import numpy as np, solverpilot as om
from solverpilot.minlp import solve_outer_approximation

# Reconstructed from the official MINLPLib 'batch' GAMS formulation shown at
# https://www.minlplib.org/batch.html (page last updated 2026-05-25).
m=om.Model('MINLPLib-batch-official-page-reconstruction')
x1=m.variable(6,lower=0,upper=1.38629436111989,name='x1_6')
x7=m.variable(6,lower=5.7037824746562,upper=8.00636756765025,name='x7_12')
lo13=np.array([4.45966,3.7495,4.49144,3.14988,3.04452]); hi13=np.array([397.747,882.353,833.333,638.298,666.667])
x13=m.variable(5,lower=lo13,upper=hi13,name='x13_17')
lo18=np.array([0.729961,0.530628,1.09024,-0.133531,0.0487901]); hi18=np.array([2.11626,1.91626,2.47654,1.25276,1.43508])
x18=m.variable(5,lower=lo18,upper=hi18,name='x18_22')
b=m.binary((6,4),name='b23_46_groups')

m.minimize(sum(250.0*om.exp(x1[j]+0.6*x7[j]) for j in range(6)))
C1=np.array([
[2.06686275947298,0.693147180559945,1.64865862558738,1.58923520511658,1.80828877117927,1.43508452528932],
[-0.356674943938732,-0.22314355131421,-0.105360515657826,1.22377543162212,0.741937344729377,0.916290731874155],
[-0.356674943938732,0.955511445027436,0.470003629245736,1.28093384546206,1.16315080980568,1.06471073699243],
[1.54756250871601,0.832909122935104,0.470003629245736,0.993251773010283,0.182321556793955,0.916290731874155],
[0.182321556793955,1.28093384546206,0.8754687373539,1.50407739677627,0.470003629245736,0.741937344729377],
])
C2=np.array([
[1.85629799036563,1.54756250871601,2.11625551480255,1.3609765531356,0.741937344729377,0.182321556793955],
[1.91692261218206,1.85629799036563,1.87180217690159,1.48160454092422,0.832909122935104,1.16315080980568],
[0.0,1.84054963339749,1.68639895357023,2.47653840011748,1.7404661748405,1.82454929205105],
[1.16315080980568,1.09861228866811,1.25276296849537,1.19392246847243,1.02961941718116,1.22377543162212],
[0.741937344729377,0.916290731874155,1.43508452528932,1.28093384546206,1.30833281965018,0.78845736036427],
])
for i in range(5):
    for j in range(6):
        m.add(x7[j]-x13[i] >= float(C1[i,j]))
        m.add(x1[j]+x18[i] >= float(C2[i,j]))
coef=np.array([250000.,150000.,180000.,160000.,120000.])
m.add(sum(float(coef[i])*om.exp(x18[i]-x13[i]) for i in range(5)) <= 6000.0)
logs=np.array([0.0,0.693147180559945,1.09861228866811,1.38629436111989])
for j in range(6):
    m.add(x1[j] - sum(float(logs[q])*b[j,q] for q in range(4)) == 0.0)
    m.add(sum(b[j,q] for q in range(4)) == 1.0)

compiled=m.compile(); p=compiled.execution_ir
t0=time.perf_counter(); r=solve_outer_approximation(p,max_iter=100,atol=1e-6); elapsed=time.perf_counter()-t0
ref=285506.5082
out={'schema':'solverpilot.p8.minlplib-batch-page-reconstruction.v1','source':'https://www.minlplib.org/batch.html','source_kind':'official_page_formulation_reconstruction_not_raw_archive','reference_primal_bound':ref,'globally_proven':r.globally_proven,'status':r.status,'objective':r.objective,'lower_bound':r.lower_bound,'gap':r.gap,'iterations':len(r.iterations),'elapsed_s':elapsed,'reference_abs_error':None if r.objective is None else abs(r.objective-ref),'validation_valid':r.validation_valid,'raw_statistics':r.raw_statistics}
Path('benchmarks/results/p8-minlplib-batch-page-reconstruction.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
