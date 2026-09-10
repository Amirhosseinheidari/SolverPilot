from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
import numpy as np

import solverpilot as om


def run(output: Path) -> dict:
    rng=np.random.default_rng(20260906)
    backend=om.CasadiSuperSCSBackend(eps=1e-8,max_iter=100000)
    out={"schema":"solverpilot.p6.conic-campaign.v1","backend":backend.name,"binding_version":backend.binding_version}

    soc_errors=[]; soc_valid=0; t0=perf_counter()
    for _ in range(100):
        n=int(rng.integers(2,6)); c=rng.normal(size=n); R=float(rng.uniform(0.25,4.0))
        m=om.Model(); x=m.variable(n,lower=-10,upper=10); m.soc(R,x); m.minimize(c @ x)
        comp=m.compile(use_cache=False); res=comp.solve(backend=backend)
        expected=-R*float(np.linalg.norm(c))
        soc_errors.append(abs(float(res.objective_reported)-expected))
        soc_valid += int(res.validation.valid and comp.validate_original(m,res.x,atol=5e-5).valid)
    out["soc"]={"cases":100,"valid":soc_valid,"max_objective_error":max(soc_errors),"median_objective_error":float(np.median(soc_errors)),"wall_s":perf_counter()-t0}

    rs_errors=[]; rs_valid=0; t0=perf_counter()
    for _ in range(60):
        n=int(rng.integers(1,5)); zval=rng.normal(size=n); norm=float(np.linalg.norm(zval))
        m=om.Model(); z=m.variable(n,lower=zval,upper=zval); u=m.variable(lower=0,upper=20); v=m.variable(lower=0,upper=20); m.rotated_soc(u,v,z); m.minimize(u+v)
        comp=m.compile(use_cache=False); res=comp.solve(backend=backend)
        expected=np.sqrt(2.0)*norm
        rs_errors.append(abs(float(res.objective_reported)-expected))
        rs_valid += int(res.validation.valid and comp.validate_original(m,res.x,atol=8e-5).valid)
    out["rotated_soc"]={"cases":60,"valid":rs_valid,"max_objective_error":max(rs_errors),"median_objective_error":float(np.median(rs_errors)),"wall_s":perf_counter()-t0}

    qp_ir_valid=0; qp_gate_rejected=0
    for _ in range(50):
        n=int(rng.integers(2,5)); a=rng.normal(size=n); R=float(rng.uniform(.25,3.0)); an=float(np.linalg.norm(a))
        expected_x=a if an <= R else (R/an)*a
        m=om.Model(); x=m.variable(n,lower=-10,upper=10); m.soc(R,x); m.minimize(0.5*((x-m.constant(a))**2).sum())
        comp=m.compile(use_cache=False)
        qp_ir_valid += int(comp.execution_ir.P.nnz > 0 and om.validate_conic_solution(comp.execution_ir,expected_x,atol=1e-8,rtol=1e-8).valid)
        try:
            comp.solve(backend=backend)
        except RuntimeError as exc:
            qp_gate_rejected += int("problem.conic_quadratic: capability is unverified" in str(exc))
    out["quadratic_soc_ir"]={"cases":50,"known_optimum_valid":qp_ir_valid,"solver_gate_rejected":qp_gate_rejected,"solver_capability":"unverified_fail_closed"}

    psd_ok=0; psd_bad_rejected=0
    for _ in range(100):
        B=rng.normal(size=(3,3)); B=.5*(B+B.T); shift=max(0.0,-float(np.linalg.eigvalsh(B)[0]))+1e-6
        m=om.Model(); t=m.variable(lower=0,upper=30); m.psd(B+t*m.constant(np.eye(3))); m.minimize(t)
        comp=m.compile(use_cache=False); prob=comp.execution_ir
        psd_ok += int(om.validate_conic_solution(prob,np.array([shift]),atol=1e-8,rtol=1e-8).valid)
        psd_bad_rejected += int(not om.validate_conic_solution(prob,np.array([shift-2e-6]),atol=1e-8,rtol=1e-8).valid)
    out["psd_ir_validation"]={"cases":100,"known_feasible_valid":psd_ok,"known_infeasible_rejected":psd_bad_rejected,"solver_capability":"unverified_fail_closed"}

    m=om.Model(); radius=m.parameter(value=1.0); x=m.variable(3,lower=-10,upper=10); c=np.array([1.,-2.,.5]); m.soc(radius,x); m.minimize(c @ x)
    param_ok=0; max_param_err=0.; cache_status={}
    for R in rng.uniform(.2,4.,100):
        radius.value=float(R)
        cached=m.compile(use_cache=True); fresh=m.compile(use_cache=False)
        max_param_err=max(max_param_err, float(np.max(np.abs(cached.execution_ir.cones[0].g-fresh.execution_ir.cones[0].g))))
        cache_status[cached.compilation_report.cache_status]=cache_status.get(cached.compilation_report.cache_status,0)+1
        res=cached.solve(backend=backend); expected=-float(R)*float(np.linalg.norm(c))
        param_ok += int(res.validation.valid and abs(float(res.objective_reported)-expected)<=5e-4)
    out["parameter_campaign"]={"cases":100,"passed":param_ok,"max_cached_full_cone_offset_error":max_param_err,"cache_status":cache_status}

    conf=om.conform_casadi_superscs_backend(backend)
    out["conformance"]={"passed":conf.passed,"checks":[{"name":c.name,"passed":c.passed,"detail":c.detail} for c in conf.checks]}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    return out

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); args=ap.parse_args()
    data=run(Path(args.output)); print(json.dumps(data,indent=2,sort_keys=True))
