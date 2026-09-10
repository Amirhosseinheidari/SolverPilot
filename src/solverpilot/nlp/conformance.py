from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from solverpilot.model import Model, sin, exp
from .backend import CasadiIpoptBackend
from .compiler import compile_nlp_model
from .ad import NLPDerivativeEngine


@dataclass(frozen=True, slots=True)
class NLPConformanceCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class NLPConformanceReport:
    backend: str
    available: bool
    passed: bool
    checks: tuple[NLPConformanceCheck, ...]


def conform_casadi_ipopt_backend(backend: CasadiIpoptBackend | None = None) -> NLPConformanceReport:
    backend = CasadiIpoptBackend(tol=1e-10, max_iter=2000) if backend is None else backend
    checks: list[NLPConformanceCheck] = []
    if not backend.is_available():
        return NLPConformanceReport(backend.name, False, False, (NLPConformanceCheck('availability', False, 'CasADi Ipopt plugin unavailable'),))

    # Local nonlinear objective with known minimizer.
    m=Model(); x=m.variable(2,lower=-3,upper=3); c=np.array([0.4,-0.7])
    m.minimize((x[0]-c[0])**4+(x[1]-c[1])**4+0.2*((x[0]-c[0])**2+(x[1]-c[1])**2))
    p=compile_nlp_model(m).execution_ir; r=backend.solve(p,x0=[0,0])
    ok=bool(r.validation.valid and r.local_optimal_candidate and not r.globally_proven and np.max(np.abs(r.x-c))<2e-5)
    checks.append(NLPConformanceCheck('local_nlp_solve',ok,f'x_error={np.max(np.abs(r.x-c)) if r.x is not None else None}, status={r.backend_status}'))

    # Nonlinear inequality with analytic projection.
    m=Model(); x=m.variable(2,lower=0,upper=2); m.minimize((x[0]-1)**2+(x[1]-1)**2); m.add(x[0]**2+x[1]**2<=1)
    p=compile_nlp_model(m).execution_ir; r=backend.solve(p,x0=[0.2,0.3]); target=np.array([1/np.sqrt(2)]*2)
    ok=bool(r.validation.valid and r.local_optimal_candidate and not r.globally_proven and np.max(np.abs(r.x-target))<5e-5)
    checks.append(NLPConformanceCheck('nonlinear_constraint',ok,f'x_error={np.max(np.abs(r.x-target)) if r.x is not None else None}, violation={r.validation.max_constraint_violation}'))

    # AD correctness at a deterministic point.
    m=Model(); x=m.variable(2,lower=[0.2,-2],upper=[2,2]); m.minimize(exp(0.2*x[0])+sin(x[0]*x[1])+x[0]/(1+x[1]**2)); m.add(sin(x[0])+x[1]**3<=2)
    ad=NLPDerivativeEngine(compile_nlp_model(m).execution_ir); z=np.array([1.1,-0.25]); h=1e-6
    gfd=np.zeros(2); Jfd=np.zeros((1,2))
    for i in range(2):
        zp=z.copy(); zm=z.copy(); zp[i]+=h; zm[i]-=h
        gfd[i]=(ad.objective(zp)-ad.objective(zm))/(2*h)
        Jfd[:,i]=(ad.constraints(zp)-ad.constraints(zm))/(2*h)
    ge=float(np.max(np.abs(ad.gradient(z)-gfd))); je=float(np.max(np.abs(ad.jacobian(z)-Jfd)))
    ok=ge<3e-6 and je<3e-6
    checks.append(NLPConformanceCheck('automatic_differentiation',ok,f'gradient_error={ge}, jacobian_error={je}'))

    return NLPConformanceReport(backend.name, True, all(c.passed for c in checks), tuple(checks))
