from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .backend import CasadiSuperSCSBackend


@dataclass(frozen=True, slots=True)
class ConicConformanceCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class ConicConformanceReport:
    backend: str
    binding_version: str | None
    checks: tuple[ConicConformanceCheck, ...]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


def conform_casadi_superscs_backend(backend: CasadiSuperSCSBackend | None = None) -> ConicConformanceReport:
    import solverpilot as om
    backend = CasadiSuperSCSBackend(eps=1e-9, max_iter=100000) if backend is None else backend
    checks: list[ConicConformanceCheck] = []
    if not backend.is_available():
        return ConicConformanceReport(backend.name, backend.binding_version, (ConicConformanceCheck("availability", False, "plugin unavailable"),))

    m=om.Model(); x=m.variable(2,lower=[3,4],upper=[3,4]); t=m.variable(lower=0,upper=10); m.soc(t,x); m.minimize(t)
    r=backend.solve(m.compile(use_cache=False).execution_ir)
    ok=r.validation.valid and r.objective_reported is not None and abs(r.objective_reported-5.0) <= 2e-6
    checks.append(ConicConformanceCheck("soc-3-4-5", ok, f"objective={r.objective_reported}, valid={r.validation.valid}"))

    m=om.Model(); z=m.variable(1,lower=[1],upper=[1]); u=m.variable(lower=0,upper=5); v=m.variable(lower=0,upper=5); m.rotated_soc(u,v,z); m.minimize(u+v)
    r=backend.solve(m.compile(use_cache=False).execution_ir)
    expected=float(np.sqrt(2.0)); ok=r.validation.valid and r.objective_reported is not None and abs(r.objective_reported-expected) <= 5e-6
    checks.append(ConicConformanceCheck("rotated-soc-analytic", ok, f"objective={r.objective_reported}, expected={expected}, valid={r.validation.valid}"))

    # Negative evidence is part of conformance: generic PSD must not be silently accepted.
    m=om.Model(); tt=m.variable(lower=0,upper=5); m.psd(tt*m.constant(np.eye(2))+m.constant([[0.,1.],[1.,0.]])); m.minimize(tt)
    try:
        backend.solve(m.compile(use_cache=False).execution_ir)
        refused=False; detail="PSD unexpectedly solved"
    except RuntimeError as exc:
        refused="PSD solve conformance did not pass" in str(exc); detail=str(exc)
    checks.append(ConicConformanceCheck("psd-fail-closed", refused, detail))

    m=om.Model(); xx=m.variable(3,lower=-10,upper=10); a=np.array([0.4322322767575165,-0.861140634080589,0.06823807914880933]); m.soc(2.848110356815995,xx); m.minimize(0.5*((xx-m.constant(a))**2).sum())
    try:
        backend.solve(m.compile(use_cache=False).execution_ir)
        qrefused=False; qdetail="quadratic conic objective unexpectedly solved"
    except RuntimeError as exc:
        qrefused="quadratic conic objective conformance did not pass" in str(exc); qdetail=str(exc)
    checks.append(ConicConformanceCheck("quadratic-conic-fail-closed", qrefused, qdetail))
    return ConicConformanceReport(backend.name, backend.binding_version, tuple(checks))
