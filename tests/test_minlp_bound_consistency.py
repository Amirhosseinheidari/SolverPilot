from types import SimpleNamespace

import numpy as np
import pytest
import solverpilot as sp
from solverpilot.minlp import solve_outer_approximation
from solverpilot.minlp.orchestrator import _bound_closure
from solverpilot.nlp import validate_nlp_solution
from solverpilot.backends.base import BackendSolveResult
from solverpilot.capabilities import BackendManifest, Capability, SupportLevel


def model():
    m = sp.Model()
    x = m.variable(lower=0, upper=10)
    z = m.binary()
    m.minimize(x**2 + (z-1)**2)
    return m.compile().execution_ir


class ContradictoryBackend:
    def solve(self, problem, **kwargs):
        x = np.array([0., 1.]) if 'fixed_binary_assignment' in problem.metadata else np.array([10., .9999998])
        report = validate_nlp_solution(problem, x)
        assert report.valid
        return SimpleNamespace(local_optimal_candidate=True, validation=report, x=x, backend_status='optimal')


def test_near_integral_relaxation_cannot_close_inverted_bounds():
    r = solve_outer_approximation(model(), nlp_backend=ContradictoryBackend())
    assert r.status == 'valid_feasible' and r.validation_valid and r.objective == 0
    assert not r.globally_proven and not r.independently_verified_global
    assert r.lower_bound is None and r.gap is None and r.proof_scope == 'none'
    assert r.raw_statistics['bound_inversion'] == pytest.approx(100.)


def test_fractional_oa_does_not_hide_inversion_by_clipping_evaluated_bounds(monkeypatch):
    m = sp.Model()
    x = m.variable(lower=0,upper=10)
    a, b = m.binary(), m.binary()
    m.minimize(x**2+(b-1)**2+.1*a)
    p = m.compile().execution_ir
    class AnalyticDerivatives:
        def __init__(self, problem): pass
        def evaluate(self, point):
            x, a, b = point
            return float(x*x+(b-1)**2+.1*a),np.empty(0),np.array([2*x,.1,2*(b-1)]),np.empty((0,3))
    import solverpilot.minlp.orchestrator as orchestrator
    monkeypatch.setattr(orchestrator,'NLPDerivativeEngine',AnalyticDerivatives)
    class NLP:
        def solve(self, sub, **kwargs):
            bits = sub.metadata.get('fixed_binary_assignment')
            point = np.array([10.,.5,.5]) if bits is None else np.array([0.,*bits])
            report = validate_nlp_solution(sub,point)
            assert report.valid
            return SimpleNamespace(x=point,validation=report,local_optimal_candidate=True,backend_status='optimal')
    class MIP:
        manifest = BackendManifest(name='contradictory-relaxation-master',version=None,
            capabilities={Capability.LP:SupportLevel.NATIVE,Capability.MILP:SupportLevel.NATIVE})
        def is_available(self): return True
        def solve(self, sub):
            return BackendSolveResult('optimal',np.array([0.,0.,0.,100.3]))
    result = solve_outer_approximation(p,nlp_backend=NLP(),milp_backend=MIP())
    assert result.status == 'valid_feasible' and result.objective == 1.
    assert not result.globally_proven and result.lower_bound is None and result.gap is None


@pytest.mark.parametrize('ub,lb,atol', [(np.inf,np.inf,1e-7), (np.nan,0.,1e-7), (1.,np.nan,1e-7), (0.,0.,np.inf), (0.,0.,-1.)])
def test_nonfinite_or_invalid_bounds_never_close(ub, lb, atol):
    assert _bound_closure(ub, lb, atol)[0] is False


@pytest.mark.parametrize('kwargs', [{'atol':True}, {'atol':np.inf}, {'atol':-1.}, {'max_iter':True}, {'max_iter':1.5}, {'max_iter':0}])
def test_oa_options_do_not_coerce_invalid_values(kwargs):
    with pytest.raises(ValueError):
        solve_outer_approximation(model(), **kwargs)
