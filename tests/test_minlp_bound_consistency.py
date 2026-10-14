from types import SimpleNamespace

import numpy as np
import pytest
import solverpilot as sp
from solverpilot.minlp import solve_outer_approximation
from solverpilot.minlp.orchestrator import _bound_closure
from solverpilot.nlp import validate_nlp_solution


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


@pytest.mark.parametrize('ub,lb,atol', [(np.inf,np.inf,1e-7), (np.nan,0.,1e-7), (1.,np.nan,1e-7), (0.,0.,np.inf), (0.,0.,-1.)])
def test_nonfinite_or_invalid_bounds_never_close(ub, lb, atol):
    assert _bound_closure(ub, lb, atol)[0] is False


@pytest.mark.parametrize('kwargs', [{'atol':True}, {'atol':np.inf}, {'atol':-1.}, {'max_iter':True}, {'max_iter':1.5}, {'max_iter':0}])
def test_oa_options_do_not_coerce_invalid_values(kwargs):
    with pytest.raises(ValueError):
        solve_outer_approximation(model(), **kwargs)
