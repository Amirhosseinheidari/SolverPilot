"""Deterministic numerical oracles for the 0.2 additions."""
import numpy as np
import pytest
import solverpilot as sp


@pytest.mark.parametrize('sense', ['minimize', 'maximize'])
def test_lp_signed_duals_against_analytic_box_optima(sense):
    from solverpilot.backends import ScipyHighsLPBackend
    rng = np.random.default_rng(20260912)
    for _ in range(40):
        n = 8; c = rng.normal(size=n); lo = rng.uniform(-3, -1, n); hi = rng.uniform(1, 3, n)
        p = sp.LinearProblem.from_data(A=np.empty((0, n)), c=c, variable_lower=lo, variable_upper=hi,
                                      constraint_lower=[], constraint_upper=[], objective_sense=sense)
        result = sp.execute(p, ScipyHighsLPBackend())
        expected = np.where(c > 0, lo, hi) if sense == 'minimize' else np.where(c > 0, hi, lo)
        assert result.objective == pytest.approx(float(c@expected), abs=1e-8)
        assert result.optimality_evidence.independently_verified_optimal


def test_quadratic_reuse_campaign_against_analytic_optimum():
    pytest.importorskip('osqp')
    from solverpilot.backends import OSQPNativeBackend
    rng = np.random.default_rng(20260912); n = 12
    diagonal = rng.uniform(1, 4, n); backend = OSQPNativeBackend(eps_abs=1e-9, eps_rel=1e-9)
    for i in range(100):
        q = rng.normal(size=n)
        p = sp.QuadraticProblem.from_data(P=np.diag(diagonal), A=np.empty((0, n)), q=q,
                                         variable_lower=np.zeros(n), variable_upper=np.ones(n),
                                         constraint_lower=[], constraint_upper=[])
        result = sp.execute(p, backend)
        expected = np.clip(-q/diagonal, 0, 1)
        assert result.x == pytest.approx(expected, abs=1e-7)
        assert result.optimality_evidence.independently_verified_optimal
        assert result.trace.reuse_applied == (i != 0)


def test_random_psd_analytic_eigenvalue_oracle():
    pytest.importorskip('clarabel')
    from solverpilot.conic import ClarabelBackend
    rng = np.random.default_rng(20260912)
    for _ in range(30):
        M = rng.normal(size=(4, 4)); M = (M+M.T)/2
        m = sp.Model(); t = m.variable(lower=-10, upper=10)
        m.psd(t*m.constant(np.eye(4))+m.constant(M)); m.minimize(t)
        result = m.compile().solve(backend=ClarabelBackend())
        assert result.validated
        assert result.objective_reported == pytest.approx(-np.linalg.eigvalsh(M)[0], abs=3e-6)


def test_exponential_and_power_negative_coordinates_cannot_hide_in_scale():
    from solverpilot.conic import ConeKind
    from solverpilot.conic.validation import _cone_check
    assert not _cone_check(ConeKind.POWER, np.array([-1., 1e200, 0.]), atol=1e-8, rtol=1e-8, source_id=None, alpha=.5).valid
    assert not _cone_check(ConeKind.EXPONENTIAL, np.array([-1e200, -1., 1.]), atol=1e-8, rtol=1e-8, source_id=None).valid


def test_unbounded_certificate_requires_feasible_origin_and_recession():
    from solverpilot.validate import verify_unboundedness
    p = sp.LinearProblem.from_data(A=np.empty((0, 1)), c=[-1.], variable_lower=[0.], variable_upper=[np.inf],
                                  constraint_lower=[], constraint_upper=[])
    assert verify_unboundedness(p, [0.], [1.])
    assert not verify_unboundedness(p, [-1.], [1.])
    assert not verify_unboundedness(p, [0.], [-1.])
