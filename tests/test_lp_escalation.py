import pytest
from solverpilot.exact import ExactSolveResult
from solverpilot.validate import escalation
from solverpilot.validate.optimality import OptimalityCheck


def check(verified):
    return OptimalityCheck(verified, True, verified, 0., 0., 0., 'test')


@pytest.mark.parametrize('status,verified', [('optimal', True), ('bound_verified', False),
                                           ('unverified', False)])
def test_exact_fallback_only_accepts_checked_optimality(monkeypatch, status, verified):
    monkeypatch.setattr(escalation, 'recover_lp_optimality', lambda *a, **k: check(False))
    seen = []
    def exact(*a, **kw):
        seen.append(kw['time_limit'])
        return ExactSolveResult(status, status != 'unverified', 'test')
    monkeypatch.setattr(escalation, 'solve_exact', exact)
    result = escalation.recover_lp_with_fallback(None, None, None,
        scip_executable='solver', checker_executable='checker', time_limit_s=1)
    assert result.optimality_verified is verified
    assert len(seen) == 1 and 0 < seen[0] <= 1


def test_verified_numerical_check_never_calls_exact(monkeypatch):
    monkeypatch.setattr(escalation, 'recover_lp_optimality', lambda *a, **k: check(True))
    def unexpected(*a, **kw): raise AssertionError('unneeded exact solve')
    monkeypatch.setattr(escalation, 'solve_exact', unexpected)
    assert escalation.recover_lp_with_fallback(None, None, None).optimality_verified


def test_deadline_exhaustion_rejects_late_positive_check(monkeypatch):
    monkeypatch.setattr(escalation, 'recover_lp_optimality', lambda *a, **k: check(True))
    ticks = iter([0., 2., 2.])
    monkeypatch.setattr(escalation, 'perf_counter', lambda: next(ticks))
    result = escalation.recover_lp_with_fallback(None, None, None, time_limit_s=1)
    assert not result.optimality_verified and not result.within_budget


def test_invalid_budgets_and_half_configured_fallback():
    for value in (True, 0, -1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            escalation.recover_lp_with_fallback(None, None, None, time_limit_s=value)
    with pytest.raises(ValueError, match='both'):
        escalation.recover_lp_with_fallback(None, None, None, scip_executable='solver')


def test_disappearing_native_certificate_fragment_is_not_a_failure():
    from solverpilot.exact.runtime import _output_exceeds_limit, MAX_CERTIFICATE_BYTES
    from types import SimpleNamespace
    class Fragment:
        def is_file(self): return True
        def stat(self): raise FileNotFoundError('removed by solver merge')
    big = SimpleNamespace(is_file=lambda: True,
                          stat=lambda: SimpleNamespace(st_size=MAX_CERTIFICATE_BYTES+1))
    assert not _output_exceeds_limit(SimpleNamespace(iterdir=lambda: iter([Fragment()])))
    assert _output_exceeds_limit(SimpleNamespace(iterdir=lambda: iter([Fragment(), big])))


def test_long_rational_keeps_individual_component_caps():
    from fractions import Fraction
    from solverpilot.exact.certificate import Tokens, MAX_INTEGER_DIGITS
    numerator = '1'+'0'*3499
    denominator = '1'+'0'*3498+'1'
    assert Tokens((numerator+'/'+denominator).encode()).number() == Fraction(int(numerator), int(denominator))
    with pytest.raises(ValueError, match='component too long'):
        Tokens(('1'*(MAX_INTEGER_DIGITS+1)).encode()).number()
    with pytest.raises(ValueError, match='decimal too long'):
        Tokens(('1'*3000+'.'+'1'*3000).encode()).number()
