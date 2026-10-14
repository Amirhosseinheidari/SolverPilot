import pytest

from solverpilot import LinearProblem
from solverpilot.validate.optimality import verify_optimality, verify_infeasibility, verify_unboundedness
from solverpilot.validate.lp_dual import prepare_lp_dual, recover_lp_optimality


def problem():
    return LinearProblem.from_data(A=[[1.]], c=[1.], variable_lower=[0.], variable_upper=[2.],
        constraint_lower=[1.], constraint_upper=[float('inf')])


@pytest.mark.parametrize('huge', [10**400, -(10**400)])
def test_out_of_float64_certificate_components_are_rejected_consistently(huge):
    p = problem()
    assert not verify_optimality(p, [1.], [huge, 0.]).verified
    assert not verify_optimality(p, [huge], [-1., 0.]).verified
    assert not verify_infeasibility(p, [huge, 0.])
    assert not verify_unboundedness(p, [1.], [huge])
    assert not verify_unboundedness(p, [huge], [1.])
    assert prepare_lp_dual(p, [huge, 0.]) is None
    with pytest.raises(ValueError, match='finite continuous LP dual'):
        recover_lp_optimality(p, [1.], [huge, 0.])
