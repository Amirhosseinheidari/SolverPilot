import math

import numpy as np
import pytest

from solverpilot.conic.ir import ConeAffineBlock, ConeKind, ConicProblem
from solverpilot.conic.validation import _cone_check
from solverpilot.conic.optimality import verify_conic_optimality
from solverpilot.validate import ValidationTolerances


@pytest.mark.parametrize('kind,value,valid', [
    (ConeKind.ROTATED_SECOND_ORDER, [1e300, 0., 1.], False),
    (ConeKind.ROTATED_SECOND_ORDER, [1e300, 0., 0.], True),
    (ConeKind.ROTATED_SECOND_ORDER, [1e300, 1e-300, 1.], True),
    (ConeKind.ROTATED_SECOND_ORDER, [1e300, 1e-300, 2.], False),
    (ConeKind.SECOND_ORDER, [1e300, 1e300, 1.], False),
    (ConeKind.SECOND_ORDER, [1e300, 1e300, 0.], True),
    (ConeKind.EXPONENTIAL, [-1e308, 1e-300, 0.], False),
    (ConeKind.EXPONENTIAL, [-1e308, 0., 0.], True),
    (ConeKind.EXPONENTIAL, [-1e308, 1e-300, 5e-324], True),
    (ConeKind.EXPONENTIAL, [1., 1., math.e], False),
    (ConeKind.EXPONENTIAL, [0., 1e-300, 2e-300], True),
    (ConeKind.POWER, [1e308, 0., 5e-324], False),
    (ConeKind.POWER, [2., 2., 2.], True),
    (ConeKind.POWER, [1e300, 1e300, 1e300], True),
    (ConeKind.GENERALIZED_POWER, [1e300, 1e300, 1e300], True),
    (ConeKind.GENERALIZED_POWER, [1e308, 0., 5e-324], False),
    (ConeKind.GENERALIZED_POWER, [1., 1., 1., 1e-300], False),
])
def test_extreme_cone_membership_does_not_erase_nonzero_coordinates(kind, value, valid):
    check = _cone_check(kind, np.asarray(value), atol=0., rtol=0., source_id=None,
                        alpha=.5, weights=(.5, .5))
    assert check.valid is valid


def test_power_complement_is_exact_at_strict_boundary():
    check = _cone_check(ConeKind.POWER, np.array([1.,1e300,1e270]), atol=0., rtol=0.,
                        source_id=None, alpha=.1)
    assert not check.valid


@pytest.mark.parametrize('kind,value', [(ConeKind.EXPONENTIAL, [1.,1.,math.e]),
    (ConeKind.GENERALIZED_POWER, [1.,1.,1.,1e-8])])
def test_tight_positive_tolerance_does_not_hide_roundoff(kind, value):
    check = _cone_check(kind, np.array(value), atol=1e-20, rtol=0., source_id=None, weights=(.5,.5))
    assert not check.valid


def test_invalid_constant_rsoc_cannot_become_independently_verified_optimal():
    block = ConeAffineBlock(ConeKind.ROTATED_SECOND_ORDER, np.zeros((3,1)), np.array([1e300,0.,1.]), (3,))
    p = ConicProblem.from_data(P=[[0.]], q=[0.], A=np.empty((0,1)), variable_lower=[0.],
        variable_upper=[0.], constraint_lower=[], constraint_upper=[], cones=[block])
    tol = ValidationTolerances(feasibility=0., feasibility_rel=0., objective_abs=0., objective_rel=0.)
    check = verify_conic_optimality(p, [0.], [0.], [[0.,0.,0.]], tolerances=tol)
    assert not check.verified and not check.primal_valid
