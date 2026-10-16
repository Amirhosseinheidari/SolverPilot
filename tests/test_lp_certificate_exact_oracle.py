"""Tiny exact vertex oracle, independent of solver/production certificate math.

Enumeration is only an oracle for these bounded, nonempty, <=2-D fixtures.
No vertices in a general LP does not distinguish infeasible from unbounded.
"""
from dataclasses import asdict, replace
from fractions import Fraction as F
from itertools import combinations
import math

import numpy as np
import pytest
from scipy import sparse

from solverpilot import LinearProblem
from solverpilot.validate.lp_dual import implied_lp_box
from solverpilot.validate.optimality import verify_optimality
from solverpilot.validate.workspace import ProofPolicy, VerificationWorkspace, verification_scope


def exact_vertices(problem):
    n = len(problem.c)
    rows = []
    for coefficients, lower, upper in zip(
            np.vstack((problem.A.toarray(), np.eye(n))),
            np.r_[problem.constraint_lower, problem.variable_lower],
            np.r_[problem.constraint_upper, problem.variable_upper]):
        a = tuple(F(float(v)) for v in coefficients)
        if math.isfinite(upper):
            rows.append((a, F(float(upper))))
        if math.isfinite(lower):
            rows.append((tuple(-v for v in a), -F(float(lower))))
    vertices = set()
    for chosen in combinations(rows, n):
        matrix = [list(a) + [b] for a, b in chosen]
        for j in range(n):
            pivot = next((i for i in range(j, n) if matrix[i][j]), None)
            if pivot is None:
                break
            matrix[j], matrix[pivot] = matrix[pivot], matrix[j]
            divisor = matrix[j][j]
            matrix[j] = [v/divisor for v in matrix[j]]
            for i in range(n):
                if i != j:
                    multiplier = matrix[i][j]
                    matrix[i] = [a-multiplier*b for a, b in zip(matrix[i], matrix[j])]
        else:
            point = tuple(matrix[i][-1] for i in range(n))
            if all(sum(a*x for a, x in zip(row, point)) <= b for row, b in rows):
                vertices.add(point)
    assert vertices, "fixture must have vertices; no general infeasibility claim"
    return vertices


def exact_optimum(problem):
    values = [(sum(F(float(c))*x for c, x in zip(problem.c, point))
               + F(float(problem.objective_offset)), point) for point in exact_vertices(problem)]
    return (min(values) if problem.objective_sense.value == 'minimize' else max(values))


def make(A, c, lo, hi, vl=(0, 0), vu=(1, 1), **kwargs):
    return LinearProblem.from_data(A=A, c=c, constraint_lower=lo, constraint_upper=hi,
                                   variable_lower=vl, variable_upper=vu, **kwargs)


def fixtures():
    a = [[1, 2], [2, 1]]
    yield make(a, [1, 1], [1, 1], [np.inf]*2), [-1/3, -1/3, 0, 0], F(2, 3)
    yield make(a, [1, 1], [2, 1], [np.inf]*2), [-.5, 0, -.5, 0], F(1)
    yield make(a, [1, 1], [1, 1], [np.inf]*2, vu=(.25, 1)), [0, -1, 1, 0], F(3, 4)
    yield make([[1, 1], [2, 2]], [1, 2], [1, 2], [1, 2]), [-1, 0, 0, -1], F(1)
    yield make([[1, 1]], [3, 2], [-np.inf], [1], objective_sense='maximize',
               objective_offset=7), [3, 0, -1], F(10)
    for low, high, cost in [(-2, 3, 1), (-5, 7, 1), (-5, 7, -1)]:
        yield make([[1, 0]], [0, cost], [0], [0], vl=(-1, low), vu=(1, high)), [0, 0, -cost], F(low if cost > 0 else -high)
    epsilon = 2.**-30
    for y in (.25, .75):
        rhs = [1, 1+epsilon*y]
        yield make([[1, 1], [1, 1+epsilon]], [1, -1], rhs, rhs), [-1-2/epsilon, 2/epsilon, 0, 0], F(1-2*y)
    # All finite domain information is in rows, forcing residual correction to
    # propagate a box. The two variables themselves have infinite endpoints.
    rows = [[1, 2], [2, 1], [1, 0], [0, 1]]
    for cap in (1, 2, 1):
        p = make(rows, [1, 1], [1, 1, 0, 0], [np.inf, np.inf, cap, cap],
                 vl=(-np.inf, -np.inf), vu=(np.inf, np.inf))
        yield p, [-1/3+2.**-40, -1/3, 0, 0, 0, 0], F(2, 3)


@pytest.mark.parametrize('reverse', [False, True])
def test_bound_soundness_against_independent_exact_vertices(reverse):
    sequence = list(fixtures())
    if reverse:
        sequence.reverse()
    workspace = VerificationWorkspace(max_entries=3, max_coefficient_terms=100)
    for p, dual, expected in sequence + sequence[:2]:
        optimum, vertex = exact_optimum(p)
        assert optimum == expected
        x = np.array(vertex, dtype=float)
        uncached = verify_optimality(p, x, dual)
        cached = verify_optimality(p, x, dual, workspace=workspace)
        assert asdict(cached) == asdict(uncached)
        assert cached.verified
        reported = F(cached.dual_bound)
        assert reported <= optimum if p.objective_sense.value == 'minimize' else reported >= optimum
        assert not verify_optimality(p, x+100, dual, workspace=workspace).verified
        assert not verify_optimality(p, x, np.full(len(dual), np.nan), workspace=workspace).verified
        assert not verify_optimality(p, x, dual, workspace=workspace,
                                    proof_policy=ProofPolicy(max_terms=1)).verified
    assert workspace.info()['hits'] > 0
    assert workspace.info()['entries'] <= 3


@pytest.mark.parametrize('seed', range(12))
def test_exact_oracle_metamorphic_row_column_scaling_and_objective(seed):
    rng = np.random.default_rng(seed)
    p, dual, expected = list(fixtures())[-1]
    nrows = p.n_constraints
    order = rng.permutation(nrows)
    columns = rng.permutation(2)
    scales = np.exp2(rng.integers(-10, 11, nrows))
    cost_scale = float(2.**rng.integers(-4, 5))
    a = p.A.toarray()[order][:, columns]*scales[:, None]
    changed = make(a, p.c[columns]*cost_scale, p.constraint_lower[order]*scales,
                   p.constraint_upper[order]*scales, vl=(-np.inf,)*2, vu=(np.inf,)*2,
                   objective_offset=3)
    y = np.r_[np.array(dual)[:nrows][order]*cost_scale/scales, [0, 0]]
    optimum, vertex = exact_optimum(changed)
    assert optimum == expected*F(cost_scale)+3
    check = verify_optimality(changed, np.array(vertex, dtype=float), y,
                              workspace=VerificationWorkspace())
    assert check.verified and F(check.dual_bound) <= optimum


def test_preparation_cannot_reuse_old_bounds_matrix_or_old_acceptance():
    workspace = VerificationWorkspace(max_entries=1, max_coefficient_terms=10)
    p = make([[1, 0], [0, 1]], [1, 1], [0, 0], [1, 2],
             vl=(-np.inf,)*2, vu=(np.inf,)*2)
    variants = [p, replace(p, constraint_upper=np.array([3., 4.])),
                replace(p, A=sparse.csr_matrix([[2., 0.], [0., 1.]])), p]
    for item in variants:
        expected = implied_lp_box(item)
        with verification_scope(workspace=workspace):
            actual = implied_lp_box(item)
        for observed, reference in zip(actual, expected):
            np.testing.assert_array_equal(observed, reference)
        for vertex in exact_vertices(item):
            assert all(F(float(lo)) <= x <= F(float(hi))
                       for x, lo, hi in zip(vertex, *actual))
    assert workspace.info()['hits'] > 0 and workspace.info()['misses'] >= 3
    # A previous bounded solve cannot certify an unbounded changed model.
    unbounded = make([[1, 0]], [0, 1], [0], [0], vl=(-1, -np.inf), vu=(1, 7))
    result = verify_optimality(unbounded, [0, 0], [0, 0, 0], workspace=workspace)
    assert not result.verified and result.dual_bound is None
    inconsistent = make([[1, 1], [2, 2]], [1, 2], [1, 3], [1, 3])
    assert not verify_optimality(inconsistent, [1, 0], [-1, 0, 0, -1], workspace=workspace).verified


@pytest.mark.parametrize('value', [0., -0., 2.**-1074, -2.**-1074, 1., -1.,
                                  float.fromhex('0x1.fffffffffffffp+1023')])
@pytest.mark.parametrize('sign', [1., -1.])
def test_exact_nonimproving_bounds_skip_rounding(value, sign, monkeypatch):
    import solverpilot.validate.lp_dual as module
    p = LinearProblem.from_data(A=[[sign]], c=[0.], variable_lower=[value],
        variable_upper=[value], constraint_lower=[sign*value], constraint_upper=[sign*value])
    def unexpected(*args, **kwargs):
        pytest.fail('equal exact endpoint cannot tighten; rounding is unnecessary')
    monkeypatch.setattr(module, '_outward_ratio', unexpected)
    lower, upper = implied_lp_box(p)
    assert lower[0] == upper[0] == value


def test_cache_does_not_eagerly_prepare_with_tiny_work_budget(monkeypatch):
    workspace = VerificationWorkspace()
    p = make([[1, 0]], [0, 0], [0], [1])
    def unexpected(*args, **kwargs):
        pytest.fail('budgeted propagation must retain lazy preparation')
    monkeypatch.setattr(workspace, 'prepare', unexpected)
    with verification_scope(ProofPolicy(max_terms=10), workspace):
        implied_lp_box(p, max_visits=0)


@pytest.mark.parametrize('seed', range(24))
def test_arbitrary_duals_never_report_bound_above_exact_minimum(seed):
    rng = np.random.default_rng(seed)
    workspace = VerificationWorkspace()
    a = [[1, 0], [0, 1], [1, 1], [2, -1]]
    for _ in range(3):
        # Each changed polygon contains zero and is bounded by its first two
        # rows. No variable-box endpoint is available without propagation.
        p = make(a, rng.integers(-4, 5, 2), -rng.integers(1, 5, 4),
                 rng.integers(1, 5, 4), vl=(-np.inf,)*2, vu=(np.inf,)*2)
        optimum, vertex = exact_optimum(p)
        for dual in (np.zeros(6), np.r_[rng.integers(-4, 5, 4)/8, [0, 0]]):
            check = verify_optimality(p, np.array(vertex, dtype=float), dual, workspace=workspace)
            assert check.dual_bound is not None and math.isfinite(check.dual_bound)
            assert F(check.dual_bound) <= optimum
        with verification_scope(workspace=workspace):
            lower, upper = implied_lp_box(p)
        for point in exact_vertices(p):
            assert all(F(float(lo)) <= value <= F(float(hi))
                       for value, lo, hi in zip(point, lower, upper))
