from __future__ import annotations

import numpy as np
import pytest

from solverpilot import LinearProblem
from solverpilot.backends import ScipyVendoredHighsDevBackend
from solverpilot.diagnose import (
    ViolationKind,
    diagnose_infeasibility,
    elastic_relaxation,
    find_static_infeasibility,
)


def _lp(*, A, lo, hi, vl, vu, domains=None):
    n = np.asarray(A).shape[1]
    return LinearProblem.from_data(
        A=A,
        c=np.zeros(n),
        variable_lower=vl,
        variable_upper=vu,
        constraint_lower=lo,
        constraint_upper=hi,
        domains=domains,
    )


def test_static_empty_row_proves_infeasibility():
    p = _lp(A=[[0.0]], lo=[1.0], hi=[np.inf], vl=[-np.inf], vu=[np.inf])
    issues = find_static_infeasibility(p)
    assert len(issues) == 1
    assert issues[0].kind is ViolationKind.EMPTY_ROW
    assert issues[0].index == 0


def test_static_integer_bounds_with_no_integer_value():
    p = _lp(A=np.zeros((0, 1)), lo=[], hi=[], vl=[0.2], vu=[0.8], domains=["integer"])
    issues = find_static_infeasibility(p)
    assert len(issues) == 1
    assert issues[0].kind is ViolationKind.INTEGER_DOMAIN


def test_elastic_relaxation_identifies_conflicting_rows():
    # x >= 2 and x <= 1.  Minimum raw/relative relaxation is total violation 1.
    p = _lp(A=[[1.0], [1.0]], lo=[2.0, -np.inf], hi=[np.inf, 1.0], vl=[-np.inf], vu=[np.inf])
    r = elastic_relaxation(p, relative_weighting=False)
    assert r.feasible
    assert r.objective == pytest.approx(1.0, abs=1e-8)
    assert sum(v.amount for v in r.violations) == pytest.approx(1.0, abs=1e-8)
    assert {v.kind for v in r.violations} <= {ViolationKind.ROW_LOWER, ViolationKind.ROW_UPPER}


def test_elastic_relaxation_can_expose_variable_bound_conflict_with_integrality():
    p = _lp(A=np.zeros((0, 1)), lo=[], hi=[], vl=[0.2], vu=[0.8], domains=["integer"])
    r = elastic_relaxation(p, relative_weighting=False)
    assert r.feasible
    assert r.objective == pytest.approx(0.2, abs=1e-8)
    assert len(r.violations) == 1
    assert r.violations[0].kind in {ViolationKind.VARIABLE_LOWER, ViolationKind.VARIABLE_UPPER}


def test_elastic_zero_for_feasible_problem():
    p = _lp(A=[[1.0]], lo=[0.0], hi=[2.0], vl=[0.0], vu=[2.0])
    r = elastic_relaxation(p)
    assert r.feasible
    assert r.objective == pytest.approx(0.0, abs=1e-10)
    assert r.violations == ()


def test_diagnose_normalizes_native_highs_iis_when_available():
    backend = ScipyVendoredHighsDevBackend()
    if not backend.is_available():
        pytest.skip("SciPy vendored HiGHS unavailable")
    p = _lp(A=[[1.0], [1.0]], lo=[2.0, -np.inf], hi=[np.inf, 1.0], vl=[-np.inf], vu=[np.inf])
    report = diagnose_infeasibility(p, backend=backend)
    assert report.iis is not None
    assert report.iis.backend == backend.manifest.name
    assert report.iis.valid
    assert report.elastic is not None
    # The generic report keeps IIS and elastic relaxation distinct.
    assert report.iis.raw_type == "HighsDevIISResult"


def test_diagnose_static_issue_is_independent_proof():
    p = _lp(A=[[0.0]], lo=[1.0], hi=[np.inf], vl=[-np.inf], vu=[np.inf])
    report = diagnose_infeasibility(p, include_elastic=False)
    assert report.confirmed_infeasible
    assert len(report.static_issues) == 1
    assert report.elastic is None


def test_elastic_positive_optimum_confirms_infeasibility():
    p = _lp(A=[[1.0], [1.0]], lo=[2.0, -np.inf], hi=[np.inf, 1.0], vl=[-np.inf], vu=[np.inf])
    report = diagnose_infeasibility(p, include_elastic=True)
    assert report.confirmed_infeasible
    assert report.elastic is not None
    assert report.elastic.objective is not None and report.elastic.objective > 0


def test_auto_solve_can_attach_infeasibility_diagnostics():
    from solverpilot import solve

    p = _lp(A=[[1.0], [1.0]], lo=[2.0, -np.inf], hi=[np.inf, 1.0], vl=[-np.inf], vu=[np.inf])
    result = solve(p, backend="scipy-highs-ds", diagnose_infeasible=True)
    assert result.status.value == "infeasible"
    assert result.diagnostics is not None
    assert result.diagnostics.confirmed_infeasible
    assert result.trace.timings.diagnose_s > 0.0


def test_auto_solve_does_not_diagnose_unless_requested():
    from solverpilot import solve

    p = _lp(A=[[1.0], [1.0]], lo=[2.0, -np.inf], hi=[np.inf, 1.0], vl=[-np.inf], vu=[np.inf])
    result = solve(p, backend="scipy-highs-ds")
    assert result.status.value == "infeasible"
    assert result.diagnostics is None
    assert result.trace.timings.diagnose_s == 0.0


def test_deletion_filter_finds_irreducible_two_row_conflict():
    from solverpilot.diagnose import deletion_filter_conflict

    p = _lp(
        A=[[1.0], [1.0], [1.0]],
        lo=[2.0, -np.inf, -100.0],
        hi=[np.inf, 1.0, 100.0],
        vl=[-np.inf],
        vu=[np.inf],
    )
    c = deletion_filter_conflict(p)
    assert c.complete and c.irreducible
    assert len(c.atoms) == 2
    assert {(a.kind, a.index) for a in c.atoms} == {
        (ViolationKind.ROW_LOWER, 0),
        (ViolationKind.ROW_UPPER, 1),
    }


def test_deletion_filter_handles_integer_bound_conflict():
    from solverpilot.diagnose import deletion_filter_conflict

    p = _lp(A=np.zeros((0, 1)), lo=[], hi=[], vl=[0.2], vu=[0.8], domains=["integer"])
    c = deletion_filter_conflict(p)
    assert c.complete and c.irreducible
    assert {(a.kind, a.index) for a in c.atoms} == {
        (ViolationKind.VARIABLE_LOWER, 0),
        (ViolationKind.VARIABLE_UPPER, 0),
    }


def test_deletion_filter_check_budget_never_claims_iis():
    from solverpilot.diagnose import deletion_filter_conflict

    p = _lp(A=[[1.0], [1.0]], lo=[2.0, -np.inf], hi=[np.inf, 1.0], vl=[-np.inf], vu=[np.inf])
    c = deletion_filter_conflict(p, max_checks=1)
    assert not c.complete
    assert not c.irreducible
    assert c.solver_status == "check_limit"


def test_diagnose_full_can_include_distinct_conflict_evidence():
    p = _lp(A=[[1.0], [1.0]], lo=[2.0, -np.inf], hi=[np.inf, 1.0], vl=[-np.inf], vu=[np.inf])
    report = diagnose_infeasibility(p, include_conflict=True)
    assert report.conflict is not None
    assert report.conflict.irreducible
    assert report.elastic is not None
