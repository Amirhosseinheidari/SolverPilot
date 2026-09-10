from dataclasses import replace

import pytest

from solverpilot import HealthPolicy, LinearProblem
from solverpilot.backends import (
    BackendProbeStatus,
    BackendRegistry,
    ScipyHighsLPBackend,
    probe_backends,
)
from solverpilot.plan import NoCompatibleBackendError, plan_solve
from solverpilot.runtime import solve


def lp():
    return LinearProblem.from_data(
        A=[[1.0]],
        c=[1.0],
        variable_lower=[0.0],
        variable_upper=[2.0],
        constraint_lower=[1.0],
        constraint_upper=[2.0],
    )


def registry():
    r = BackendRegistry()
    r.register(ScipyHighsLPBackend("highs-ds"))
    r.register(ScipyHighsLPBackend("highs-ipm"))
    return r


def test_require_healthy_uses_active_probe_evidence():
    r = registry()
    reports = probe_backends(r.all())
    plan = plan_solve(
        lp(),
        r,
        health_reports=reports,
        health_policy=HealthPolicy.REQUIRE_HEALTHY,
    )
    assert plan.selected_backend == "scipy-highs-ds"
    assert plan.evidence_level == "capability+active_health"
    assert plan.health_policy is HealthPolicy.REQUIRE_HEALTHY
    assert all(c.health_status == "healthy" for c in plan.candidates)


def test_prefer_healthy_excludes_explicitly_unhealthy_backend():
    r = registry()
    reports = list(probe_backends(r.all()))
    reports[0] = replace(reports[0], status=BackendProbeStatus.UNHEALTHY)
    plan = plan_solve(
        lp(),
        r,
        health_reports=reports,
        health_policy=HealthPolicy.PREFER_HEALTHY,
    )
    assert plan.selected_backend == "scipy-highs-ipm"
    assert [c.backend for c in plan.candidates] == ["scipy-highs-ipm"]


def test_require_healthy_abstains_without_probe_evidence():
    with pytest.raises(NoCompatibleBackendError, match="rejected by health policy"):
        plan_solve(lp(), registry(), health_policy=HealthPolicy.REQUIRE_HEALTHY)


def test_solve_accepts_health_aware_planning_without_activating_performance_rule():
    r = registry()
    reports = probe_backends(r.all())
    result = solve(
        lp(),
        registry=r,
        health_reports=reports,
        health_policy=HealthPolicy.REQUIRE_HEALTHY,
    )
    assert result.status.value == "valid_optimal"
    assert result.plan is not None
    assert result.plan.evidence_level == "capability+active_health"
    assert "performance model" in " ".join(result.plan.rationale)


def test_require_healthy_rejects_stale_version_probe():
    r = registry()
    reports = list(probe_backends(r.all()))
    reports[0] = replace(reports[0], version="stale-version")
    # The second backend still has a current healthy report, so it should be selected.
    plan = plan_solve(
        lp(), r,
        health_reports=reports,
        health_policy=HealthPolicy.REQUIRE_HEALTHY,
    )
    assert plan.selected_backend == "scipy-highs-ipm"
