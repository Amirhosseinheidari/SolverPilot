from __future__ import annotations

import numpy as np
import pytest

from solverpilot import (
    EvidenceClass,
    LinearProblem,
    ObjectiveSense,
    PerformancePolicy,
    ProductionEvidence,
    QuadraticProblem,
    SolveIntent,
    VariableDomain,
    plan_production_solve,
    solve_production,
)
from solverpilot.runtime import default_registry
from solverpilot.experimental import M22_OFFICIAL_EVIDENCE
from solverpilot.plan.planner import NoCompatibleBackendError


def lp_problem():
    return LinearProblem.from_data(
        A=[[1.0, 1.0]], c=[1.0, 2.0],
        variable_lower=[0.0, 0.0], variable_upper=[1.0, 1.0],
        constraint_lower=[1.0], constraint_upper=[np.inf],
    )


def milp_problem():
    return LinearProblem.from_data(
        A=[[1.0, 1.0]], c=[1.0, 2.0],
        variable_lower=[0.0, 0.0], variable_upper=[1.0, 1.0],
        constraint_lower=[1.0], constraint_upper=[np.inf],
        domains=[VariableDomain.BINARY, VariableDomain.BINARY],
    )


def qp_problem():
    return QuadraticProblem.from_data(
        P=[[2.0, 0.0], [0.0, 2.0]], q=[-2.0, -4.0],
        A=[[1.0, 1.0]], variable_lower=[0.0, 0.0], variable_upper=[10.0, 10.0],
        constraint_lower=[-np.inf], constraint_upper=[10.0],
    )


def test_m22_evidence_is_correctness_not_performance_evidence():
    assert M22_OFFICIAL_EVIDENCE.evidence_class is EvidenceClass.CORPUS_VALIDATED
    assert M22_OFFICIAL_EVIDENCE.corpus_integrity_passed
    assert M22_OFFICIAL_EVIDENCE.reference_crosscheck_passed
    assert not M22_OFFICIAL_EVIDENCE.supports_performance_ranking


def test_lp_and_milp_use_conservative_highs_baselines():
    reg = default_registry()
    lp = plan_production_solve(lp_problem(), reg)
    mip = plan_production_solve(milp_problem(), reg)
    assert lp.plan.selected_backend == "scipy-highs-ds"
    assert mip.plan.selected_backend == "scipy-highs-bridge"
    assert not lp.auto_performance_ranking_enabled
    assert not mip.auto_performance_ranking_enabled


def test_m22_evidence_rejects_performance_override():
    reg = default_registry()
    decision = plan_production_solve(
        lp_problem(), reg,
        performance_override={"lp": "scipy-highs-ipm"},
        performance_policy=PerformancePolicy.ALLOW_COMPARATIVE,
    )
    assert decision.plan.selected_backend == "scipy-highs-ds"
    assert decision.rejected_performance_override_reason is not None
    assert "not comparative held-out" in decision.rejected_performance_override_reason


def test_valid_comparative_evidence_can_override_only_eligible_backend():
    reg = default_registry()
    ev = ProductionEvidence(
        evidence_class=EvidenceClass.COMPARATIVE_HELDOUT,
        source="unit-test heldout",
        corpus_integrity_passed=True,
        outcome_accounting_passed=True,
        independent_validation_passed=True,
        reference_crosscheck_passed=True,
        comparative_backends=("scipy-highs-ds", "scipy-highs-ipm"),
        heldout=True,
        feature_cost_accounted=True,
        fixed_environment=True,
        public_ood=True,
        pre_registered_policy=True,
        selection_opportunity_validated=True,
        performance_ranking_validated=True,
    )
    d = plan_production_solve(
        lp_problem(), reg, evidence=ev,
        performance_policy=PerformancePolicy.ALLOW_COMPARATIVE,
        performance_override={"lp": "scipy-highs-ipm"},
    )
    assert d.plan.selected_backend == "scipy-highs-ipm"
    assert d.auto_performance_ranking_enabled


def test_prove_optimal_qp_fails_when_only_noncertifying_routes_are_available():
    reg = default_registry()
    with pytest.raises(NoCompatibleBackendError):
        plan_production_solve(qp_problem(), reg, intent=SolveIntent.PROVE_OPTIMAL)


def test_balanced_qp_uses_available_conservative_route_but_does_not_claim_perf_model():
    reg = default_registry()
    d = plan_production_solve(qp_problem(), reg, intent=SolveIntent.BALANCED)
    assert d.plan.selected_backend in {"scipy-slsqp-qp-bridge", "nlopt-slsqp-native"}
    assert not d.auto_performance_ranking_enabled
    assert d.plan.evidence_level == "corpus_validated"


def test_solve_production_returns_decision_and_independently_validated_solution():
    result, decision = solve_production(lp_problem())
    assert result.validation is not None and result.validation.valid
    assert decision.plan.selected_backend == "scipy-highs-ds"
    assert result.plan is not None
    assert result.plan.strategy == "production_conservative"


def test_real_m22_gate_parser_preserves_noncomparative_boundary():
    import json
    from pathlib import Path
    from solverpilot.experimental import production_evidence_from_m22_gate
    gate_path = Path('/mnt/data/m22_gate_evidence/M22-FINALIZATION-GATE.json')
    if not gate_path.exists():
        pytest.skip('M22 external evidence not mounted')
    ev = production_evidence_from_m22_gate(json.loads(gate_path.read_text()), source=str(gate_path))
    assert ev.evidence_class is EvidenceClass.CORPUS_VALIDATED
    assert ev.corpus_integrity_passed and ev.outcome_accounting_passed
    assert ev.independent_validation_passed and ev.reference_crosscheck_passed
    assert not ev.supports_performance_ranking
