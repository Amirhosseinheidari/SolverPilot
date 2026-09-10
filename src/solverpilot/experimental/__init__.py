"""Experimental and research-only SolverPilot APIs.

Everything in this namespace may change or disappear without the compatibility
policy applied to the stable top-level ``solverpilot`` API.  M30 deliberately moves
milestone-specific evidence translators and the rejected LP selector evaluator
here so research artifacts cannot be mistaken for supported product features.
"""

from solverpilot.plan.production import (
    M22_OFFICIAL_EVIDENCE,
    production_evidence_from_m22_gate,
    production_evidence_from_m24_public_ood,
    production_evidence_from_m25_opportunity,
    production_evidence_from_m26_validation,
    production_evidence_from_m27_heldout,
    production_evidence_from_m28_native_choose,
    production_evidence_from_m29_value_audit,
)
from solverpilot.plan.selective_lp import SelectiveLPDecision, decide_selective_lp_backend

__all__ = [
    "M22_OFFICIAL_EVIDENCE",
    "SelectiveLPDecision",
    "decide_selective_lp_backend",
    "production_evidence_from_m22_gate",
    "production_evidence_from_m24_public_ood",
    "production_evidence_from_m25_opportunity",
    "production_evidence_from_m26_validation",
    "production_evidence_from_m27_heldout",
    "production_evidence_from_m28_native_choose",
    "production_evidence_from_m29_value_audit",
]
