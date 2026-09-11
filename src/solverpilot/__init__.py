"""SolverPilot public API.

The historical 78-name top-level surface remains compatible in version 0.3.
New functionality is documented under explicit submodule namespaces.
Research-only milestone translators and the rejected LP selector remain available
under :mod:`solverpilot.experimental` and are intentionally excluded from this list.
"""

from __future__ import annotations

import warnings

from ._version import __version__
from .capabilities import (
    BackendManifest,
    Capability,
    CapabilityRequirements,
    SupportLevel,
    compatible,
    requirements_for,
)
from .backends import (
    Backend,
    BackendHealthReport,
    BackendProbeCheck,
    BackendProbeStatus,
    BackendRegistry,
    BackendSolveResult,
    probe_backend,
    probe_backends,
)
from .diagnose import (
    ConflictAtom,
    ConflictSetResult,
    DiagnosticIssue,
    ElasticRelaxationResult,
    EvidenceKind,
    InfeasibilityReport,
    NormalizedIISEvidence,
    RelaxationViolation,
    ViolationKind,
    deletion_filter_conflict,
    diagnose_infeasibility,
    elastic_relaxation,
    find_static_infeasibility,
)
from .exceptions import (
    BackendUnavailableError,
    BudgetNotSupportedError,
    CapabilityMismatchError,
    NoCompatibleBackendError,
    SolverPilotError,
    UnknownBackendError,
)
from .inspect import DistributionStats, ProblemFingerprint, inspect_problem
from .plan import (
    CandidatePlan,
    EvidenceClass,
    HealthPolicy,
    PerformancePolicy,
    PlannerContext,
    ProductionDecision,
    ProductionEvidence,
    SolveBudget,
    SolveIntent,
    SolvePlan,
    plan_production_solve,
    plan_solve,
)
from .problem import (
    ConvexityStatus,
    LinearProblem,
    MPSParseError,
    MPSUnsupportedFeatureError,
    ObjectiveSense,
    QuadraticProblem,
    VariableDomain,
    parse_mps,
    read_mps,
)
from .runtime import (
    PortfolioAttempt,
    PortfolioSolveResult,
    SolveResult,
    builtin_backend_candidates,
    default_registry,
    execute,
    execute_portfolio,
    solve,
    solve_production,
)
from .session import MutationKind, MutationRecord, ReuseAssessment, Session
from .trace import PhaseTimings, SolveTrace, TRACE_SCHEMA_VERSION
from .validate import (
    CandidateSolution,
    PublicStatus,
    ValidationReport,
    ValidationTolerances,
    validate_solution,
)

__all__ = [
    "Backend",
    "BackendHealthReport",
    "BackendManifest",
    "BackendProbeCheck",
    "BackendProbeStatus",
    "BackendRegistry",
    "BackendSolveResult",
    "BackendUnavailableError",
    "BudgetNotSupportedError",
    "CandidatePlan",
    "CandidateSolution",
    "Capability",
    "CapabilityMismatchError",
    "CapabilityRequirements",
    "ConflictAtom",
    "ConflictSetResult",
    "ConvexityStatus",
    "DiagnosticIssue",
    "DistributionStats",
    "ElasticRelaxationResult",
    "EvidenceClass",
    "EvidenceKind",
    "HealthPolicy",
    "InfeasibilityReport",
    "LinearProblem",
    "MPSParseError",
    "MPSUnsupportedFeatureError",
    "MutationKind",
    "MutationRecord",
    "NormalizedIISEvidence",
    "NoCompatibleBackendError",
    "ObjectiveSense",
    "SolverPilotError",
    "UnknownBackendError",
    "PerformancePolicy",
    "PhaseTimings",
    "PlannerContext",
    "PortfolioAttempt",
    "PortfolioSolveResult",
    "ProblemFingerprint",
    "ProductionDecision",
    "ProductionEvidence",
    "PublicStatus",
    "RelaxationViolation",
    "QuadraticProblem",
    "ReuseAssessment",
    "Session",
    "SolveBudget",
    "SolveIntent",
    "SolvePlan",
    "SolveResult",
    "SolveTrace",
    "SupportLevel",
    "TRACE_SCHEMA_VERSION",
    "ValidationReport",
    "ValidationTolerances",
    "VariableDomain",
    "ViolationKind",
    "builtin_backend_candidates",
    "compatible",
    "default_registry",
    "deletion_filter_conflict",
    "diagnose_infeasibility",
    "elastic_relaxation",
    "execute",
    "execute_portfolio",
    "find_static_infeasibility",
    "inspect_problem",
    "parse_mps",
    "plan_production_solve",
    "plan_solve",
    "probe_backend",
    "probe_backends",
    "read_mps",
    "requirements_for",
    "solve",
    "solve_production",
    "validate_solution",
]


# Track P P0-P9 merged feature surface.  These aliases are intentionally lazy and
# excluded from ``__all__`` so the M30-frozen stable top-level API remains unchanged.
# Canonical imports for new functionality live under solverpilot.model, .conic,
# .nlp, .minlp, .cp, .capabilities, and .session.
_TRACK_P_LAZY_EXPORTS = {
    "Model": ("solverpilot.model", "Model"),
    "CPModel": ("solverpilot.cp", "CPModel"),
    "ReferenceCPBackend": ("solverpilot.cp", "ReferenceCPBackend"),
    "ORToolsCPSATBackend": ("solverpilot.cp", "ORToolsCPSATBackend"),
    "validate_cp_solution": ("solverpilot.cp", "validate_cp_solution"),
    "reference_cp_conformance": ("solverpilot.cp", "reference_cp_conformance"),
    "ortools_cp_sat_conformance": ("solverpilot.cp", "ortools_cp_sat_conformance"),
    "ConeKind": ("solverpilot.conic", "ConeKind"),
    "ConicProblem": ("solverpilot.conic", "ConicProblem"),
    "CasadiSuperSCSBackend": ("solverpilot.conic", "CasadiSuperSCSBackend"),
    "validate_conic_solution": ("solverpilot.conic", "validate_conic_solution"),
    "conform_casadi_superscs_backend": ("solverpilot.conic", "conform_casadi_superscs_backend"),
    "conform_minlp_orchestrator": ("solverpilot.minlp", "conform_minlp_orchestrator"),
    "CapabilityKey": ("solverpilot.capabilities", "CapabilityKey"),
    "VerificationLevel": ("solverpilot.capabilities", "VerificationLevel"),
    "compatible_v2": ("solverpilot.capabilities", "compatible_v2"),
    "requirements_v2_for": ("solverpilot.capabilities", "requirements_v2_for"),
    "sum": ("solverpilot.model", "sum"),
    "symbolic_sum": ("solverpilot.model", "sum"),
    "exp": ("solverpilot.model", "exp"),
    "MutationExecutionPath": ("solverpilot.session", "MutationExecutionPath"),
    "PersistentSession": ("solverpilot.session", "PersistentSession"),
    "conform_persistent_backend": ("solverpilot.session", "conform_persistent_backend"),
}

# Pre-M30 compatibility shim. These names are intentionally not in ``__all__`` and
# must not be treated as stable product APIs. They remain lazily accessible so old
# research notebooks fail softly while emitting an explicit migration warning.
_DEPRECATED_EXPERIMENTAL = {
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
}


def __getattr__(name: str):
    if name in _TRACK_P_LAZY_EXPORTS:
        import importlib

        module_name, attr_name = _TRACK_P_LAZY_EXPORTS[name]
        return getattr(importlib.import_module(module_name), attr_name)
    if name in _DEPRECATED_EXPERIMENTAL:
        from . import experimental

        warnings.warn(
            f"solverpilot.{name} is research-only and moved to solverpilot.experimental.{name}; "
            "the top-level compatibility alias will be removed before/at 1.0 if not explicitly retained",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(experimental, name)
    raise AttributeError(f"module 'solverpilot' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _DEPRECATED_EXPERIMENTAL | set(_TRACK_P_LAZY_EXPORTS))
