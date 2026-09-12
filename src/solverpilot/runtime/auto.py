from __future__ import annotations

from dataclasses import asdict, replace
from time import perf_counter

from solverpilot.diagnose import diagnose_infeasibility
from solverpilot.validate import PublicStatus, ValidationTolerances
from solverpilot.backends import (
    Backend,
    BackendRegistry,
    BackendHealthReport,
    ScipyHighsBackend,
    ScipyHighsLPBackend,
    ScipySLSQPQPBackend,
    HighspyNativeBackend,
    OSQPNativeBackend,
    PySCIPOptNativeBackend,
    NLoptNativeBackend,
    CasadiOSQPBridgeBackend,
    CasadiHighsBridgeBackend,
    CasadiCBCBridgeBackend,
    BundledOSQPCAPIBackend,
    BundledHighsCAPIBackend,
)
from solverpilot.inspect import inspect_problem
from solverpilot.plan import HealthPolicy, PlannerContext, SolveBudget, SolveIntent, plan_solve, PerformancePolicy, ProductionEvidence, M22_OFFICIAL_EVIDENCE, ProductionDecision, plan_production_solve
from solverpilot.problem import LinearProblem, QuadraticProblem
from solverpilot.trace import PhaseTimings

from .budgeting import apply_budget
from .executor import execute
from .result import SolveResult


def builtin_backend_candidates() -> tuple[Backend, ...]:
    """Return all supported adapter candidates, including unavailable/verification-only ones.

    This audit surface is intentionally broader than ``default_registry``: verification
    bridges and unavailable optional native packages remain visible for provenance and
    health checks, while production solving still excludes verification-only adapters.
    """

    return (
        HighspyNativeBackend(),
        OSQPNativeBackend(),
        PySCIPOptNativeBackend(),
        ScipyHighsLPBackend(method="highs-ds"),
        ScipyHighsLPBackend(method="highs-ipm"),
        ScipyHighsBackend(),
        ScipySLSQPQPBackend(),
        NLoptNativeBackend(),
        CasadiOSQPBridgeBackend(),
        CasadiHighsBridgeBackend(),
        CasadiCBCBridgeBackend(),
        BundledOSQPCAPIBackend(),
        BundledHighsCAPIBackend(),
    )


def default_registry() -> BackendRegistry:
    """Create the conservative built-in runtime registry.

    The registry contains only adapters that can actually be imported in the base
    dependency set.  Native highspy/OSQP/SCIP backends are intentionally not
    invented when their dependencies are absent.
    """

    registry = BackendRegistry()
    for backend in builtin_backend_candidates():
        # Optional native adapters remain absent from the solving registry when their
        # dependency is absent. ``builtin_backend_candidates`` is the explicit audit surface.
        if backend.manifest.metadata.get("native_adapter") and not backend.is_available():
            continue
        if backend.manifest.metadata.get("verification_only") is True:
            continue
        registry.register(backend)
    return registry


def solve(
    problem: LinearProblem | QuadraticProblem,
    *,
    registry: BackendRegistry | None = None,
    backend: str | Backend | None = None,
    intent: SolveIntent | str = SolveIntent.BALANCED,
    budget: SolveBudget | None = None,
    context: PlannerContext | None = None,
    health_reports: tuple[BackendHealthReport, ...] | None = None,
    health_policy: HealthPolicy | str = HealthPolicy.IGNORE,
    diagnose_infeasible: bool = False,
    tolerances: ValidationTolerances | None = None,
    certificate_recovery: float | None = None,
) -> SolveResult:
    """Solve a problem either with an explicit backend or the deterministic planner."""

    total_t0 = perf_counter()
    if certificate_recovery is not None:
        import math
        if isinstance(certificate_recovery, bool) or not math.isfinite(certificate_recovery) or certificate_recovery <= 0:
            raise ValueError("certificate_recovery must be a positive time budget in seconds")
    if registry is None and backend == "ortools-pdlp":
        from solverpilot.backends.pdlp import PDLPBackend
        backend = PDLPBackend()
    if registry is None and (backend is None or isinstance(backend, str)):
        registry = default_registry()

    inspect_t0 = perf_counter()
    fingerprint = inspect_problem(problem)
    inspect_s = perf_counter() - inspect_t0

    plan_s = 0.0
    plan = None
    if backend is None:
        plan_t0 = perf_counter()
        plan = plan_solve(
            problem,
            registry,
            intent=intent,
            budget=budget,
            fingerprint=fingerprint,
            context=context,
            health_reports=health_reports,
            health_policy=health_policy,
        )
        plan_s = perf_counter() - plan_t0
        chosen = registry.get(plan.selected_backend)
    elif isinstance(backend, str):
        chosen = registry.get(backend)
    else:
        chosen = backend

    chosen = apply_budget(chosen, budget)
    result = execute(problem, chosen, tolerances=tolerances)
    recovery_s = 0.0
    if certificate_recovery is not None:
        if isinstance(problem, LinearProblem) and not problem.has_integer_variables and result.backend_status in {"infeasible","unbounded","infeasible_or_unbounded"}:
            from solverpilot.validate.recovery import recover_lp_certificate, LPCertificate
            recovery_t0 = perf_counter()
            remaining=certificate_recovery
            if budget is not None and budget.wall_time_s is not None:
                remaining=min(remaining,budget.wall_time_s-(perf_counter()-total_t0))
            certificate=(recover_lp_certificate(problem,termination=result.backend_status,time_limit_s=remaining,tolerances=tolerances)
                         if remaining>0 else LPCertificate(problem.data_hash,"unknown",False,"solve budget exhausted"))
            raw=dict(result.raw_statistics or {});raw["termination_certificate"]=asdict(certificate)
            result=replace(result,raw_statistics=raw)
            recovery_s = perf_counter() - recovery_t0

    diagnose_s = 0.0
    diagnostics = None
    if diagnose_infeasible and result.status is PublicStatus.INFEASIBLE and isinstance(problem, LinearProblem):
        diagnose_t0 = perf_counter()
        diagnostics = diagnose_infeasibility(
            problem,
            backend=chosen,
            time_limit_s=None if budget is None else budget.wall_time_s,
        )
        diagnose_s = perf_counter() - diagnose_t0
        result = replace(result, diagnostics=diagnostics)

    total_s = perf_counter() - total_t0
    old = result.trace.timings
    timings = PhaseTimings(
        inspect_s=inspect_s,
        plan_s=plan_s,
        backend_build_s=old.backend_build_s,
        backend_update_s=old.backend_update_s,
        backend_total_s=old.backend_total_s,
        solve_s=old.solve_s,
        validate_s=old.validate_s + recovery_s,
        diagnose_s=old.diagnose_s + diagnose_s,
        total_s=total_s,
    )
    trace = replace(
        result.trace,
        timings=timings,
        fingerprint=asdict(fingerprint),
        planner_selected_backend=None if plan is None else plan.selected_backend,
        planner_evidence_level=None if plan is None else plan.evidence_level,
        planner_health_policy=None if plan is None else plan.health_policy.value,
        parameters={**dict(result.trace.parameters), 'budget': None if budget is None else asdict(budget)},
    )
    return replace(result, trace=trace, plan=plan)


# Backward-compatible internal alias retained for M1-M7 regression tests and callers
# that imported the previous private helper. New code should import apply_budget from
# solverpilot.runtime.
_apply_budget = apply_budget


def solve_production(
    problem: LinearProblem | QuadraticProblem,
    *,
    registry: BackendRegistry | None = None,
    intent: SolveIntent | str = SolveIntent.BALANCED,
    budget: SolveBudget | None = None,
    context: PlannerContext | None = None,
    health_reports: tuple[BackendHealthReport, ...] | None = None,
    health_policy: HealthPolicy | str = HealthPolicy.PREFER_HEALTHY,
    evidence: ProductionEvidence = M22_OFFICIAL_EVIDENCE,
    performance_policy: PerformancePolicy | str = PerformancePolicy.REQUIRE_COMPARATIVE,
    performance_override: dict[str, str] | None = None,
    diagnose_infeasible: bool = False,
    tolerances: ValidationTolerances | None = None,
    certificate_recovery: float | None = None,
) -> tuple[SolveResult, ProductionDecision]:
    """Conservative proof-safe solve plus the auditable production routing decision."""
    registry = default_registry() if registry is None else registry
    fingerprint = inspect_problem(problem)
    decision = plan_production_solve(
        problem, registry, evidence=evidence, intent=intent, budget=budget,
        fingerprint=fingerprint, context=context, health_reports=health_reports,
        health_policy=health_policy, performance_policy=performance_policy,
        performance_override=performance_override,
    )
    result = solve(
        problem, registry=registry, backend=decision.plan.selected_backend, intent=intent,
        budget=budget, context=context, health_reports=health_reports,
        health_policy=health_policy, diagnose_infeasible=diagnose_infeasible, tolerances=tolerances,
        certificate_recovery=certificate_recovery,
    )
    return replace(result, plan=decision.plan), decision
