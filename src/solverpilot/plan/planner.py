from __future__ import annotations

from collections.abc import Iterable

from solverpilot.backends import Backend, BackendHealthReport, BackendProbeStatus, BackendRegistry
from solverpilot.capabilities import SupportLevel, compatible, requirements_for
from solverpilot.inspect import ProblemFingerprint, inspect_problem
from solverpilot.exceptions import NoCompatibleBackendError
from solverpilot.problem import LinearProblem, QuadraticProblem

from .model import CandidatePlan, HealthPolicy, PlannerContext, SolveBudget, SolveIntent, SolvePlan


def _honors_intent(backend: Backend, intent: SolveIntent) -> tuple[bool, str | None]:
    """Return whether a backend can honor hard semantic parts of an intent.

    Most intents are performance preferences and remain ranking concerns until we have
    empirical models. ``PROVE_OPTIMAL`` requests a backend capable of a solver-level
    optimal termination/certificate. It does not promise independent proof reconstruction;
    callers must inspect ``SolveResult.optimality_evidence`` for the actual trust level.
    """
    if intent is SolveIntent.PROVE_OPTIMAL and backend.manifest.metadata.get(
        "no_optimality_certificate"
    ) is True:
        return False, "backend explicitly lacks an optimality certificate"
    return True, None


def _health_map(reports: Iterable[BackendHealthReport] | None) -> dict[str, BackendHealthReport]:
    if reports is None:
        return {}
    result: dict[str, BackendHealthReport] = {}
    for report in reports:
        if report.backend in result:
            raise ValueError(f"duplicate backend health report: {report.backend}")
        result[report.backend] = report
    return result


def _candidate(
    backend: Backend,
    problem: LinearProblem | QuadraticProblem,
    *,
    context: PlannerContext,
    health_report: BackendHealthReport | None,
    health_policy: HealthPolicy,
) -> CandidatePlan:
    requirements = requirements_for(problem)
    rationale: list[str] = []
    score = 0.0

    for capability in sorted(requirements.required, key=lambda c: c.value):
        level = backend.manifest.support(capability)
        if level is SupportLevel.NATIVE:
            score += 100.0
            rationale.append(f"native support: {capability.value}")
        elif level is SupportLevel.EMULATED_SAFE:
            score += 80.0
            rationale.append(f"safe emulation: {capability.value}")

    health_status = None if health_report is None else health_report.status.value
    if health_policy is not HealthPolicy.IGNORE:
        if health_report is None:
            rationale.append("no active backend health probe supplied")
        elif health_report.status is BackendProbeStatus.HEALTHY:
            # Deliberately smaller than a capability support step: health is a tie-break,
            # not an empirical performance claim.
            score += 1.0
            rationale.append("active backend smoke probe passed")
        else:
            rationale.append(f"backend health probe status: {health_report.status.value}")

    # Stability is a deliberately tiny tie-breaker, not a performance claim.
    if context.previous_backend == backend.manifest.name:
        score += 0.1
        rationale.append("small continuity tie-break: previous backend")

    if backend.manifest.metadata.get("development_bridge"):
        rationale.append("development bridge; no native state-reuse claim")

    return CandidatePlan(
        backend=backend.manifest.name,
        score=score,
        rationale=tuple(rationale),
        health_status=health_status,
    )


def plan_solve(
    problem: LinearProblem | QuadraticProblem,
    registry: BackendRegistry,
    *,
    intent: SolveIntent | str = SolveIntent.BALANCED,
    budget: SolveBudget | None = None,
    fingerprint: ProblemFingerprint | None = None,
    context: PlannerContext | None = None,
    health_reports: Iterable[BackendHealthReport] | None = None,
    health_policy: HealthPolicy | str = HealthPolicy.IGNORE,
) -> SolvePlan:
    intent = SolveIntent(intent)
    health_policy = HealthPolicy(health_policy)
    budget = SolveBudget() if budget is None else budget
    fingerprint = inspect_problem(problem) if fingerprint is None else fingerprint
    context = PlannerContext() if context is None else context
    requirements = requirements_for(problem)
    health = _health_map(health_reports)

    eligible: list[tuple[int, CandidatePlan]] = []
    unavailable_compatible: list[str] = []
    rejected_by_health: list[str] = []
    rejected_by_intent: list[str] = []

    for registration_index, backend in enumerate(registry.all()):
        if not compatible(backend.manifest, requirements):
            continue
        if not backend.is_available():
            unavailable_compatible.append(backend.manifest.name)
            continue

        honors, intent_reason = _honors_intent(backend, intent)
        if not honors:
            rejected_by_intent.append(
                f"{backend.manifest.name} ({intent_reason})"
            )
            continue

        report = health.get(backend.manifest.name)
        if report is not None and report.version != backend.manifest.version:
            # Health evidence is version-specific. A report from another solver/package
            # version is stale and must not authorize execution.
            report = None
        if health_policy is HealthPolicy.REQUIRE_HEALTHY:
            if report is None or report.status is not BackendProbeStatus.HEALTHY:
                rejected_by_health.append(backend.manifest.name)
                continue
        elif health_policy is HealthPolicy.PREFER_HEALTHY:
            if report is not None and report.status in {
                BackendProbeStatus.UNHEALTHY,
                BackendProbeStatus.UNAVAILABLE,
            }:
                rejected_by_health.append(backend.manifest.name)
                continue

        eligible.append(
            (
                registration_index,
                _candidate(
                    backend,
                    problem,
                    context=context,
                    health_report=report,
                    health_policy=health_policy,
                ),
            )
        )

    if not eligible:
        required = ", ".join(sorted(c.value for c in requirements.required))
        extras: list[str] = []
        if unavailable_compatible:
            extras.append(f"compatible but unavailable: {', '.join(unavailable_compatible)}")
        if rejected_by_health:
            extras.append(f"rejected by health policy: {', '.join(rejected_by_health)}")
        if rejected_by_intent:
            extras.append(f"rejected by solve intent: {', '.join(rejected_by_intent)}")
        suffix = "" if not extras else " " + "; ".join(extras) + "."
        raise NoCompatibleBackendError(
            f"no available backend supports required capabilities: {required}.{suffix}"
        )

    # Score first. Registration order is the explicit deterministic tie-breaker.
    ranked = sorted(eligible, key=lambda item: (-item[1].score, item[0]))
    candidates = tuple(item[1] for item in ranked)
    selected = candidates[0]

    rationale = [
        f"selected {selected.backend} after hard capability and availability filtering",
        "no empirical performance model is active",
    ]
    if health_policy is not HealthPolicy.IGNORE:
        rationale.append(f"backend health policy: {health_policy.value}")
    if intent is SolveIntent.PROVE_OPTIMAL:
        rationale.append("hard intent gate excluded backends that explicitly lack optimality certificates")
    tied = [c for c in candidates if c.score == selected.score]
    if len(tied) > 1:
        rationale.append(
            "top candidates tied on evidence-backed score; registration order used as tie-break"
        )

    evidence_level = "capability_only" if health_policy is HealthPolicy.IGNORE else "capability+active_health"
    return SolvePlan(
        selected_backend=selected.backend,
        candidates=candidates,
        intent=intent,
        budget=budget,
        fingerprint=fingerprint,
        strategy="single_backend",
        rationale=tuple(rationale),
        evidence_level=evidence_level,
        health_policy=health_policy,
    )
