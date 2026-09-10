from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Iterable, Mapping

from solverpilot.backends import Backend, BackendHealthReport, BackendRegistry
from solverpilot.inspect import ProblemFingerprint, inspect_problem
from solverpilot.problem import LinearProblem, QuadraticProblem

from .model import HealthPolicy, PlannerContext, SolveBudget, SolveIntent, SolvePlan
from .planner import NoCompatibleBackendError, plan_solve


class EvidenceClass(str, Enum):
    NONE = "none"
    CAPABILITY = "capability"
    CORPUS_VALIDATED = "corpus_validated"
    COMPARATIVE_VALIDATION = "comparative_validation"
    COMPARATIVE_HELDOUT = "comparative_heldout"


class PerformancePolicy(str, Enum):
    CONSERVATIVE = "conservative"
    REQUIRE_COMPARATIVE = "require_comparative"
    ALLOW_COMPARATIVE = "allow_comparative"


def _canonical_payload_sha256(payload: Mapping[str, object], *, exclude: tuple[str, ...] = ()) -> str:
    body = {k: v for k, v in payload.items() if k not in exclude}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class ProductionEvidence:
    evidence_class: EvidenceClass
    source: str
    corpus_integrity_passed: bool = False
    outcome_accounting_passed: bool = False
    independent_validation_passed: bool = False
    reference_crosscheck_passed: bool = False
    comparative_backends: tuple[str, ...] = ()
    heldout: bool = False
    feature_cost_accounted: bool = False
    fixed_environment: bool = False
    public_ood: bool = False
    pre_registered_policy: bool = False
    selection_opportunity_validated: bool = False
    performance_ranking_validated: bool = False
    selector_name: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def supports_performance_ranking(self) -> bool:
        return (
            self.evidence_class is EvidenceClass.COMPARATIVE_HELDOUT
            and len(set(self.comparative_backends)) >= 2
            and self.heldout
            and self.feature_cost_accounted
            and self.fixed_environment
            and self.public_ood
            and self.pre_registered_policy
            and self.selection_opportunity_validated
            and self.performance_ranking_validated
        )


@dataclass(frozen=True, slots=True)
class ProductionDecision:
    plan: SolvePlan
    auto_performance_ranking_enabled: bool
    evidence: ProductionEvidence
    conservative_baseline: str | None
    rationale: tuple[str, ...]
    rejected_performance_override_reason: str | None = None


M22_OFFICIAL_EVIDENCE = ProductionEvidence(
    evidence_class=EvidenceClass.CORPUS_VALIDATED,
    source="M22 official MIPLIB/QPLIB/PACE fixed-host gate",
    corpus_integrity_passed=True,
    outcome_accounting_passed=True,
    independent_validation_passed=True,
    reference_crosscheck_passed=True,
    comparative_backends=(),
    heldout=False,
    feature_cost_accounted=False,
    fixed_environment=True,
    public_ood=True,
    pre_registered_policy=False,
    performance_ranking_validated=False,
    notes=(
        "official-byte integrity and outcome accounting passed",
        "M22 did not benchmark multiple candidate backends under a held-out comparative protocol",
        "timeouts were explicit outcomes and no solver-superiority claim was made",
    ),
)


def _backend_names(registry: BackendRegistry) -> set[str]:
    return {b.manifest.name for b in registry.all() if b.is_available()}


def _conservative_baseline(problem: LinearProblem | QuadraticProblem, registry: BackendRegistry) -> str | None:
    names = _backend_names(registry)
    if isinstance(problem, LinearProblem):
        if problem.has_integer_variables:
            for name in ("scipy-highs-bridge", "highspy-native", "pyscipopt-native"):
                if name in names:
                    return name
        else:
            for name in ("scipy-highs-ds", "scipy-highs-ipm", "scipy-highs-bridge", "highspy-native"):
                if name in names:
                    return name
    if isinstance(problem, QuadraticProblem):
        # Public native OSQP is preferred only when actually available. The SLSQP
        # bridges explicitly lack an optimality certificate and are therefore never
        # forced for PROVE_OPTIMAL by this production layer.
        for name in ("osqp-native", "scipy-slsqp-qp-bridge", "nlopt-slsqp-native"):
            if name in names:
                return name
    return None



def production_evidence_from_m24_public_ood(payload: Mapping[str, object], *, source: str) -> ProductionEvidence:
    """Translate an M24 public/OOD comparative report into production evidence.

    A comparative experiment is not automatically routing authority. M24 requires the
    pre-registered selector to pass every promotion gate before ``supports_performance_ranking``
    can become true. This prevents a negative or inconclusive held-out experiment from
    accidentally authorizing a user-supplied performance override.
    """
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
    gates = payload.get("promotion_gates") if isinstance(payload.get("promotion_gates"), Mapping) else {}
    rule = payload.get("pre_registered_rule") if isinstance(payload.get("pre_registered_rule"), Mapping) else {}
    claims = payload.get("claims_boundary")
    promoted = bool(payload.get("selector_promoted")) and bool(gates) and all(bool(v) for v in gates.values())
    fixed_environment = isinstance(payload.get("environment"), Mapping)
    public_ood = str(payload.get("source", "")).lower().find("miplib") >= 0
    notes = [
        "M24 public/OOD DS-vs-IPM comparative benchmark parsed into production evidence",
        f"selector promoted: {promoted}",
    ]
    if isinstance(metrics, Mapping):
        if "sbs_backend" in metrics:
            notes.append(f"SBS: {metrics['sbs_backend']}")
        boot = metrics.get("policy_to_sbs_bootstrap")
        if isinstance(boot, Mapping) and "high" in boot:
            notes.append(f"policy/SBS bootstrap upper: {boot['high']}")
    if isinstance(claims, list):
        notes.extend(str(x) for x in claims[:2])
    return ProductionEvidence(
        evidence_class=EvidenceClass.COMPARATIVE_HELDOUT,
        source=source,
        corpus_integrity_passed=bool(payload.get("official_corpus_integrity_passed")),
        outcome_accounting_passed=int(payload.get("instances", 0) or 0) >= 1,
        independent_validation_passed=(isinstance(metrics, Mapping) and int(metrics.get("objective_mismatches", 1) or 0) == 0),
        reference_crosscheck_passed=(isinstance(metrics, Mapping) and int(metrics.get("objective_mismatches", 1) or 0) == 0),
        comparative_backends=("scipy-highs-ds", "scipy-highs-ipm"),
        heldout=True,
        feature_cost_accounted=True,
        fixed_environment=fixed_environment,
        public_ood=public_ood,
        pre_registered_policy=bool(rule) and str(rule.get("origin", "")).find("frozen before M24") >= 0,
        performance_ranking_validated=promoted,
        selector_name=None if not rule else str(rule.get("name")),
        notes=tuple(notes),
    )


def production_evidence_from_m25_opportunity(payload: Mapping[str, object], *, source: str) -> ProductionEvidence:
    """Translate M25 opportunity-stability evidence into production evidence.

    M25 answers a narrower question than selector validation: is there enough stable
    SBS->VBS opportunity, *after feature cost*, to justify investing in a selector?
    Even a positive answer cannot authorize production ranking because M25 trains no
    selector and validates no deployment policy.
    """
    run1 = payload.get("run1") if isinstance(payload.get("run1"), Mapping) else {}
    run2 = payload.get("run2") if isinstance(payload.get("run2"), Mapping) else {}
    metrics1 = run1.get("metrics") if isinstance(run1.get("metrics"), Mapping) else {}
    metrics2 = run2.get("metrics") if isinstance(run2.get("metrics"), Mapping) else {}
    env_audit = payload.get("environment_audit") if isinstance(payload.get("environment_audit"), Mapping) else {}
    opportunity_ok = bool(payload.get("opportunity_validated_both")) and bool(payload.get("same_cohort")) and bool(payload.get("same_cohort_signature"))
    objective_ok = int(metrics1.get("objective_mismatches", 1) or 0) == 0 and int(metrics2.get("objective_mismatches", 1) or 0) == 0
    notes = [
        "M25 public/OOD opportunity audit parsed into production evidence",
        f"selection opportunity validated: {opportunity_ok}",
        "M25 did not train or validate a selector; ranking remains disabled",
    ]
    for label, metrics in (("run1", metrics1), ("run2", metrics2)):
        boot = metrics.get("oracle_with_inspection_to_sbs_bootstrap") if isinstance(metrics, Mapping) else None
        if isinstance(boot, Mapping) and "high" in boot:
            notes.append(f"{label} oracle+inspection/SBS bootstrap upper: {boot['high']}")
    return ProductionEvidence(
        evidence_class=EvidenceClass.COMPARATIVE_HELDOUT,
        source=source,
        corpus_integrity_passed=True,
        outcome_accounting_passed=bool(payload.get("same_cohort")),
        independent_validation_passed=objective_ok,
        reference_crosscheck_passed=objective_ok,
        comparative_backends=("scipy-highs-ds", "scipy-highs-ipm"),
        heldout=True,
        feature_cost_accounted=True,
        fixed_environment=bool(env_audit.get("captured_in_same_active_container_session")),
        public_ood=True,
        pre_registered_policy=False,
        selection_opportunity_validated=opportunity_ok,
        performance_ranking_validated=False,
        selector_name=None,
        notes=tuple(notes),
    )


def production_evidence_from_m26_validation(
    model_payload: Mapping[str, object],
    opportunity_payload: Mapping[str, object],
    *,
    source: str,
) -> ProductionEvidence:
    """Translate the sealed M26 validation result into non-deployable evidence.

    M26 intentionally stops before test evaluation when the pre-registered validation
    gate fails.  This evidence class records that a selector was trained under a
    leakage-controlled protocol, while keeping performance ranking disabled.
    """
    run1 = opportunity_payload.get("run1") if isinstance(opportunity_payload.get("run1"), Mapping) else {}
    run2 = opportunity_payload.get("run2") if isinstance(opportunity_payload.get("run2"), Mapping) else {}
    opportunity_ok = bool(opportunity_payload.get("opportunity_validated_both"))
    validation = model_payload.get("validation") if isinstance(model_payload.get("validation"), Mapping) else {}
    validation_gate = bool(model_payload.get("validation_gate_passed"))
    test_seen = bool(model_payload.get("test_outcomes_seen_during_fit"))
    notes = (
        "M26 low-capacity ridge selector was trained only on frozen train/validation evidence",
        f"M25 selection opportunity validated: {opportunity_ok}",
        f"M26 validation gate passed: {validation_gate}",
        f"validation policy/SBS: {validation.get('policy_to_sbs')}",
        "M26 final test remained sealed because the validation gate failed" if not validation_gate else "M26 validation passed; final test still requires a separate held-out gate",
    )
    return ProductionEvidence(
        evidence_class=EvidenceClass.COMPARATIVE_VALIDATION,
        source=source,
        corpus_integrity_passed=True,
        outcome_accounting_passed=True,
        independent_validation_passed=True,
        reference_crosscheck_passed=True,
        comparative_backends=("scipy-highs-ds", "scipy-highs-ipm"),
        heldout=False,
        feature_cost_accounted=True,
        fixed_environment=True,
        public_ood=True,
        pre_registered_policy=True,
        selection_opportunity_validated=opportunity_ok,
        performance_ranking_validated=False,
        selector_name=str(model_payload.get("model_family") or "m26-ridge-log-cost-ratio"),
        notes=notes + (("WARNING: test outcomes were seen during fit",) if test_seen else ()),
    )


def production_evidence_from_m27_heldout(
    model_payload: Mapping[str, object],
    test_payload: Mapping[str, object],
    opportunity_payload: Mapping[str, object],
    *,
    source: str,
) -> ProductionEvidence:
    """Translate M27 one-time held-out selector evidence.

    The exact selector is deployment-authoritative only when the frozen model passed
    development, the original M25 opportunity remained valid, and every pre-registered
    M27 held-out promotion gate passed.  A consumed-but-negative held-out test remains
    comparative evidence, but it must never enable solver-speed ranking.
    """
    opportunity_ok = bool(opportunity_payload.get("opportunity_validated_both"))
    dev_ok = bool(model_payload.get("development_gate_passed"))
    test_seen_during_fit = bool(model_payload.get("test_outcomes_seen_during_fit"))
    gates = test_payload.get("promotion_gates") if isinstance(test_payload.get("promotion_gates"), Mapping) else {}
    metrics = test_payload.get("metrics") if isinstance(test_payload.get("metrics"), Mapping) else {}
    model_self_hash_ok = str(model_payload.get("model_sha256")) == _canonical_payload_sha256(
        model_payload, exclude=("model_sha256",)
    )
    heldout_integrity = bool(
        test_payload.get("test_was_sealed_until_this_execution")
        and test_payload.get("test_execution_is_one_time")
        and model_self_hash_ok
        and str(test_payload.get("model_sha256")) == str(model_payload.get("model_sha256"))
        and str(test_payload.get("split_sha256")) == str(model_payload.get("split_sha256"))
        and not test_seen_during_fit
    )
    promoted = bool(test_payload.get("performance_ranking_validated")) and bool(gates) and all(bool(v) for v in gates.values())
    objective_ok = int(metrics.get("objective_mismatches", 1) or 0) == 0
    accounting_ok = int(metrics.get("outcomes", 0) or 0) == int(metrics.get("expected_outcomes", -1) or -1) and int(metrics.get("worker_errors", 1) or 0) == 0
    notes = (
        "M27 selective absolute-saving tree evaluated on the one-time sealed 16-instance public/OOD test",
        f"M25 opportunity validated: {opportunity_ok}",
        f"M27 development gate passed: {dev_ok}",
        f"M27 held-out promotion passed: {promoted}",
        f"frozen model self-hash valid: {model_self_hash_ok}",
        f"held-out policy/SBS: {metrics.get('policy_to_sbs')}",
        f"held-out bootstrap upper: {(metrics.get('bootstrap_policy_to_sbs') or {}).get('high') if isinstance(metrics.get('bootstrap_policy_to_sbs'), Mapping) else None}",
        "M27 test is consumed; post-test retuning requires a new untouched cohort",
    )
    return ProductionEvidence(
        evidence_class=EvidenceClass.COMPARATIVE_HELDOUT,
        source=source,
        corpus_integrity_passed=bool(test_payload.get("official_corpus_integrity_passed", False)),
        outcome_accounting_passed=accounting_ok,
        independent_validation_passed=objective_ok,
        reference_crosscheck_passed=objective_ok,
        comparative_backends=("scipy-highs-ds", "scipy-highs-ipm"),
        heldout=heldout_integrity,
        feature_cost_accounted=True,
        # M27's consumed test payload predates explicit fixed-host capture in this runner.
        # Never infer timing comparability from a same-session assumption. A future
        # positive selector must carry explicit environment provenance in the payload.
        fixed_environment=bool(test_payload.get("fixed_environment_provenance_passed", False)),
        public_ood=bool(test_payload.get("public_ood_provenance_passed", True)),
        pre_registered_policy=True,
        selection_opportunity_validated=opportunity_ok,
        performance_ranking_validated=bool(promoted and dev_ok and heldout_integrity and accounting_ok and objective_ok),
        selector_name=str(model_payload.get("model_family") or "m27-selective-absolute-saving-tree"),
        notes=notes,
    )


def production_evidence_from_m28_native_choose(
    payload: Mapping[str, object],
    *,
    source: str,
) -> ProductionEvidence:
    """Translate M28 native-choose baseline evidence.

    M28 is deliberately *not* selector-authority evidence. It asks whether the
    solver's own ``choose`` route already removes the DS-vs-IPM portfolio gap on a
    fresh public/OOD cohort. Even a strong native baseline would not validate an
    SolverPilot learned ranking policy. In the verified M28 environment, ``choose``
    did not exercise IPM on nontrivial solves and did not close the pre-registered
    fraction of the forced-simplex/IPM VBS gap.
    """
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
    gates = payload.get("pre_registered_gates") if isinstance(payload.get("pre_registered_gates"), Mapping) else {}
    gap = float(metrics.get("forced_sbs_to_vbs_relative_gap", 0.0) or 0.0)
    choose_diag = metrics.get("choose_algorithm_diagnostics") if isinstance(metrics.get("choose_algorithm_diagnostics"), Mapping) else {}
    opportunity_ok = gap >= 0.03 and int(metrics.get("objective_mismatch_instances", 1) or 0) == 0
    corpus_ok = bool(gates.get("official_corpus_integrity")) and bool(gates.get("fresh_48_unique_zero_prior_overlap"))
    accounting_ok = bool(gates.get("complete_three_round_accounting")) and bool(gates.get("single_fixed_host_and_runner_provenance"))
    validation_ok = bool(gates.get("zero_independent_invalidity"))
    objective_ok = bool(gates.get("zero_material_objective_mismatch"))
    closes = bool(payload.get("native_choose_closes_portfolio_opportunity"))
    notes = (
        "M28 compared choose/simplex/ipm within one bundled HiGHS 1.10.0 C-API build on a fresh MIPLIB-derived OOD cohort",
        f"forced SBS-to-VBS gap: {gap}",
        f"native choose closes portfolio opportunity: {closes}",
        f"choose simplex diagnostics: {choose_diag.get('simplex')}",
        f"choose IPM diagnostics: {choose_diag.get('ipm')}",
        "M28 never authorizes learned performance ranking; it is a baseline/semantics audit",
    )
    return ProductionEvidence(
        evidence_class=EvidenceClass.COMPARATIVE_HELDOUT,
        source=source,
        corpus_integrity_passed=corpus_ok,
        outcome_accounting_passed=accounting_ok,
        independent_validation_passed=validation_ok,
        reference_crosscheck_passed=objective_ok,
        comparative_backends=(
            "bundled-highs-capi:choose",
            "bundled-highs-capi:simplex",
            "bundled-highs-capi:ipm",
        ),
        heldout=True,
        feature_cost_accounted=True,
        fixed_environment=bool(gates.get("single_fixed_host_and_runner_provenance")),
        public_ood=True,
        pre_registered_policy=True,
        selection_opportunity_validated=opportunity_ok,
        performance_ranking_validated=False,
        selector_name=None,
        notes=notes,
    )


def production_evidence_from_m29_value_audit(
    payload: Mapping[str, object],
    *,
    source: str,
) -> ProductionEvidence:
    """Translate M29 value-of-information evidence conservatively.

    M29 is explicitly a post-outcome feature-representation audit over the already
    observed M25 and M28 cohorts. It can authorize at most a *future fresh-corpus
    experiment*; it can never authorize production performance ranking itself.
    When no feature family passes the pre-registered M30-authorization gate, the
    learned LP-routing branch is considered closed for the 1.0 release line.
    """
    cohorts = payload.get("cohorts") if isinstance(payload.get("cohorts"), Mapping) else {}
    families = payload.get("family_authorization") if isinstance(payload.get("family_authorization"), Mapping) else {}
    authorized = payload.get("m30_authorized_families")
    authorized_names = tuple(str(x) for x in authorized) if isinstance(authorized, list) else ()
    m30_authorized = bool(payload.get("m30_authorized")) and bool(authorized_names)
    overlap = int(cohorts.get("overlap", -1) or 0)
    notes = (
        "M29 is post-outcome representation discovery, not held-out promotion evidence",
        f"M25/M28 cohort overlap: {overlap}",
        f"M30 authorized feature families: {authorized_names}",
        f"learned LP routing closes for 1.0 when M30 authorization is false: {not m30_authorized}",
        f"audited cumulative families: {tuple(sorted(str(k) for k in families))}",
    )
    return ProductionEvidence(
        evidence_class=EvidenceClass.COMPARATIVE_VALIDATION,
        source=source,
        corpus_integrity_passed=overlap == 0 and int(cohorts.get("m25", 0) or 0) == 48 and int(cohorts.get("m28", 0) or 0) == 48,
        outcome_accounting_passed=True,
        independent_validation_passed=True,
        reference_crosscheck_passed=True,
        comparative_backends=("highs-simplex", "highs-ipm"),
        heldout=False,
        feature_cost_accounted=True,
        fixed_environment=False,
        public_ood=True,
        pre_registered_policy=False,
        selection_opportunity_validated=bool(payload.get("selection_opportunity_context_validated", False)),
        performance_ranking_validated=False,
        selector_name=None,
        notes=notes,
    )

def plan_production_solve(
    problem: LinearProblem | QuadraticProblem,
    registry: BackendRegistry,
    *,
    evidence: ProductionEvidence = M22_OFFICIAL_EVIDENCE,
    intent: SolveIntent | str = SolveIntent.BALANCED,
    budget: SolveBudget | None = None,
    fingerprint: ProblemFingerprint | None = None,
    context: PlannerContext | None = None,
    health_reports: Iterable[BackendHealthReport] | None = None,
    health_policy: HealthPolicy | str = HealthPolicy.PREFER_HEALTHY,
    performance_policy: PerformancePolicy | str = PerformancePolicy.REQUIRE_COMPARATIVE,
    performance_override: Mapping[str, str] | None = None,
) -> ProductionDecision:
    """Create a proof-safe, evidence-aware production decision.

    M23 deliberately separates *correctness/corpus evidence* from *comparative
    performance evidence*. M22 official evidence is strong enough to authorize the
    conservative runtime baselines, but not strong enough to rank competing solvers.
    """
    intent = SolveIntent(intent)
    performance_policy = PerformancePolicy(performance_policy)
    fingerprint = inspect_problem(problem) if fingerprint is None else fingerprint
    baseline = _conservative_baseline(problem, registry)
    rationale: list[str] = [
        f"evidence class: {evidence.evidence_class.value}",
        f"evidence source: {evidence.source}",
    ]
    override_rejection: str | None = None
    enable_perf = evidence.supports_performance_ranking and performance_policy is not PerformancePolicy.CONSERVATIVE

    chosen_override: str | None = None
    if performance_override:
        family = fingerprint.problem_class
        requested = performance_override.get(family)
        if requested:
            if performance_policy is PerformancePolicy.CONSERVATIVE:
                override_rejection = "performance override disabled by conservative policy"
            elif not evidence.supports_performance_ranking:
                override_rejection = (
                    "performance override rejected: evidence is not comparative held-out evidence "
                    "with feature-cost, fixed-environment, public/OOD, preregistration, and successful promotion accounting"
                )
            elif requested not in _backend_names(registry):
                override_rejection = f"performance override backend unavailable: {requested}"
            else:
                chosen_override = requested
                rationale.append(f"accepted comparative performance override for {family}: {requested}")

    if chosen_override is not None:
        # Explicitly ask the capability planner to validate the selected backend by
        # temporarily constructing the ordinary plan and ensuring the requested route
        # appears among eligible candidates. We never bypass capability/intent gates.
        ordinary = plan_solve(
            problem,
            registry,
            intent=intent,
            budget=budget,
            fingerprint=fingerprint,
            context=context,
            health_reports=health_reports,
            health_policy=health_policy,
        )
        eligible = {c.backend for c in ordinary.candidates}
        if chosen_override not in eligible:
            override_rejection = f"performance override failed capability/intent gate: {chosen_override}"
            chosen_override = None
        else:
            selected = next(c for c in ordinary.candidates if c.backend == chosen_override)
            reordered = (selected,) + tuple(c for c in ordinary.candidates if c.backend != chosen_override)
            plan = SolvePlan(
                selected_backend=chosen_override,
                candidates=reordered,
                intent=ordinary.intent,
                budget=ordinary.budget,
                fingerprint=ordinary.fingerprint,
                strategy="evidence_performance_override",
                rationale=ordinary.rationale + ("M23 comparative override passed all hard gates",),
                evidence_level="comparative_heldout",
                health_policy=ordinary.health_policy,
            )
            return ProductionDecision(
                plan=plan,
                auto_performance_ranking_enabled=True,
                evidence=evidence,
                conservative_baseline=baseline,
                rationale=tuple(rationale),
                rejected_performance_override_reason=None,
            )

    # Without comparative evidence, never performance-rank. We can still prefer a
    # conservative backend family baseline, but only if the ordinary planner says it
    # is compatible with all capability/intent/health requirements.
    ordinary = plan_solve(
        problem,
        registry,
        intent=intent,
        budget=budget,
        fingerprint=fingerprint,
        context=context,
        health_reports=health_reports,
        health_policy=health_policy,
    )
    selected_backend = ordinary.selected_backend
    if baseline is not None:
        eligible = {c.backend for c in ordinary.candidates}
        if baseline in eligible:
            selected_backend = baseline
            selected = next(c for c in ordinary.candidates if c.backend == baseline)
            candidates = (selected,) + tuple(c for c in ordinary.candidates if c.backend != baseline)
            rationale.append(f"conservative baseline selected for {fingerprint.problem_class}: {baseline}")
        else:
            candidates = ordinary.candidates
            rationale.append(f"conservative baseline {baseline} failed a hard capability/intent/health gate")
    else:
        candidates = ordinary.candidates
        rationale.append("no M23 family baseline available; retained capability planner selection")

    if evidence.evidence_class is EvidenceClass.CORPUS_VALIDATED:
        rationale.append("corpus evidence authorizes correctness-oriented production baseline, not solver-speed ranking")
    if override_rejection:
        rationale.append(override_rejection)

    plan = SolvePlan(
        selected_backend=selected_backend,
        candidates=candidates,
        intent=ordinary.intent,
        budget=ordinary.budget,
        fingerprint=ordinary.fingerprint,
        strategy="production_conservative",
        rationale=ordinary.rationale + tuple(rationale),
        evidence_level=evidence.evidence_class.value,
        health_policy=ordinary.health_policy,
    )
    return ProductionDecision(
        plan=plan,
        auto_performance_ranking_enabled=enable_perf,
        evidence=evidence,
        conservative_baseline=baseline,
        rationale=tuple(rationale),
        rejected_performance_override_reason=override_rejection,
    )


def production_evidence_from_m22_gate(payload: Mapping[str, object], *, source: str = "M22 gate JSON") -> ProductionEvidence:
    """Convert a verified M22 finalization payload into M23 planner evidence.

    This parser is deliberately conservative. A passed M22 gate proves corpus identity,
    outcome accounting and selected solution/reference checks. It does not become
    comparative performance evidence unless the payload explicitly carries a separate
    comparative held-out contract (which M22 does not).
    """
    gate_passed = payload.get("gate_passed") is True
    integrity = payload.get("integrity") if isinstance(payload.get("integrity"), Mapping) else {}
    protocol = payload.get("protocol") if isinstance(payload.get("protocol"), Mapping) else {}
    summary = payload.get("campaign_summary") if isinstance(payload.get("campaign_summary"), Mapping) else {}
    ref = payload.get("reference_crosscheck") if isinstance(payload.get("reference_crosscheck"), Mapping) else {}
    integrity_ok = bool(
        gate_passed
        and integrity.get("miplib_sha256_match") is True
        and integrity.get("qplib_sha256_match") is True
        and integrity.get("pace_official_sha1_match") is True
    )
    outcome_ok = bool(gate_passed and protocol.get("missing_outcomes_allowed") is False)
    validation_ok = bool(
        gate_passed
        and isinstance(summary.get("pace"), Mapping)
        and summary["pace"].get("invalid_solutions", 1) == 0
        and all(
            isinstance(summary.get(name), Mapping) and summary[name].get("read_or_worker_errors", 1) == 0
            for name in ("miplib", "qplib", "pace")
        )
    )
    ref_ok = bool(
        gate_passed
        and isinstance(ref.get("miplib"), Mapping)
        and ref["miplib"].get("reference_match") is True
        and isinstance(ref.get("qplib"), Mapping)
        and ref["qplib"].get("meets_or_beats_best_known") is True
    )
    evidence_class = EvidenceClass.CORPUS_VALIDATED if all((integrity_ok, outcome_ok, validation_ok, ref_ok)) else EvidenceClass.CAPABILITY
    return ProductionEvidence(
        evidence_class=evidence_class,
        source=source,
        corpus_integrity_passed=integrity_ok,
        outcome_accounting_passed=outcome_ok,
        independent_validation_passed=validation_ok,
        reference_crosscheck_passed=ref_ok,
        comparative_backends=(),
        heldout=False,
        feature_cost_accounted=False,
        fixed_environment=bool(gate_passed and isinstance(payload.get("fixed_host"), Mapping)),
        notes=(
            "M22 official gate parsed into M23 evidence",
            "no comparative multi-backend held-out performance contract present",
        ),
    )
