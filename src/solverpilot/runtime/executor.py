from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict, replace
from time import perf_counter

from solverpilot._immutability import deep_freeze, readonly_array
from solverpilot.backends import Backend
from solverpilot.capabilities import compatible, requirements_for
from solverpilot.exceptions import BackendUnavailableError, CapabilityMismatchError
from solverpilot.problem import LinearProblem, QuadraticProblem
from solverpilot.trace import PhaseTimings, SolveTrace
from solverpilot.validate import (
    CandidateSolution,
    PublicStatus,
    ValidationTolerances,
    validate_solution,
)

from .manifest import backend_configuration
from .result import OptimalityEvidence, SolveResult, _with_termination_certificate


def _problem_class(problem: LinearProblem | QuadraticProblem) -> str:
    if isinstance(problem, QuadraticProblem):
        return "convex_qp"
    return "milp" if problem.has_integer_variables else "lp"


def _normalize_status(backend_status: str, validation_valid: bool | None) -> PublicStatus:
    if validation_valid is False:
        return PublicStatus.INVALID_SOLUTION
    if backend_status == "optimal":
        return PublicStatus.VALID_OPTIMAL if validation_valid else PublicStatus.ERROR
    if backend_status == "converged_candidate":
        return PublicStatus.VALID_FEASIBLE if validation_valid else PublicStatus.ERROR
    if backend_status == "limit_feasible":
        return PublicStatus.FEASIBLE_LIMIT if validation_valid else PublicStatus.ERROR
    if backend_status == "infeasible":
        return PublicStatus.ERROR if validation_valid else PublicStatus.INFEASIBLE
    if backend_status == "unbounded":
        return PublicStatus.UNBOUNDED
    if backend_status == "infeasible_or_unbounded":
        return PublicStatus.INFEASIBLE_OR_UNBOUNDED
    if backend_status in {"solver_error", "limit_no_solution"}:
        return PublicStatus.ERROR
    return PublicStatus.UNKNOWN


def execute(
    problem: LinearProblem | QuadraticProblem,
    backend: Backend,
    *,
    tolerances: ValidationTolerances | None = None,
) -> SolveResult:
    total_t0 = perf_counter()
    if not backend.is_available():
        raise BackendUnavailableError(f"backend is unavailable: {backend.manifest.name}")
    manifest = backend.manifest
    requirements = requirements_for(problem)
    if not compatible(manifest, requirements):
        raise CapabilityMismatchError(
            f"backend {backend.manifest.name!r} is incompatible with "
            f"required capabilities {sorted(x.value for x in requirements.required)}"
        )

    # Owned backends explicitly expose a reentrant execution lock, also used by
    # call-local budgets. A third-party private _lock has no such contract.
    # Keep settings and result snapshots in that boundary: another call must
    # neither lend us its temporary options nor overwrite a shared workspace.
    with getattr(backend, "_execution_lock", nullcontext()):
        configuration = backend_configuration(backend)
        solve_t0 = perf_counter()
        backend_result = backend.solve(problem)
        solve_s = perf_counter() - solve_t0
        candidate_x = None if backend_result.x is None else readonly_array(backend_result.x, dtype=float)
        raw_stats = dict(deep_freeze(backend_result.raw_statistics or {}))

    validation = None
    primal_validate_s = 0.0
    if candidate_x is not None:
        validate_t0 = perf_counter()
        validation = validate_solution(
            problem,
            CandidateSolution(
                x=candidate_x,
                objective_reported=backend_result.objective_reported,
            ),
            tolerances=tolerances,
        )
        primal_validate_s = perf_counter() - validate_t0

    public_status = _normalize_status(
        backend_result.backend_status,
        None if validation is None else validation.valid,
    )
    contradictory_infeasibility = backend_result.backend_status == 'infeasible' and validation is not None and validation.valid
    if contradictory_infeasibility:
        raw_stats['termination_contradiction'] = 'backend infeasibility contradicts a canonically validated feasible candidate'
    from solverpilot._identity import process_source_sha256
    source_digest = process_source_sha256()
    total_s = perf_counter() - total_t0

    raw = raw_stats
    reuse_applied_raw = raw.get("reuse_applied")
    reuse_applied = reuse_applied_raw if isinstance(reuse_applied_raw, bool) else None
    reuse_mode_raw = raw.get("reuse_mode")
    reuse_mode = str(reuse_mode_raw) if reuse_mode_raw is not None else None

    phase = raw.get("phase_timings", {})
    phase = phase if isinstance(phase, Mapping) else {}
    measured = {k: float(v) for k, v in phase.items()
                if k in {"backend_build_s", "backend_update_s", "solve_s"}
                and isinstance(v, (int, float)) and 0 <= v <= solve_s}
    trace = SolveTrace(
        source_sha256=source_digest,
        problem_structural_hash=problem.structural_hash,
        problem_data_hash=problem.data_hash,
        problem_class=_problem_class(problem),
        backend=manifest.name,
        backend_version=manifest.version,
        parameters={"validation_tolerances": asdict(tolerances or ValidationTolerances()),
                    "backend_configuration": configuration},
        timings=PhaseTimings(solve_s=measured.get("solve_s", solve_s), backend_total_s=solve_s,
                            backend_build_s=measured.get("backend_build_s", 0.),
                            backend_update_s=measured.get("backend_update_s", 0.),
                            validate_s=primal_validate_s,
                            primal_validate_s=primal_validate_s, total_s=total_s),
        termination=backend_result.backend_status,
        validation_valid=None if validation is None else validation.valid,
        reuse_applied=reuse_applied,
        reuse_mode=reuse_mode,
        warnings=(() if validation is None else validation.warnings) +
                 (('backend infeasibility contradicts a validated feasible candidate',) if contradictory_infeasibility else ()),
    )

    objective = None if validation is None else validation.objective_recomputed
    evidence = OptimalityEvidence(
        backend_reported_optimal=backend_result.backend_status == "optimal",
        primal_validated=bool(validation is not None and validation.valid),
        # Backend statistics are claims, not an independently checked proof.
        # Keep them in raw_stats, but only a canonical verifier may promote them.
        dual_verified=False,
        gap_verified=False,
        certificate_verified=False,
    )
    certificate_s = 0.0
    if candidate_x is not None and raw_stats.get("canonical_dual") is not None:
        primal_report = validation
        if primal_report.objective_consistent is False:
            # The result report also checks the backend's objective claim, but
            # the certificate verifier has always checked only canonical x.
            # Preserve that distinction without charging this recheck to proof.
            validate_t0 = perf_counter()
            primal_report = validate_solution(
                problem, CandidateSolution(candidate_x), tolerances=tolerances,
            )
            primal_validate_s += perf_counter() - validate_t0
        proof_start = perf_counter()
        from solverpilot.validate.optimality import _verify_optimality_with_primal_report
        checked = _verify_optimality_with_primal_report(
            problem, candidate_x, raw_stats["canonical_dual"], primal_report,
            tolerances=tolerances, fast_reject=True,
        )
        if (not checked.verified and isinstance(problem, LinearProblem)
                and checked.reason == "dual multiplier points toward an infinite bound"):
            from solverpilot.validate.lp_dual import prepare_lp_dual
            repaired = prepare_lp_dual(problem, raw_stats["canonical_dual"])
            if repaired is not None:
                raw_stats["original_optimality_check"] = asdict(checked)
                checked = _verify_optimality_with_primal_report(
                    problem, candidate_x, repaired, primal_report,
                    tolerances=tolerances, fast_reject=True,
                )
                raw_stats["prepared_canonical_dual"] = repaired.tolist()
        raw_stats["optimality_check"] = asdict(checked)
        evidence = replace(evidence, dual_verified=checked.dual_valid,
                           gap_verified=checked.verified)
        certificate_s = perf_counter() - proof_start
    raw_stats["solverpilot_trust"] = {
        "backend_reported_optimal": evidence.backend_reported_optimal,
        "primal_validated": evidence.primal_validated,
        "dual_verified": evidence.dual_verified,
        "gap_verified": evidence.gap_verified,
        "certificate_verified": evidence.certificate_verified,
        "independently_verified_optimal": evidence.independently_verified_optimal,
    }
    termination_certificate = None
    if 'termination_certificate' in raw_stats:
        from solverpilot.validate.recovery import checked_termination_certificate
        proof_start = perf_counter()
        checked = checked_termination_certificate(problem, raw_stats['termination_certificate'],
            termination=backend_result.backend_status, tolerances=tolerances)
        raw_stats['termination_certificate'] = asdict(checked)
        if checked.verified and not contradictory_infeasibility and public_status in {
                PublicStatus.INFEASIBLE, PublicStatus.UNBOUNDED, PublicStatus.INFEASIBLE_OR_UNBOUNDED}:
            termination_certificate = checked
        certificate_s += perf_counter() - proof_start
    trace = replace(trace, timings=replace(trace.timings,
                    validate_s=primal_validate_s+certificate_s,
                    primal_validate_s=primal_validate_s, certificate_s=certificate_s,
                    total_s=perf_counter()-total_t0))
    return _with_termination_certificate(SolveResult(
        status=public_status,
        x=candidate_x,
        objective=objective,
        backend_status=backend_result.backend_status,
        validation=validation,
        trace=trace,
        raw_statistics=raw_stats,
    ), termination_certificate)
