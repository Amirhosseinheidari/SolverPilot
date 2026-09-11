from __future__ import annotations

from dataclasses import asdict, replace
from time import perf_counter

from solverpilot.backends import Backend
from solverpilot.exceptions import BackendUnavailableError, CapabilityMismatchError
from solverpilot.capabilities import compatible, requirements_for
from solverpilot.problem import LinearProblem, QuadraticProblem
from solverpilot.trace import PhaseTimings, SolveTrace
from solverpilot.validate import CandidateSolution, PublicStatus, ValidationTolerances, validate_solution

from .result import OptimalityEvidence, SolveResult
from .manifest import backend_configuration


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
        return PublicStatus.INFEASIBLE
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

    solve_t0 = perf_counter()
    backend_result = backend.solve(problem)
    solve_s = perf_counter() - solve_t0

    validation = None
    validate_s = 0.0
    if backend_result.x is not None:
        validate_t0 = perf_counter()
        validation = validate_solution(
            problem,
            CandidateSolution(
                x=backend_result.x,
                objective_reported=backend_result.objective_reported,
            ),
            tolerances=tolerances,
        )
        validate_s = perf_counter() - validate_t0

    public_status = _normalize_status(
        backend_result.backend_status,
        None if validation is None else validation.valid,
    )
    total_s = perf_counter() - total_t0

    raw = backend_result.raw_statistics or {}
    reuse_applied_raw = raw.get("reuse_applied")
    reuse_applied = reuse_applied_raw if isinstance(reuse_applied_raw, bool) else None
    reuse_mode_raw = raw.get("reuse_mode")
    reuse_mode = str(reuse_mode_raw) if reuse_mode_raw is not None else None

    phase = raw.get("phase_timings", {})
    phase = phase if isinstance(phase, dict) else {}
    measured = {k: float(v) for k, v in phase.items()
                if k in {"backend_build_s", "backend_update_s", "solve_s"}
                and isinstance(v, (int, float)) and 0 <= v <= solve_s}
    trace = SolveTrace(
        problem_structural_hash=problem.structural_hash,
        problem_data_hash=problem.data_hash,
        problem_class=_problem_class(problem),
        backend=manifest.name,
        backend_version=manifest.version,
        parameters={"validation_tolerances": asdict(tolerances or ValidationTolerances()),
                    "backend_configuration": backend_configuration(backend)},
        timings=PhaseTimings(solve_s=measured.get("solve_s", solve_s), backend_total_s=solve_s,
                            backend_build_s=measured.get("backend_build_s", 0.),
                            backend_update_s=measured.get("backend_update_s", 0.),
                            validate_s=validate_s, total_s=total_s),
        termination=backend_result.backend_status,
        validation_valid=None if validation is None else validation.valid,
        reuse_applied=reuse_applied,
        reuse_mode=reuse_mode,
        warnings=() if validation is None else validation.warnings,
    )

    objective = None if validation is None else validation.objective_recomputed
    raw_stats = dict(backend_result.raw_statistics or {})
    evidence = OptimalityEvidence(
        backend_reported_optimal=backend_result.backend_status == "optimal",
        primal_validated=bool(validation is not None and validation.valid),
        # Backend statistics are claims, not an independently checked proof.
        # Keep them in raw_stats, but only a canonical verifier may promote them.
        dual_verified=False,
        gap_verified=False,
        certificate_verified=False,
    )
    proof_start = perf_counter()
    if backend_result.x is not None and raw_stats.get("canonical_dual") is not None:
        from solverpilot.validate.optimality import verify_optimality
        checked = verify_optimality(problem, backend_result.x, raw_stats["canonical_dual"], tolerances=tolerances)
        raw_stats["optimality_check"] = asdict(checked)
        evidence = replace(evidence, dual_verified=checked.dual_valid,
                           gap_verified=checked.verified)
    proof_s = perf_counter() - proof_start
    raw_stats["solverpilot_trust"] = {
        "backend_reported_optimal": evidence.backend_reported_optimal,
        "primal_validated": evidence.primal_validated,
        "dual_verified": evidence.dual_verified,
        "gap_verified": evidence.gap_verified,
        "certificate_verified": evidence.certificate_verified,
        "independently_verified_optimal": evidence.independently_verified_optimal,
    }
    trace = replace(trace, timings=replace(trace.timings,
                    validate_s=validate_s+proof_s, total_s=perf_counter()-total_t0))
    return SolveResult(
        status=public_status,
        x=backend_result.x,
        objective=objective,
        backend_status=backend_result.backend_status,
        validation=validation,
        trace=trace,
        raw_statistics=raw_stats,
    )
