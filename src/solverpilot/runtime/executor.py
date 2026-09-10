from __future__ import annotations

from dataclasses import replace
from time import perf_counter

from solverpilot.backends import Backend
from solverpilot.exceptions import BackendUnavailableError, CapabilityMismatchError
from solverpilot.capabilities import compatible, requirements_for
from solverpilot.problem import LinearProblem, QuadraticProblem
from solverpilot.trace import PhaseTimings, SolveTrace
from solverpilot.validate import CandidateSolution, PublicStatus, validate_solution

from .result import OptimalityEvidence, SolveResult


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
) -> SolveResult:
    total_t0 = perf_counter()
    if not backend.is_available():
        raise BackendUnavailableError(f"backend is unavailable: {backend.manifest.name}")
    requirements = requirements_for(problem)
    if not compatible(backend.manifest, requirements):
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

    trace = SolveTrace(
        problem_structural_hash=problem.structural_hash,
        problem_data_hash=problem.data_hash,
        problem_class=_problem_class(problem),
        backend=backend.manifest.name,
        backend_version=backend.manifest.version,
        timings=PhaseTimings(solve_s=solve_s, validate_s=validate_s, total_s=total_s),
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
    raw_stats["solverpilot_trust"] = {
        "backend_reported_optimal": evidence.backend_reported_optimal,
        "primal_validated": evidence.primal_validated,
        "dual_verified": evidence.dual_verified,
        "gap_verified": evidence.gap_verified,
        "certificate_verified": evidence.certificate_verified,
        "independently_verified_optimal": evidence.independently_verified_optimal,
    }
    return SolveResult(
        status=public_status,
        x=backend_result.x,
        objective=objective,
        backend_status=backend_result.backend_status,
        validation=validation,
        trace=trace,
        raw_statistics=raw_stats,
    )
