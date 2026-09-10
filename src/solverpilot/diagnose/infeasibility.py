from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from solverpilot.problem import LinearProblem, VariableDomain

from .model import (
    ConflictAtom,
    ConflictSetResult,
    DiagnosticIssue,
    ElasticRelaxationResult,
    EvidenceKind,
    InfeasibilityReport,
    NormalizedIISEvidence,
    RelaxationViolation,
    ViolationKind,
)


def find_static_infeasibility(problem: LinearProblem, *, atol: float = 1e-12) -> tuple[DiagnosticIssue, ...]:
    """Find contradictions that require no optimization solve.

    The checks are deliberately proof-producing: every returned issue is independently
    sufficient to prove infeasibility.  Heuristic numerical warnings belong elsewhere.
    """
    issues: list[DiagnosticIssue] = []

    # A zero row has activity exactly zero.  If zero is outside the ranged row bounds,
    # that row alone proves infeasibility.
    row_nnz = np.diff(problem.A.indptr)
    for i in np.flatnonzero(row_nnz == 0):
        lo = float(problem.constraint_lower[i])
        hi = float(problem.constraint_upper[i])
        if lo > atol or hi < -atol:
            distance = lo if lo > 0 else -hi
            issues.append(
                DiagnosticIssue(
                    kind=ViolationKind.EMPTY_ROW,
                    index=int(i),
                    amount=float(distance),
                    message=(
                        f"constraint row {int(i)} has no nonzero coefficients, so its activity is 0, "
                        f"outside [{lo}, {hi}]"
                    ),
                )
            )

    # Continuous feasibility of bounds is enforced by LinearProblem.  Integer variables
    # have the additional requirement that [lb, ub] contains at least one integer.
    for j, domain in enumerate(problem.domains):
        if domain not in {VariableDomain.INTEGER.value, VariableDomain.BINARY.value}:
            continue
        lo = float(problem.variable_lower[j])
        hi = float(problem.variable_upper[j])
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        ceil_lo = np.ceil(lo - atol)
        floor_hi = np.floor(hi + atol)
        if ceil_lo > floor_hi:
            issues.append(
                DiagnosticIssue(
                    kind=ViolationKind.INTEGER_DOMAIN,
                    index=j,
                    amount=float(ceil_lo - floor_hi),
                    message=(
                        f"integer variable {j} has bounds [{lo}, {hi}] containing no integer value"
                    ),
                )
            )

    return tuple(issues)


@dataclass(frozen=True, slots=True)
class _SlackSource:
    kind: ViolationKind
    index: int
    bound_value: float


def elastic_relaxation(
    problem: LinearProblem,
    *,
    time_limit_s: float | None = None,
    relative_weighting: bool = True,
    violation_tol: float = 1e-9,
) -> ElasticRelaxationResult:
    """Solve a phase-I elastic model and report which bounds must move.

    Every finite row and variable bound gets its own nonnegative slack.  Original
    integrality is preserved.  This is *not* an IIS: it finds one minimum-weight
    relaxation, which may contain redundant violations and may not be unique.
    """
    n = problem.n_variables
    sources: list[_SlackSource] = []
    rows: list[sparse.csr_matrix] = []
    lower: list[float] = []
    upper: list[float] = []

    # We build one inequality per finite side.  Slack columns are appended after x.
    row_specs: list[tuple[sparse.csr_matrix, float, float, _SlackSource, float]] = []
    A = problem.A.tocsr()
    for i in range(problem.n_constraints):
        arow = A.getrow(i)
        lo = float(problem.constraint_lower[i])
        hi = float(problem.constraint_upper[i])
        if np.isfinite(lo):
            source = _SlackSource(ViolationKind.ROW_LOWER, i, lo)
            row_specs.append((arow, lo, np.inf, source, +1.0))
        if np.isfinite(hi):
            source = _SlackSource(ViolationKind.ROW_UPPER, i, hi)
            row_specs.append((arow, -np.inf, hi, source, -1.0))

    eye = sparse.eye(n, format="csr")
    for j in range(n):
        lo = float(problem.variable_lower[j])
        hi = float(problem.variable_upper[j])
        erow = eye.getrow(j)
        if np.isfinite(lo):
            source = _SlackSource(ViolationKind.VARIABLE_LOWER, j, lo)
            row_specs.append((erow, lo, np.inf, source, +1.0))
        if np.isfinite(hi):
            source = _SlackSource(ViolationKind.VARIABLE_UPPER, j, hi)
            row_specs.append((erow, -np.inf, hi, source, -1.0))

    k = len(row_specs)
    if k == 0:
        # No finite bounds means the model cannot be infeasible from linear constraints.
        return ElasticRelaxationResult(
            feasible=True,
            objective=0.0,
            x=np.zeros(n, dtype=np.float64),
            violations=(),
            solver_status="trivial",
            message="no finite row or variable bounds required relaxation",
        )

    # Sparse [original variables | one private slack per bound side].
    pieces: list[sparse.csr_matrix] = []
    for slack_idx, (base, lo, hi, source, slack_sign) in enumerate(row_specs):
        slack = sparse.csr_matrix(
            ([slack_sign], ([0], [slack_idx])), shape=(1, k), dtype=np.float64
        )
        pieces.append(sparse.hstack([base, slack], format="csr"))
        lower.append(lo)
        upper.append(hi)
        sources.append(source)

    M = sparse.vstack(pieces, format="csr")

    weights = np.ones(k, dtype=np.float64)
    if relative_weighting:
        weights = np.asarray(
            [1.0 / max(1.0, abs(src.bound_value)) for src in sources], dtype=np.float64
        )
    objective = np.concatenate([np.zeros(n, dtype=np.float64), weights])

    integrality = np.zeros(n + k, dtype=np.int32)
    for j, domain in enumerate(problem.domains):
        if domain in {VariableDomain.INTEGER.value, VariableDomain.BINARY.value}:
            integrality[j] = 1

    # Original variable bounds are represented as elastic rows above, so x itself is free.
    lb = np.concatenate([np.full(n, -np.inf), np.zeros(k)])
    ub = np.full(n + k, np.inf)
    options: dict[str, Any] = {"disp": False}
    if time_limit_s is not None:
        if time_limit_s <= 0:
            raise ValueError("time_limit_s must be positive")
        options["time_limit"] = float(time_limit_s)

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(M, np.asarray(lower), np.asarray(upper)),
        options=options,
    )

    if result.x is None or not bool(result.success):
        return ElasticRelaxationResult(
            feasible=False,
            objective=None,
            x=None,
            violations=(),
            solver_status=str(getattr(result, "status", "unknown")),
            message=str(getattr(result, "message", "elastic relaxation failed")),
        )

    z = np.asarray(result.x, dtype=np.float64)
    x = z[:n].copy()
    slack_values = z[n:]
    violations: list[RelaxationViolation] = []
    for idx, amount in enumerate(slack_values):
        raw = float(max(0.0, amount))
        if raw <= violation_tol:
            continue
        src = sources[idx]
        violations.append(
            RelaxationViolation(
                kind=src.kind,
                index=src.index,
                amount=raw,
                normalized_amount=float(raw * weights[idx]),
            )
        )
    violations.sort(key=lambda v: (-v.normalized_amount, v.kind.value, v.index))
    return ElasticRelaxationResult(
        feasible=True,
        objective=float(result.fun),
        x=x,
        violations=tuple(violations),
        solver_status=str(getattr(result, "status", "0")),
        message=str(getattr(result, "message", "")),
    )



def _all_bound_atoms(problem: LinearProblem) -> list[ConflictAtom]:
    atoms: list[ConflictAtom] = []
    for i in range(problem.n_constraints):
        if np.isfinite(problem.constraint_lower[i]):
            atoms.append(ConflictAtom(ViolationKind.ROW_LOWER, i))
        if np.isfinite(problem.constraint_upper[i]):
            atoms.append(ConflictAtom(ViolationKind.ROW_UPPER, i))
    for j in range(problem.n_variables):
        if np.isfinite(problem.variable_lower[j]):
            atoms.append(ConflictAtom(ViolationKind.VARIABLE_LOWER, j))
        if np.isfinite(problem.variable_upper[j]):
            atoms.append(ConflictAtom(ViolationKind.VARIABLE_UPPER, j))
    return atoms


def _feasibility_status_for_atoms(
    problem: LinearProblem,
    atoms: list[ConflictAtom],
    *,
    time_limit_s: float | None,
) -> str:
    n = problem.n_variables
    rows: list[sparse.csr_matrix] = []
    lo: list[float] = []
    hi: list[float] = []
    A = problem.A.tocsr()
    eye = sparse.eye(n, format="csr")
    for atom in atoms:
        if atom.kind is ViolationKind.ROW_LOWER:
            rows.append(A.getrow(atom.index)); lo.append(float(problem.constraint_lower[atom.index])); hi.append(np.inf)
        elif atom.kind is ViolationKind.ROW_UPPER:
            rows.append(A.getrow(atom.index)); lo.append(-np.inf); hi.append(float(problem.constraint_upper[atom.index]))
        elif atom.kind is ViolationKind.VARIABLE_LOWER:
            rows.append(eye.getrow(atom.index)); lo.append(float(problem.variable_lower[atom.index])); hi.append(np.inf)
        elif atom.kind is ViolationKind.VARIABLE_UPPER:
            rows.append(eye.getrow(atom.index)); lo.append(-np.inf); hi.append(float(problem.variable_upper[atom.index]))
        else:
            raise ValueError(f"unsupported conflict atom kind: {atom.kind}")

    constraints = ()
    if rows:
        M = sparse.vstack(rows, format="csr")
        constraints = LinearConstraint(M, np.asarray(lo), np.asarray(hi))

    integrality = np.zeros(n, dtype=np.int32)
    for j, domain in enumerate(problem.domains):
        if domain in {VariableDomain.INTEGER.value, VariableDomain.BINARY.value}:
            integrality[j] = 1

    options: dict[str, Any] = {"disp": False}
    if time_limit_s is not None:
        options["time_limit"] = float(time_limit_s)
    # Variable bounds are represented only through selected atoms.  Keep x otherwise free.
    result = milp(
        c=np.zeros(n, dtype=np.float64),
        integrality=integrality,
        bounds=Bounds(np.full(n, -np.inf), np.full(n, np.inf)),
        constraints=constraints,
        options=options,
    )
    status = int(getattr(result, "status", -1))
    if status == 2:
        return "infeasible"
    if status == 0:
        return "feasible"
    if status == 1:
        return "limit"
    if status == 3:
        # With zero objective this normally means the feasibility model is feasible but
        # solver termination classified it unbounded; it does not prove infeasibility.
        return "feasible_or_unbounded"
    return "error"


def deletion_filter_conflict(
    problem: LinearProblem,
    *,
    max_checks: int = 250,
    time_limit_s_per_check: float | None = None,
) -> ConflictSetResult:
    """Find an irreducible set of row/variable bound sides by deletion filtering.

    This is a solver-agnostic fallback built on SciPy/HiGHS feasibility solves.  If the
    check budget is exhausted, ``complete`` and ``irreducible`` are false; callers must
    not present the partial set as an IIS.
    """
    if max_checks <= 0:
        raise ValueError("max_checks must be positive")
    if time_limit_s_per_check is not None and time_limit_s_per_check <= 0:
        raise ValueError("time_limit_s_per_check must be positive")

    active = _all_bound_atoms(problem)
    checks = 1
    initial = _feasibility_status_for_atoms(problem, active, time_limit_s=time_limit_s_per_check)
    if initial != "infeasible":
        return ConflictSetResult(
            atoms=(), irreducible=False, complete=True, checks=checks,
            solver_status=initial,
            message="full bound-side model was not proven infeasible",
        )

    pos = 0
    while pos < len(active):
        if checks >= max_checks:
            return ConflictSetResult(
                atoms=tuple(active), irreducible=False, complete=False, checks=checks,
                solver_status="check_limit",
                message="deletion-filter check budget exhausted before irreducibility proof",
            )
        trial = active[:pos] + active[pos + 1 :]
        status = _feasibility_status_for_atoms(problem, trial, time_limit_s=time_limit_s_per_check)
        checks += 1
        if status == "infeasible":
            # Atom was unnecessary; keep it deleted and test the new atom at this position.
            active = trial
            continue
        if status == "limit":
            return ConflictSetResult(
                atoms=tuple(active), irreducible=False, complete=False, checks=checks,
                solver_status="time_limit",
                message="a feasibility subcheck hit its time limit",
            )
        if status == "error":
            return ConflictSetResult(
                atoms=tuple(active), irreducible=False, complete=False, checks=checks,
                solver_status="solver_error",
                message="a feasibility subcheck failed",
            )
        # Removing this atom made the set feasible, so it is necessary.  By monotonicity
        # it will remain necessary after later deletions.
        pos += 1

    return ConflictSetResult(
        atoms=tuple(active), irreducible=True, complete=True, checks=checks,
        solver_status="infeasible",
        message="every retained atom is necessary for infeasibility",
    )

def _normalize_native_iis(backend: Any, native: Any) -> NormalizedIISEvidence:
    backend_name = getattr(getattr(backend, "manifest", None), "name", type(backend).__name__)
    # HiGHS development/public-style result.
    if hasattr(native, "row_indices") and hasattr(native, "col_indices"):
        return NormalizedIISEvidence(
            backend=str(backend_name),
            valid=bool(getattr(native, "valid", True)),
            row_indices=tuple(int(x) for x in getattr(native, "row_indices", ())),
            variable_indices=tuple(int(x) for x in getattr(native, "col_indices", ())),
            raw_type=type(native).__name__,
        )
    # SCIP-style normalized result.
    if hasattr(native, "constraint_names") and hasattr(native, "variable_names"):
        return NormalizedIISEvidence(
            backend=str(backend_name),
            valid=True,
            constraint_names=tuple(str(x) for x in getattr(native, "constraint_names", ())),
            variable_names=tuple(str(x) for x in getattr(native, "variable_names", ())),
            irreducible=bool(getattr(native, "irreducible", True)),
            raw_type=type(native).__name__,
        )
    raise TypeError(f"unsupported native IIS evidence type: {type(native)!r}")


def _try_native_iis(problem: LinearProblem, backend: Any) -> NormalizedIISEvidence | None:
    if hasattr(backend, "compute_iis"):
        return _normalize_native_iis(backend, backend.compute_iis(problem))
    if hasattr(backend, "generate_iis"):
        return _normalize_native_iis(backend, backend.generate_iis(problem))
    return None


def diagnose_infeasibility(
    problem: LinearProblem,
    *,
    backend: Any | None = None,
    include_elastic: bool = True,
    include_conflict: bool = False,
    conflict_max_checks: int = 250,
    time_limit_s: float | None = None,
) -> InfeasibilityReport:
    """Build typed infeasibility evidence without conflating IIS and relaxation.

    The function does not infer infeasibility merely because an elastic model has a
    positive objective; callers should normally invoke it after a solver has classified
    the original model infeasible.  Static contradictions *do* independently prove it.
    """
    static = find_static_infeasibility(problem)
    order: list[EvidenceKind] = []
    warnings: list[str] = []
    if static:
        order.append(EvidenceKind.STATIC_CONTRADICTION)

    iis = None
    if backend is not None:
        try:
            iis = _try_native_iis(problem, backend)
            if iis is not None:
                order.append(EvidenceKind.NATIVE_IIS)
        except (NotImplementedError, ValueError) as exc:
            warnings.append(f"native IIS unavailable for this model: {exc}")
        except Exception as exc:  # diagnostic failure must not corrupt the solve result
            warnings.append(f"native IIS failed: {type(exc).__name__}: {exc}")

    elastic = None
    if include_elastic:
        try:
            elastic = elastic_relaxation(problem, time_limit_s=time_limit_s)
            order.append(EvidenceKind.ELASTIC_RELAXATION)
        except Exception as exc:
            warnings.append(f"elastic relaxation failed: {type(exc).__name__}: {exc}")

    conflict = None
    if include_conflict:
        try:
            conflict = deletion_filter_conflict(
                problem,
                max_checks=conflict_max_checks,
                time_limit_s_per_check=time_limit_s,
            )
            order.append(EvidenceKind.DELETION_FILTER)
        except Exception as exc:
            warnings.append(f"deletion filter failed: {type(exc).__name__}: {exc}")

    elastic_proves_infeasible = bool(
        elastic is not None
        and elastic.feasible
        and elastic.objective is not None
        and elastic.objective > 1e-9
    )
    iis_proves_infeasible = bool(iis is not None and iis.valid)
    conflict_proves_infeasible = bool(conflict is not None and conflict.complete and conflict.irreducible)
    confirmed = bool(static) or iis_proves_infeasible or elastic_proves_infeasible or conflict_proves_infeasible
    return InfeasibilityReport(
        confirmed_infeasible=confirmed,
        static_issues=static,
        iis=iis,
        elastic=elastic,
        conflict=conflict,
        evidence_order=tuple(order),
        warnings=tuple(warnings),
    )
