"""LP dual candidate repair and exact, work-bounded implied box propagation.

Neither a projection nor a derived box is an optimality claim. The original
certificate checker must still validate the candidate and its corrected gap.
"""
from collections import deque
import numpy as np

from solverpilot.problem import LinearProblem
from ._certificate_arithmetic import rational, bounded_float, box_min


def prepare_lp_dual(problem, dual):
    """Project multipliers onto the signs allowed by finite original bounds.

    Zeroing an illegal multiplier can change stationarity: always recheck it.
    The raw solver candidate is never mutated and no tolerance is used here.
    """
    if not isinstance(problem, LinearProblem) or problem.has_integer_variables:
        return None
    try:
        y = np.array(dual, dtype=float, copy=True)
    except (TypeError, ValueError):
        return None
    lo = np.r_[problem.constraint_lower, problem.variable_lower]
    hi = np.r_[problem.constraint_upper, problem.variable_upper]
    if y.shape != lo.shape or not np.isfinite(y).all():
        return None
    y[((y > 0) & ~np.isfinite(hi)) | ((y < 0) & ~np.isfinite(lo))] = 0.
    return y


def implied_lp_box(problem, *, max_visits=50000, max_passes=3):
    """Enclose the exact feasible set using original rows and outward rounding.

    For a*x + rest <= b, a finite lower bound on rest gives an upper
    bound on a*x. The lower-row case uses an upper bound on rest. Exact
    binary64 rational arithmetic prevents cancellation from shrinking the box.
    A deterministic nonzero-visit cap bounds work; partial propagation is safe.
    A contradictory box is discarded, never used as an infeasibility proof.
    """
    lower, upper = problem.variable_lower.copy(), problem.variable_upper.copy()
    a = problem.A.tocsr()
    visits = 0
    for _ in range(max_passes):
        changed = False
        for i in range(a.shape[0]):
            start, end = a.indptr[i:i+2]
            width = end-start
            if visits + width > max_visits:
                return lower, upper
            visits += width
            indices = a.indices[start:end]
            coefs = [rational(v) for v in a.data[start:end]]
            if not coefs:
                continue
            # Snapshot the row: every update below follows from this same box.
            for row_bound, is_upper in ((problem.constraint_upper[i], True),
                                        (problem.constraint_lower[i], False)):
                if not np.isfinite(row_bound):
                    continue
                terms = []
                for j, c in zip(indices, coefs):
                    endpoint = (lower[j] if c > 0 else upper[j]) if is_upper else (
                        upper[j] if c > 0 else lower[j])
                    terms.append(None if not np.isfinite(endpoint) else c*rational(endpoint))
                total = sum((v for v in terms if v is not None), rational(0))
                missing = sum(v is None for v in terms)
                for j, c, term in zip(indices, coefs, terms):
                    if not c or missing - (term is None):
                        continue
                    exact = (rational(row_bound) - total + (term or 0))/c
                    toward_upper = is_upper == (c > 0)
                    rounded = bounded_float(exact)
                    if not np.isfinite(rounded):
                        continue
                    if toward_upper and rational(rounded) < exact:
                        rounded = np.nextafter(rounded, np.inf)
                    elif not toward_upper and rational(rounded) > exact:
                        rounded = np.nextafter(rounded, -np.inf)
                    if toward_upper and rounded < upper[j]:
                        upper[j] = rounded
                        changed = True
                    elif not toward_upper and rounded > lower[j]:
                        lower[j] = rounded
                        changed = True
                    if lower[j] > upper[j]:
                        return problem.variable_lower.copy(), problem.variable_upper.copy()
        if not changed:
            break
    return lower, upper


def row_residual_lower_bound(problem, coefficients, lower, upper, *, max_visits=50000):
    """Bound a residual linear form using exact substitutions of original rows.

    r*x = (r_j/a_ij)*(A_i*x) + the remaining linear form. Choose the
    finite row endpoint giving a lower bound on that first term, then eliminate
    x_j exactly. Never reintroduce previously eliminated variables. This also
    handles coupled free variables whose individual boxes cannot be bounded.
    A failure to finish simply returns None, with no relaxed feasibility test.
    """
    a = problem.A.tocsr(); columns = a.tocsc()
    residual = list(coefficients)
    constant = rational(0)
    eliminated, attempted = set(), set()
    visits = 0
    pending = deque(j for j, c in enumerate(residual) if c and
                    not np.isfinite(lower[j] if c > 0 else upper[j]))
    queued = set(pending)
    while pending:
        j = pending.popleft(); queued.discard(j); attempted.add(j)
        if not residual[j] or np.isfinite(lower[j] if residual[j] > 0 else upper[j]):
            continue
        choices = columns.indices[columns.indptr[j]:columns.indptr[j+1]]
        for i in sorted(choices, key=lambda i: a.indptr[i+1]-a.indptr[i]):
            begin, end = a.indptr[i:i+2]
            visits += end-begin
            if visits > max_visits:
                return None
            indices = a.indices[begin:end]
            if any(k in eliminated for k in indices):
                continue
            data = a.data[begin:end]
            pivot = next(v for k, v in zip(indices, data) if k == j)
            if not pivot:
                continue
            multiplier = residual[j]/rational(pivot)
            endpoint = problem.constraint_lower[i] if multiplier > 0 else problem.constraint_upper[i]
            if not np.isfinite(endpoint):
                continue
            updates = [residual[k]-multiplier*rational(v) for k, v in zip(indices, data)]
            if any(v.numerator.bit_length()+v.denominator.bit_length() > 8192 for v in updates):
                continue
            constant += multiplier*rational(endpoint)
            for k, value in zip(indices, updates):
                residual[k] = value
                if (value and k not in attempted and k not in queued and
                        not np.isfinite(lower[k] if value > 0 else upper[k])):
                    pending.append(k); queued.add(k)
            eliminated.add(j)
            break
    tail = box_min(residual, lower, upper)
    return None if tail is None else constant+tail


def equality_residual_lower_bound(problem, coefficients, lower, upper, *, max_pivots=128,
                                 max_visits=100000, max_bits=8192):
    """Exact sparse equality-basis elimination, explicitly bounded and optional.

    Each pivot row is an exact linear combination of original equality rows.
    Substitution preserves the residual objective plus a constant. Inequality
    rows are never used as equalities. Returning None is an incomplete recovery,
    not evidence of infeasibility or nonoptimality.
    """
    a = problem.A.tocsr(); columns = a.tocsc()
    equal = np.isfinite(problem.constraint_lower) & (problem.constraint_lower == problem.constraint_upper)
    residual = {j: v for j, v in enumerate(coefficients) if v}
    pivots = {}; constant = rational(0); visits = 0

    def small(values):
        return all(v.numerator.bit_length()+v.denominator.bit_length() <= max_bits for v in values)

    for _ in range(max_pivots):
        bad = [j for j, v in residual.items() if not np.isfinite(lower[j] if v > 0 else upper[j])]
        if not bad:
            tail = box_min([residual.get(j, rational(0)) for j in range(problem.n_variables)], lower, upper)
            return None if tail is None else constant+tail
        j = min(bad, key=lambda k: columns.indptr[k+1]-columns.indptr[k])
        found = False
        choices = columns.indices[columns.indptr[j]:columns.indptr[j+1]]
        for i in sorted((i for i in choices if equal[i]), key=lambda i: a.indptr[i+1]-a.indptr[i]):
            begin, end = a.indptr[i:i+2]; visits += end-begin
            if visits > max_visits: return None
            row = {int(a.indices[k]): rational(a.data[k]) for k in range(begin, end) if a.data[k]}
            rhs = rational(problem.constraint_lower[i])
            for k, (basis, bound) in pivots.items():
                multiplier = row.get(k, rational(0))
                if not multiplier: continue
                visits += len(basis)
                if visits > max_visits: return None
                rhs -= multiplier*bound
                for col, value in basis.items():
                    updated = row.get(col, rational(0))-multiplier*value
                    if updated: row[col] = updated
                    else: row.pop(col, None)
                if not small([rhs, *row.values()]): return None
            pivot = row.get(j)
            if not pivot: continue
            row = {k: v/pivot for k, v in row.items()}; rhs /= pivot
            if not small([rhs, *row.values()]): return None
            multiplier = residual[j]
            new_constant = constant+multiplier*rhs
            if not small([new_constant]): return None
            visits += len(row)
            if visits > max_visits: return None
            for k, value in row.items():
                updated = residual.get(k, rational(0))-multiplier*value
                if updated: residual[k] = updated
                else: residual.pop(k, None)
            if not small(residual.values()): return None
            constant = new_constant; pivots[j] = (row, rhs); found = True
            break
        if not found: return None
    tail = box_min([residual.get(j, rational(0)) for j in range(problem.n_variables)], lower, upper)
    return None if tail is None else constant+tail


def recover_lp_optimality(problem, x, dual, *, tolerances=None):
    """Explicit extended recovery; caller-owned x/dual flags are never trusted.

    Uses exact original equality combinations after ordinary domain recovery.
    Work limits do not imply a wall-clock guarantee; use process isolation when
    a hard stopping boundary is needed. No additional solver is invoked.
    """
    from .optimality import verify_optimality
    prepared = prepare_lp_dual(problem, dual)
    if prepared is None:
        raise ValueError('extended recovery requires a finite continuous LP dual')
    return verify_optimality(problem, x, prepared, tolerances=tolerances, extended_recovery=True)
