"""LP dual candidate repair and exact, work-bounded implied box propagation.

Neither a projection nor a derived box is an optimality claim. The original
certificate checker must still validate the candidate and its corrected gap.
"""
from collections import deque
from collections.abc import Mapping
from dataclasses import replace
import math
from time import perf_counter
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


def equality_residual_lower_bound(problem, coefficients, lower, upper, *, max_pivots=512,
                                 max_visits=1000000, max_bits=8192, time_limit_s=None,
                                 diagnostics=None, preferred_columns=()):
    """Exact sparse equality-basis elimination, explicitly bounded and optional.

    Each pivot row is an exact linear combination of original equality rows.
    Substitution preserves the residual objective plus a constant. Inequality
    rows are never used as equalities. Returning None is an incomplete recovery,
    not evidence of infeasibility or nonoptimality.
    """
    start = perf_counter()
    if any(type(v) is not int or v < 0 for v in (max_pivots, max_visits, max_bits)):
        raise ValueError('recovery work limits must be nonnegative integers')
    if time_limit_s is not None and (isinstance(time_limit_s, bool)
            or not math.isfinite(time_limit_s) or time_limit_s <= 0):
        raise ValueError('recovery time limit must be finite and positive')
    a = problem.A.tocsr()
    equal = np.isfinite(problem.constraint_lower) & (problem.constraint_lower == problem.constraint_upper)
    residual = {j: v for j, v in enumerate(coefficients) if v}
    constant = rational(0); visits = 0; pivots = 0
    rows = {}; incidence = {}; bounds = {}
    preferred = set(preferred_columns)

    def finish(reason, value=None):
        if diagnostics is not None:
            diagnostics.update(reason=reason, pivots=pivots, nonzero_visits=visits,
                               elapsed_s=perf_counter()-start)
        return value

    def expired():
        return time_limit_s is not None and perf_counter()-start >= time_limit_s

    def small(values):
        return all(v.numerator.bit_length()+v.denominator.bit_length() <= max_bits for v in values)

    # Maintain incidence of the *transformed* rows. Elimination creates fill-in
    # in columns absent from an original row; original CSC adjacency is unsound
    # as a search heuristic because it can miss the only remaining pivot.
    for i in np.flatnonzero(equal):
        begin, end = a.indptr[i:i+2]
        visits += int(end-begin)
        if visits > max_visits: return finish('work_limit')
        if expired(): return finish('time_limit')
        row = {int(a.indices[k]): rational(a.data[k]) for k in range(begin, end) if a.data[k]}
        rhs = rational(problem.constraint_lower[i])
        if not small([rhs, *row.values()]): return finish('bit_limit')
        rows[int(i)], bounds[int(i)] = row, rhs
        for j in row: incidence.setdefault(j, set()).add(int(i))

    while True:
        if expired(): return finish('time_limit')
        bad = [j for j, v in residual.items() if not np.isfinite(lower[j] if v > 0 else upper[j])]
        priority = [j for j in preferred if residual.get(j) and incidence.get(j)]
        if not bad and not priority:
            tail = box_min([residual.get(j, rational(0)) for j in range(problem.n_variables)], lower, upper)
            return finish('bounded', constant+tail)
        if pivots >= max_pivots: return finish('pivot_limit')
        eligible = priority or [j for j in bad if incidence.get(j)]
        if not eligible: return finish('no_equality_pivot')
        j = min(eligible, key=lambda k: (len(incidence[k]), k))
        i = min(incidence[j], key=lambda k: (len(rows[k]), k))
        pivot = rows[i][j]
        row = {k: v/pivot for k, v in rows.pop(i).items()}
        rhs = bounds.pop(i)/pivot
        if not small([rhs, *row.values()]): return finish('bit_limit')
        multiplier = residual[j]
        constant += multiplier*rhs
        visits += len(row)
        if visits > max_visits: return finish('work_limit')
        for k, value in row.items():
            updated = residual.get(k, rational(0))-multiplier*value
            if updated: residual[k] = updated
            else: residual.pop(k, None)
            incidence[k].discard(i)
        if not small([constant, *residual.values()]): return finish('bit_limit')
        pivots += 1
        for other in sorted(incidence[j]):
            if expired(): return finish('time_limit')
            target = rows[other]
            multiplier = target[j]
            visits += len(row)
            if visits > max_visits: return finish('work_limit')
            bounds[other] -= multiplier*rhs
            for k, value in row.items():
                updated = target.get(k, rational(0))-multiplier*value
                if updated:
                    target[k] = updated
                    incidence.setdefault(k, set()).add(other)
                else:
                    target.pop(k, None)
                    incidence[k].discard(other)
            if not small([bounds[other], *target.values()]): return finish('bit_limit')


def dual_slack_residual_lower_bound(problem, coefficients, lower, upper, dual, *, x=None,
                                   basis=None, **options):
    """Exact basis recovery retaining nonnegative inequality slack penalties.

    For an upper row A_i*x+s_i=b_i, y_i>=0, its Lagrangian contributes
    y_i*s_i. For a lower row A_i*x-s_i=b_i, y_i<=0, it contributes
    -y_i*s_i. These penalties must NOT be dropped before elimination. Thus
    inequalities remain inequalities, regardless of numerical activity at x.
    The resulting bound corrects the original dual bound; it is not itself
    an optimum or an exact-feasibility certificate for the numerical primal.
    """
    start = perf_counter()
    from scipy import sparse
    m, n = problem.n_constraints, problem.n_variables
    if basis is not None:
        validate_lp_basis(problem, basis)
    matrix = sparse.vstack([problem.A, sparse.eye(n)], format='csr')
    original_lo = np.r_[problem.constraint_lower, problem.variable_lower]
    original_hi = np.r_[problem.constraint_upper, problem.variable_upper]
    equal = np.isfinite(original_lo) & (original_lo == original_hi)
    indices = np.flatnonzero(np.isfinite(original_lo) | np.isfinite(original_hi))
    if basis is not None:
        selected = set(basis['nonbasic_rows'])
        selected.update(m+j for j in range(n) if j not in set(basis['basic_columns'])
                        and (np.isfinite(original_lo[m+j]) or np.isfinite(original_hi[m+j])))
        # Never omit a contribution already present in the original dual.
        selected.update(np.flatnonzero(np.asarray(dual) != 0))
        indices = np.asarray(sorted(selected), dtype=int)
    activity = None if x is None else matrix@x
    slack_rows, signs, penalties, rhs = [], [], [], []
    for row, i in enumerate(indices):
        if equal[i]:
            rhs.append(original_lo[i])
        else:
            y = rational(dual[i])
            upper_side = y > 0 or (not y and np.isfinite(original_hi[i]))
            if not y and activity is not None:
                upper_side = abs(activity[i]-original_hi[i]) < abs(activity[i]-original_lo[i])
            endpoint = original_hi[i] if upper_side else original_lo[i]
            if not np.isfinite(endpoint):
                if options.get('diagnostics') is not None:
                    options['diagnostics'].update(reason='invalid_dual_endpoint')
                return None
            rhs.append(endpoint); slack_rows.append(row)
            signs.append(1. if upper_side else -1.); penalties.append(abs(y))
    count = len(penalties)
    slack = sparse.coo_matrix((signs, (slack_rows, range(count))), shape=(len(indices), count))
    matrix = sparse.hstack([matrix[indices], slack], format='csr')
    # Infinite upper slack bounds are conservative, including for two-sided
    # rows. No rounded subtraction of endpoints can shrink the exact domain.
    lo = np.r_[lower, np.zeros(count)]; hi = np.r_[upper, np.full(count, np.inf)]
    augmented = LinearProblem.from_data(A=matrix, c=np.zeros(len(lo)),
        constraint_lower=rhs, constraint_upper=rhs, variable_lower=lo, variable_upper=hi)
    if x is not None:
        # Numerical activity only chooses elimination order. All bounds and
        # slack penalties are still checked exactly on the original domain.
        near = 1e-7*np.maximum(1., np.abs(x))
        free = (np.abs(x-problem.variable_lower) > near) & (np.abs(x-problem.variable_upper) > near)
        options['preferred_columns'] = np.flatnonzero(free)
    if basis is not None:
        options['preferred_columns'] = basis['basic_columns']
    if options.get('time_limit_s') is not None:
        options['time_limit_s'] -= perf_counter()-start
        if options['time_limit_s'] <= 0:
            if options.get('diagnostics') is not None:
                options['diagnostics'].update(reason='time_limit', elapsed_s=perf_counter()-start)
            return None
    return equality_residual_lower_bound(augmented, [*coefficients, *penalties], lo, hi, **options)


def validate_lp_basis(problem, basis):
    if not isinstance(basis, Mapping) or not {'basic_columns', 'nonbasic_rows'} <= basis.keys():
        raise ValueError('basis requires basic_columns and nonbasic_rows')
    for name, size in (('basic_columns', problem.n_variables), ('nonbasic_rows', problem.n_constraints)):
        values = basis[name]
        if (not isinstance(values, (tuple, list)) or
                any(type(v) is not int or not 0 <= v < size for v in values) or
                len(set(values)) != len(values)):
            raise ValueError('invalid basis indices')
    if basis.get('problem_hash', problem.data_hash) != problem.data_hash:
        raise ValueError('basis problem hash mismatch')


def recover_lp_optimality(problem, x, dual, *, tolerances=None, time_limit_s=10.,
                          max_pivots=1024, max_visits=1000000, max_bits=65536, basis=None):
    """Explicit extended recovery; caller-owned x/dual flags are never trusted.

    Uses exact equality/slack combinations after ordinary domain recovery.
    Work limits do not imply a wall-clock guarantee; use process isolation when
    a hard stopping boundary is needed. No additional solver is invoked.
    """
    started = perf_counter()
    from .optimality import OptimalityCheck, verify_optimality
    if time_limit_s is not None and (isinstance(time_limit_s, bool) or
            not np.isfinite(time_limit_s) or time_limit_s <= 0):
        raise ValueError('recovery time limit must be finite and positive')
    for value in (max_pivots, max_visits, max_bits):
        if type(value) is not int or value < 0:
            raise ValueError('recovery work limits must be nonnegative integers')
    prepared = prepare_lp_dual(problem, dual)
    if prepared is None:
        raise ValueError('extended recovery requires a finite continuous LP dual')
    if basis is not None: validate_lp_basis(problem, basis)
    elapsed = perf_counter()-started
    remaining = None if time_limit_s is None else time_limit_s-elapsed
    if remaining is not None and remaining <= 0:
        return OptimalityCheck(False, False, False, np.inf, np.inf, np.inf,
            'recovery time limit exceeded during preparation',
            recovery_diagnostics={'wrapper': {'reason': 'time_limit', 'elapsed_s': elapsed}})
    result = verify_optimality(problem, x, prepared, tolerances=tolerances, extended_recovery=True,
        recovery_options=dict(time_limit_s=remaining, max_pivots=max_pivots, basis=basis,
                              max_visits=max_visits, max_bits=max_bits))
    elapsed = perf_counter()-started
    if time_limit_s is not None and elapsed >= time_limit_s:
        return replace(result, verified=False, reason='recovery time limit exceeded',
            recovery_diagnostics={**(result.recovery_diagnostics or {}),
                'wrapper': {'reason': 'time_limit', 'elapsed_s': elapsed}})
    return result
