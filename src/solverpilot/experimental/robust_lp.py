"""Explicit experimental LP routing with production fallback on abstention.

The original v1 research selector and its frozen artifacts are unchanged.
This wrapper is never installed into the ordinary solve/production registry.
"""
from dataclasses import asdict, dataclass, replace
import math
from time import perf_counter

from solverpilot.experimental.learned_lp import decide_lp_backend
from solverpilot.plan import SolveBudget


@dataclass(frozen=True)
class RobustLPDecision:
    candidate: str
    reason: str
    learned_reason: str
    overhead_s: float


def decide_robust_lp(problem, model, *, environment_id, available, gain_guard=None,
                     cutoff_s=None, setup_elapsed_s=0.):
    start = perf_counter()
    d = decide_lp_backend(problem, model, environment_id=environment_id, available=available)
    abstained = d.reason in {"environment_mismatch", "outside_training_support", "candidate_unavailable"}
    fallback = abstained or d.candidate is None
    reason = "production_fallback" if fallback else "in_support"
    if not fallback and gain_guard is not None:
        permitted = gain_guard.permits(model, d.candidate,
            elapsed_s=setup_elapsed_s+perf_counter()-start, cutoff_s=cutoff_s)
        # Hashing/binding the guard is routing work too.
        permitted = permitted and setup_elapsed_s+perf_counter()-start <= gain_guard.overhead_limit_s
        if not permitted:
            fallback, reason = True, "gain_guard_fallback"
    return RobustLPDecision("production" if fallback else d.candidate,
                            reason,
                            d.reason, perf_counter()-start)


def solve_robust_lp(problem, model, *, environment_id, backends, time_limit_s=2.0,
                    tolerances=None, gain_guard=None):
    """Use one remaining budget, no failed-solve retry, and preserve trust checks.

    Caller supplies a measured environment ID (or explicit compatibility binding)
    and candidate-ID to backend objects. Setup and routing consume the same budget.
    Unavailable candidates cannot send abstentions to a training-only baseline.
    """
    from solverpilot import solve, solve_production
    if isinstance(time_limit_s, bool) or not math.isfinite(time_limit_s) or time_limit_s <= 0:
        raise ValueError("time_limit_s must be finite and positive")
    if "production" in backends:
        raise ValueError("production is reserved for the conservative planner")
    start = perf_counter()
    available = tuple(name for name, backend in backends.items() if backend.is_available())
    guard_options = {} if gain_guard is None else dict(gain_guard=gain_guard,
        cutoff_s=time_limit_s, setup_elapsed_s=perf_counter()-start)
    decision = decide_robust_lp(problem, model, environment_id=environment_id, available=available, **guard_options)
    remaining = time_limit_s-(perf_counter()-start)
    if remaining <= 0:
        raise TimeoutError("robust LP setup exhausted the call budget")
    budget = SolveBudget(wall_time_s=remaining)
    if decision.candidate == "production":
        result, _ = solve_production(problem, budget=budget, tolerances=tolerances)
    else:
        result = solve(problem, backend=backends[decision.candidate], budget=budget, tolerances=tolerances)
    raw = dict(result.raw_statistics)
    raw["experimental_lp_route"] = {**asdict(decision), "model_sha256": model.payload()["sha256"],
                                     "call_wall_s": perf_counter()-start,
                                     "automatic_production_routing_enabled": False}
    if gain_guard is not None:
        raw["experimental_lp_route"]["gain_guard_sha256"] = gain_guard.payload()['sha256']
    elapsed = perf_counter()-start
    raw['call_budget'] = {**dict(raw.get('call_budget', {})), 'requested_s': time_limit_s,
                          'elapsed_s': elapsed, 'within_budget': elapsed <= time_limit_s}
    trace = replace(result.trace, timings=replace(result.trace.timings, total_s=elapsed))
    return replace(result, raw_statistics=raw, trace=trace)
