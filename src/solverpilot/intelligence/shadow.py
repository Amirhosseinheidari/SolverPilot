"""Measure experimental LP selector suggestions without changing production routing.

The selector is explicit caller-owned Python code, receiving only immutable
pre-solve features. Training and family-separated held-out evaluation belong to
the benchmark pipeline; a shadow observation is never promotion evidence.
"""

from dataclasses import dataclass
from time import perf_counter
from solverpilot.problem import LinearProblem
from .fingerprint import feature_record_from_problem


@dataclass(frozen=True, slots=True)
class ShadowObservation:
    problem_data_hash: str
    policy_id: str
    production_backend: str
    suggested_backend: str | None
    eligible: bool
    feature_s: float
    prediction_s: float
    reason: str

    @property
    def changes_production_routing(self):
        return False

    @property
    def overhead_s(self):
        return self.feature_s + self.prediction_s


def observe_shadow_policy(problem, *, policy_id, selector, production_backend, eligible_backends):
    """Observe a synchronous suggestion; do not invoke this inside a solve budget.

    Caller-owned selector execution is not sandboxed or interrupted. Run costly
    models in the caller's worker. Failures are observations, not solver failures.
    Eligibility must come from the caller's capability checks for this problem.
    """
    if not isinstance(problem, LinearProblem) or problem.has_integer_variables:
        raise TypeError("shadow policy currently accepts continuous LP only")
    if not isinstance(policy_id, str) or not policy_id.strip() or not callable(selector):
        raise ValueError("a nonempty policy_id and callable selector are required")
    if isinstance(eligible_backends, str):
        raise TypeError("eligible_backends must be a collection of backend names")
    eligible = frozenset(eligible_backends)
    if any(not isinstance(b, str) or not b for b in eligible) or production_backend not in eligible:
        raise ValueError("production backend must belong to the eligible backend names")
    start = perf_counter()
    record = feature_record_from_problem(problem)
    built = perf_counter()
    suggestion = None
    try:
        proposed = selector(record)
        if isinstance(proposed, str) and proposed in eligible:
            suggestion = proposed
            reason = "eligible shadow suggestion; production backend unchanged"
        else:
            reason = "selector returned an ineligible backend"
    except Exception as exc:
        reason = "selector failed: " + type(exc).__name__
    return ShadowObservation(
        problem.data_hash,
        policy_id,
        production_backend,
        suggestion,
        suggestion is not None,
        built - start,
        perf_counter() - built,
        reason,
    )
