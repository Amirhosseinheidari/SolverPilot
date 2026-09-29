"""Auditable, dependency-free LP selection research. Never enables production routing.

The learned policy is a cost-sensitive decision stump with a training-only SBS
fallback. Outcomes are repeated, independently verified solves; every failed or
late repeat receives PAR10, rather than disappearing from the training target.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from solverpilot.problem import LinearProblem


FEATURE_NAMES = (
    "log_variables", "log_rows", "log_nnz", "density", "equality_fraction",
    "bounded_fraction", "objective_density", "positive_fraction",
)
SCHEMA = "solverpilot.experimental.lp-stump.v1"


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def lp_features(problem: LinearProblem) -> tuple[float, ...]:
    """Cheap, outcome-free sparse features; no densification or solver probes."""
    if not isinstance(problem, LinearProblem) or problem.has_integer_variables:
        raise ValueError("learned selection requires a continuous LinearProblem")
    n, m, nnz = problem.n_variables, problem.n_constraints, problem.A.nnz
    eq = np.isfinite(problem.constraint_lower) & (
        problem.constraint_lower == problem.constraint_upper)
    bounded = np.isfinite(problem.variable_lower) & np.isfinite(problem.variable_upper)
    return (math.log1p(n), math.log1p(m), math.log1p(nnz), nnz / max(1, n * m),
            float(np.count_nonzero(eq)) / max(1, m),
            float(np.count_nonzero(bounded)) / max(1, n),
            float(np.count_nonzero(problem.c)) / max(1, n),
            float(np.count_nonzero(problem.A.data > 0)) / max(1, nnz))


@dataclass(frozen=True)
class LPObservation:
    instance: str
    group: str
    data_hash: str
    split: str
    environment_id: str
    features: tuple[float, ...]
    # Candidate IDs must include algorithm/settings, not only a backend's name.
    samples: Mapping[str, Sequence[tuple[float, bool]]]


def _validate(rows: Sequence[LPObservation], candidates: tuple[str, ...], cutoff: float):
    if not rows or not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("nonempty rows and unique candidates required")
    if isinstance(cutoff, bool) or not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be finite and positive")
    identities, hashes, environments, repetitions = set(), set(), set(), set()
    for row in rows:
        if (not row.instance or not row.group or not row.environment_id
                or row.split not in {"train", "validation", "test"}):
            raise ValueError("missing identity/group/environment or invalid split")
        if row.instance in identities or row.data_hash in hashes:
            raise ValueError("duplicate instance or model data hash")
        if len(row.data_hash) != 64 or any(c not in "0123456789abcdef" for c in row.data_hash):
            raise ValueError("invalid model data hash")
        identities.add(row.instance)
        hashes.add(row.data_hash)
        environments.add(row.environment_id)
        if len(row.features) != len(FEATURE_NAMES) or not all(math.isfinite(x) for x in row.features):
            raise ValueError("invalid feature vector")
        if set(row.samples) != set(candidates):
            raise ValueError("incomplete candidate coverage")
        for samples in row.samples.values():
            repetitions.add(len(samples))
            if not samples:
                raise ValueError("missing repeats")
            for wall, verified in samples:
                if not math.isfinite(wall) or wall < 0 or type(verified) is not bool:
                    raise ValueError("invalid time or independent verification flag")
    if len(environments) != 1 or len(repetitions) != 1:
        raise ValueError("mixed environments or unbalanced repeats")


def _costs(rows, candidates, cutoff):
    return np.asarray([[np.mean([wall if ok and wall <= cutoff else 10 * cutoff
                                 for wall, ok in row.samples[c]])
                        for c in candidates] for row in rows], dtype=float)


@dataclass(frozen=True)
class LPSelector:
    candidates: tuple[str, ...]
    baseline: str
    environment_id: str
    protocol_sha256: str
    cutoff_s: float
    feature_index: int
    threshold: float
    left: str
    right: str
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    training_instances: tuple[str, ...]
    training_groups: tuple[str, ...]
    training_hashes: tuple[str, ...]
    training_sha256: str

    def __post_init__(self):
        for field in ('candidates', 'lower', 'upper', 'training_instances', 'training_groups', 'training_hashes'):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if (not self.candidates or len(set(self.candidates)) != len(self.candidates)
                or any(not isinstance(c, str) or not c for c in self.candidates)
                or any(c not in self.candidates for c in (self.baseline, self.left, self.right))):
            raise ValueError("invalid selector candidates")
        if (type(self.feature_index) is not int or not -1 <= self.feature_index < len(FEATURE_NAMES)
                or not math.isfinite(self.threshold) or not math.isfinite(self.cutoff_s)
                or self.cutoff_s <= 0 or not self.environment_id):
            raise ValueError("invalid selector parameters")
        if (len(self.lower) != len(FEATURE_NAMES) or len(self.upper) != len(FEATURE_NAMES)
                or any(not math.isfinite(a) or not math.isfinite(b) or a > b
                       for a, b in zip(self.lower, self.upper))):
            raise ValueError("invalid feature support")
        for digest in (self.protocol_sha256, self.training_sha256, *self.training_hashes):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid digest")
        if not self.training_instances or not self.training_groups or not self.training_hashes:
            raise ValueError("missing training identities")

    def payload(self) -> dict:
        payload = {"schema": SCHEMA, "feature_names": FEATURE_NAMES, **asdict(self)}
        return {**payload, "sha256": _digest(payload)}

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.payload(), indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> LPSelector:
        raw = Path(path).read_bytes()
        if len(raw) > 2_000_000:
            raise ValueError("selector artifact too large")
        p = json.loads(raw)
        digest = p.pop("sha256")
        if digest != _digest(p) or p.pop("schema") != SCHEMA:
            raise ValueError("selector digest/schema mismatch")
        if tuple(p.pop("feature_names")) != FEATURE_NAMES:
            raise ValueError("feature schema mismatch")
        for key in ("candidates", "lower", "upper", "training_instances", "training_groups", "training_hashes"):
            p[key] = tuple(p[key])
        return cls(**p)


def fit_lp_selector(rows: Sequence[LPObservation], *, candidates: tuple[str, ...],
                    cutoff_s: float, protocol_sha256: str,
                    min_leaf_groups: int = 4, switch_margin_s: float = 0.005) -> LPSelector:
    """Fit only train rows; split/feature preprocessing never consumes held-out rows."""
    _validate(rows, candidates, cutoff_s)
    if any(r.split != "train" for r in rows):
        raise ValueError("only training rows may be fitted")
    if type(min_leaf_groups) is not int or min_leaf_groups < 2:
        raise ValueError("min_leaf_groups must be an integer >= 2")
    if not math.isfinite(switch_margin_s) or switch_margin_s < 0:
        raise ValueError("invalid switch margin")
    rows = sorted(rows, key=lambda r: r.instance)
    x = np.asarray([r.features for r in rows])
    costs = _costs(rows, candidates, cutoff_s)
    baseline = int(np.argmin(costs.mean(axis=0)))
    best = float(costs[:, baseline].sum())
    index, threshold, left, right = -1, 0.0, baseline, baseline
    for feature in range(x.shape[1]):
        values = np.unique(x[:, feature])
        for cut in (values[:-1] + (values[1:] - values[:-1]) / 2):
            mask = x[:, feature] <= cut
            if any(len({r.group for r, use in zip(rows, side) if use}) < min_leaf_groups
                   for side in (mask, ~mask)):
                continue
            choices = []
            for side in (mask, ~mask):
                means = costs[side].mean(axis=0)
                choice = int(np.argmin(means))
                if means[baseline] - means[choice] <= switch_margin_s:
                    choice = baseline
                choices.append(choice)
            loss = float(costs[mask, choices[0]].sum() + costs[~mask, choices[1]].sum())
            if loss < best - switch_margin_s * len(rows):
                best, index, threshold = loss, feature, float(cut)
                left, right = choices
    return LPSelector(candidates, candidates[baseline], rows[0].environment_id,
                      protocol_sha256, cutoff_s, index, threshold, candidates[left], candidates[right],
                      tuple(x.min(axis=0)), tuple(x.max(axis=0)),
                      tuple(r.instance for r in rows), tuple(sorted({r.group for r in rows})),
                      tuple(r.data_hash for r in rows), _digest([asdict(r) for r in rows]))


@dataclass(frozen=True)
class LPRouteDecision:
    candidate: str | None
    reason: str
    overhead_s: float


def _choose(features, model, environment_id, available):
    fallback = model.baseline if model.baseline in available else None
    if environment_id != model.environment_id:
        return fallback, "environment_mismatch"
    if (len(features) != len(FEATURE_NAMES) or any(not math.isfinite(v) or v < lo - 1e-12
            or v > hi + 1e-12 for v, lo, hi in zip(features, model.lower, model.upper))):
        return fallback, "outside_training_support"
    choice = model.baseline if model.feature_index < 0 else (
        model.left if features[model.feature_index] <= model.threshold else model.right)
    if choice not in available:
        return fallback, "candidate_unavailable"
    return choice, "learned_stump" if choice != model.baseline else "training_baseline"


def decide_lp_backend(problem: LinearProblem, model: LPSelector, *, environment_id: str,
                      available: Sequence[str]) -> LPRouteDecision:
    """Shadow decision only. Caller supplies measured-environment and available IDs.

    No backend runs here; the decision is not production authority. On abstention,
    a missing training baseline yields None, never an invented alternative.
    """
    start = perf_counter()
    choice, reason = _choose(lp_features(problem), model, environment_id, set(available))
    return LPRouteDecision(choice, reason, perf_counter() - start)


def evaluate_lp_selector(model: LPSelector, rows: Sequence[LPObservation], *,
                         decisions: Mapping[str, LPRouteDecision],
                         min_groups: int = 6, bootstrap_draws: int = 2000,
                         seed: int = 314159) -> dict:
    """Held-out PAR10 comparison, training-frozen SBS, group bootstrap and overhead.

    This research gate cannot set ProductionEvidence or enable automatic routing.
    A positive result is scoped to supplied observations, not public/OOD evidence.
    """
    _validate(rows, model.candidates, model.cutoff_s)
    if type(min_groups) is not int or min_groups < 2 or type(bootstrap_draws) is not int or bootstrap_draws < 100:
        raise ValueError("invalid group/bootstrap settings")
    if (len({r.split for r in rows}) != 1 or rows[0].split == "train"
            or rows[0].environment_id != model.environment_id):
        raise ValueError("held-out split and training environment required")
    if (set(model.training_instances) & {r.instance for r in rows}
            or set(model.training_groups) & {r.group for r in rows}
            or set(model.training_hashes) & {r.data_hash for r in rows}):
        raise ValueError("training/held-out identity or group leakage")
    if set(decisions) != {r.instance for r in rows}:
        raise ValueError("incomplete decision coverage")
    costs = _costs(rows, model.candidates, model.cutoff_s)
    baseline = costs[:, model.candidates.index(model.baseline)]
    policy, success, baseline_success, switches = [], [], [], 0
    for row, c in zip(rows, costs):
        d = decisions[row.instance]
        expected, _ = _choose(row.features, model, model.environment_id, set(model.candidates))
        if d.candidate != expected or not math.isfinite(d.overhead_s) or d.overhead_s < 0:
            raise ValueError("decision differs from frozen model or overhead is invalid")
        effective = [(wall+d.overhead_s, ok) for wall, ok in row.samples[d.candidate]]
        policy.append(float(np.mean([wall if ok and wall <= model.cutoff_s else 10*model.cutoff_s
                                     for wall, ok in effective])))
        success.append(sum(ok and wall <= model.cutoff_s for wall, ok in effective))
        baseline_success.append(sum(ok and wall <= model.cutoff_s for wall, ok in row.samples[model.baseline]))
        switches += d.candidate != model.baseline
    policy = np.asarray(policy)
    groups = sorted({r.group for r in rows})
    indices = [np.asarray([i for i, r in enumerate(rows) if r.group == g]) for g in groups]
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(bootstrap_draws):
        idx = np.concatenate([indices[j] for j in rng.integers(0, len(groups), len(groups))])
        ratios.append(float(policy[idx].sum() / max(baseline[idx].sum(), 1e-12)))
    ratio = float(policy.sum() / max(baseline.sum(), 1e-12))
    upper = float(np.quantile(ratios, .975))
    tail = float(np.quantile(policy / np.maximum(baseline, 1e-12), .9))
    gates = {"enough_groups": len(groups) >= min_groups, "at_least_two_switches": switches >= 2,
             "three_percent_improvement": ratio <= .97, "bootstrap_upper_below_one": upper < 1,
             "p90_slowdown_at_most_25_percent": tail <= 1.25,
             "no_lost_verified_solves": all(a >= b for a, b in zip(success, baseline_success))}
    return {"model_sha256": model.payload()["sha256"], "protocol_sha256": model.protocol_sha256,
            "split": rows[0].split, "instances": len(rows), "groups": len(groups),
            "baseline": model.baseline, "baseline_par10_mean_s": float(baseline.mean()),
            "policy_par10_mean_s": float(policy.mean()), "policy_to_baseline": ratio,
            "bootstrap_95pct": [float(np.quantile(ratios, .025)), upper],
            "bootstrap_draws": bootstrap_draws, "bootstrap_seed": seed,
            "p90_ratio": tail, "switches": switches, "gates": gates,
            "research_gate_passed": all(gates.values()), "production_authorized": False,
            "verified_repeats": sum(success), "baseline_verified_repeats": sum(baseline_success),
            "cost_definition": "add measured decision cost to each repeat before cutoff; unsuccessful/late = 10*cutoff"}
