"""Small, versioned LP routing research with family-isolated calibration.

The v1 selector and its artifacts are deliberately unchanged.  Observations use
the existing LPObservation shape, but this module validates its own feature and
split schema.  A group must be a canonical application/generator family, not a
random seed.  Family provenance must be audited before supplying observations.
Neither fitting, calibration nor evaluation enables automatic production routes.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
import math
from numbers import Real
from pathlib import Path
from time import perf_counter

import numpy as np

from .learned_lp import LPObservation, _digest, lp_features
from .lp_gain import implementation_id


FEATURE_NAMES_V2 = (
    "log_variables", "log_rows", "log_nnz", "density", "equality_fraction",
    "bounded_fraction", "objective_density", "positive_fraction",
    "log_mean_row_nnz", "log_mean_column_nnz", "free_variable_fraction",
    "fixed_variable_fraction", "log_coefficient_range",
)
SCHEMA_V2 = "solverpilot.experimental.lp-small-tree.v2"
GUARD_SCHEMA_V2 = "solverpilot.experimental.lp-leaf-gain.v2"
_FRACTIONS = (3, 4, 5, 6, 7, 10, 11)


def _real(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value) or (value <= 0 if positive else value < 0)):
        raise ValueError(f"invalid {name}")
    return float(value)


def _integer(value, name, minimum, maximum=None):
    if (type(value) is not int or value < minimum
            or (maximum is not None and value > maximum)):
        raise ValueError(f"invalid {name}")
    return value


def _cutoff(value):
    value = _real(value, "cutoff", positive=True)
    if not math.isfinite(10 * value):
        raise ValueError("PAR10 cutoff overflows")
    return value


def _names(values, name, *, nonempty=True):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"invalid {name}")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValueError(f"invalid {name}") from exc
    if ((nonempty and not result) or any(not isinstance(v, str) or not v for v in result)
            or len(set(result)) != len(result)):
        raise ValueError(f"invalid {name}")
    return result


def _sha(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("invalid digest")
    return value


def _features(values):
    try:
        result = tuple(_real(v, "feature") for v in values)
    except TypeError as exc:
        raise ValueError("invalid feature vector") from exc
    if (len(result) != len(FEATURE_NAMES_V2)
            or any(result[i] > 1 for i in _FRACTIONS)):
        raise ValueError("invalid v2 feature vector")
    return result


def lp_features_v2(problem):
    """Sparse O(nnz+n+m) structural features; no probes or dense matrices.

    The coefficient range is log(max(abs(nonzero A)))-log(min(abs(nonzero A))).
    Taking the logs separately avoids overflow for extreme binary64 ranges.
    """
    original = lp_features(problem)
    n, m, nnz = problem.n_variables, problem.n_constraints, problem.A.nnz
    lower, upper = problem.variable_lower, problem.variable_upper
    free = ~np.isfinite(lower) & ~np.isfinite(upper)
    fixed = np.isfinite(lower) & (lower == upper)
    coefficients = np.abs(problem.A.data[problem.A.data != 0])
    spread = (math.log(float(coefficients.max())) - math.log(float(coefficients.min()))
              if coefficients.size else 0.)
    return _features((*original, math.log1p(nnz / max(1, m)),
                      math.log1p(nnz / max(1, n)),
                      float(np.count_nonzero(free)) / max(1, n),
                      float(np.count_nonzero(fixed)) / max(1, n), spread))


def _validate_rows(rows, candidates=None, split=None):
    rows = tuple(rows)
    if not rows:
        raise ValueError("nonempty observations required")
    if any(not isinstance(row, LPObservation) or not isinstance(row.samples, Mapping) for row in rows):
        raise ValueError("LPObservation records with sample mappings required")
    candidates = _names(tuple(rows[0].samples) if candidates is None else candidates,
                        "candidate coverage")
    instances, hashes, environments, repeats = set(), set(), set(), set()
    for row in rows:
        if (not isinstance(row.instance, str) or not row.instance
                or not isinstance(row.group, str) or not row.group
                or not isinstance(row.environment_id, str) or not row.environment_id
                or row.split not in {"train", "calibration", "test"}
                or (split is not None and row.split != split)):
            raise ValueError("invalid identity, family, environment or split")
        _sha(row.data_hash)
        if row.instance in instances or row.data_hash in hashes:
            raise ValueError("duplicate instance or canonical data hash")
        instances.add(row.instance)
        hashes.add(row.data_hash)
        environments.add(row.environment_id)
        _features(row.features)
        if set(row.samples) != set(candidates):
            raise ValueError("incomplete candidate coverage")
        for values in row.samples.values():
            if not isinstance(values, (tuple, list)) or not values:
                raise ValueError("missing repeats")
            repeats.add(len(values))
            for sample in values:
                if not isinstance(sample, (tuple, list)) or len(sample) != 2:
                    raise ValueError("invalid repeat")
                wall, verified = sample
                _real(wall, "repeat wall time")
                if type(verified) is not bool:
                    raise ValueError("invalid independent verification flag")
    if len(environments) != 1 or len(repeats) != 1:
        raise ValueError("mixed environments or unbalanced repeats")
    return rows


def _identity_sets(rows):
    return ({r.instance for r in rows}, {r.group for r in rows},
            {r.data_hash for r in rows})


def _reject_overlap(left, right):
    if any(a & b for a, b in zip(left, right)):
        raise ValueError("instance, family or canonical hash leakage")


def validate_lp_splits_v2(training, calibration, test):
    """Validate declared canonical families and hashes before collecting outcomes.

    This also accepts placeholder observations with balanced, finite samples;
    it cannot establish the truth of externally declared family provenance.
    """
    splits = {name: _validate_rows(rows, split=name) for name, rows in
              (("train", training), ("calibration", calibration), ("test", test))}
    names = tuple(splits)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            _reject_overlap(_identity_sets(splits[name]), _identity_sets(splits[other]))
    if len({rows[0].environment_id for rows in splits.values()}) != 1:
        raise ValueError("mixed split environments")
    return {"passed": True, "instances": {k: len(v) for k, v in splits.items()},
            "families": {k: len({r.group for r in v}) for k, v in splits.items()},
            "production_authorized": False}


def _row_payload(row):
    return {"instance": row.instance, "group": row.group, "data_hash": row.data_hash,
            "split": row.split, "environment_id": row.environment_id,
            "features": _features(row.features),
            "samples": {k: tuple((float(w), ok) for w, ok in row.samples[k])
                        for k in sorted(row.samples)}}


def _score(samples, cutoff, overhead=0.):
    successes = [bool(ok and wall + overhead <= cutoff) for wall, ok in samples]
    costs = [wall + overhead if ok else 10 * cutoff
             for (wall, _), ok in zip(samples, successes)]
    return math.fsum(value / len(costs) for value in costs), sum(successes)


def _read_artifact(path, schema):
    raw = Path(path).read_bytes()
    if len(raw) > 2_000_000:
        raise ValueError("artifact too large")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate artifact key")
            result[key] = value
        return result
    data = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(data, dict):
        raise ValueError("invalid artifact")
    digest = data.pop("sha256", None)
    if digest != _digest(data) or data.pop("schema", None) != schema:
        raise ValueError("artifact digest/schema mismatch")
    return data


@dataclass(frozen=True)
class LPNodeV2:
    feature_index: int
    threshold: float
    left: int
    right: int
    candidate: str

    def __post_init__(self):
        _integer(self.feature_index, "feature index", -1, len(FEATURE_NAMES_V2) - 1)
        object.__setattr__(self, "threshold", _real(self.threshold, "threshold"))
        _integer(self.left, "left node", -1)
        _integer(self.right, "right node", -1)
        if not isinstance(self.candidate, str) or not self.candidate:
            raise ValueError("invalid node candidate")
        if self.feature_index == -1 and (self.left != -1 or self.right != -1):
            raise ValueError("leaf cannot have children")
        if self.feature_index >= 0 and (self.left < 0 or self.right < 0):
            raise ValueError("split requires children")


@dataclass(frozen=True)
class LPSelectorV2:
    candidates: tuple[str, ...]
    baseline: str
    environment_id: str
    protocol_sha256: str
    cutoff_s: float
    nodes: tuple[LPNodeV2, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    training_instances: tuple[str, ...]
    training_groups: tuple[str, ...]
    training_hashes: tuple[str, ...]
    training_sha256: str

    def __post_init__(self):
        for field in ("candidates", "training_instances", "training_groups", "training_hashes"):
            object.__setattr__(self, field, _names(getattr(self, field), field))
        if (self.baseline not in self.candidates or "production" in self.candidates
                or not isinstance(self.environment_id, str) or not self.environment_id):
            raise ValueError("invalid model baseline, candidates or environment")
        if len(self.training_instances) != len(self.training_hashes):
            raise ValueError("unbalanced training identities")
        for digest in (self.protocol_sha256, self.training_sha256, *self.training_hashes):
            _sha(digest)
        object.__setattr__(self, "cutoff_s", _cutoff(self.cutoff_s))
        for field in ("lower", "upper"):
            object.__setattr__(self, field, _features(getattr(self, field)))
        if any(a > b for a, b in zip(self.lower, self.upper)):
            raise ValueError("invalid feature support")
        nodes = tuple(self.nodes)
        if not 1 <= len(nodes) <= 7 or any(type(n) is not LPNodeV2 for n in nodes):
            raise ValueError("invalid bounded tree")
        visited = set()
        def visit(index, depth):
            if index in visited or not 0 <= index < len(nodes) or depth > 2:
                raise ValueError("tree cycle, shared child, bad index or excessive depth")
            visited.add(index)
            node = nodes[index]
            if node.candidate not in self.candidates:
                raise ValueError("unknown tree candidate")
            if node.feature_index >= 0:
                if not self.lower[node.feature_index] <= node.threshold <= self.upper[node.feature_index]:
                    raise ValueError("split outside feature support")
                visit(node.left, depth + 1)
                visit(node.right, depth + 1)
        visit(0, 0)
        if len(visited) != len(nodes):
            raise ValueError("unreachable tree nodes")
        object.__setattr__(self, "nodes", nodes)

    def payload(self):
        data = {"schema": SCHEMA_V2, "feature_names": FEATURE_NAMES_V2, **asdict(self)}
        return {**data, "sha256": _digest(data)}

    def save(self, path):
        Path(path).write_text(json.dumps(self.payload(), indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = _read_artifact(path, SCHEMA_V2)
        if tuple(data.pop("feature_names", ())) != FEATURE_NAMES_V2:
            raise ValueError("v2 feature schema mismatch")
        try:
            data["nodes"] = tuple(LPNodeV2(**node) for node in data["nodes"])
            return cls(**data)
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid model artifact") from exc


def fit_lp_selector_v2(rows, *, candidates, cutoff_s, protocol_sha256,
                       min_leaf_families=4, max_depth=2, switch_margin_s=.005):
    """Fit one deterministic shallow cost tree with equal total family weights."""
    candidates = _names(candidates, "candidates")
    cutoff_s = _cutoff(cutoff_s)
    _sha(protocol_sha256)
    _integer(min_leaf_families, "minimum leaf families", 2)
    _integer(max_depth, "maximum depth", 0, 2)
    switch_margin_s = _real(switch_margin_s, "switch margin")
    rows = sorted(_validate_rows(rows, candidates, "train"), key=lambda r: r.instance)
    x = np.asarray([_features(r.features) for r in rows])
    counts = Counter(r.group for r in rows)
    weights = np.asarray([1. / (len(counts) * counts[r.group]) for r in rows])
    costs = np.asarray([[_score(r.samples[c], cutoff_s)[0] for c in candidates] for r in rows])
    baseline = int(np.argmin(weights @ costs))
    nodes = []
    def leaf_choice(indices):
        weighted = weights[indices] @ costs[indices]
        best = int(np.argmin(weighted))
        if weighted[baseline] - weighted[best] <= switch_margin_s * weights[indices].sum():
            best = baseline
        return best, float(weighted[best])
    def grow(indices, depth):
        choice, loss = leaf_choice(indices)
        slot = len(nodes)
        nodes.append(LPNodeV2(-1, 0., -1, -1, candidates[choice]))
        if depth >= max_depth:
            return slot
        best = None
        for feature in range(len(FEATURE_NAMES_V2)):
            values = np.unique(x[indices, feature])
            for low, high in zip(values[:-1], values[1:]):
                cut = float(low / 2 + high / 2)
                left = indices[x[indices, feature] <= cut]
                right = indices[x[indices, feature] > cut]
                if any(len({rows[i].group for i in side}) < min_leaf_families
                       for side in (left, right)):
                    continue
                candidate_loss = leaf_choice(left)[1] + leaf_choice(right)[1]
                if candidate_loss < loss - switch_margin_s * weights[indices].sum():
                    loss, best = candidate_loss, (feature, cut, left, right)
        if best is not None:
            feature, cut, left, right = best
            left_id, right_id = grow(left, depth + 1), grow(right, depth + 1)
            nodes[slot] = LPNodeV2(feature, cut, left_id, right_id, candidates[choice])
        return slot
    grow(np.arange(len(rows)), 0)
    return LPSelectorV2(candidates, candidates[baseline], rows[0].environment_id,
        protocol_sha256, cutoff_s, tuple(nodes), tuple(x.min(axis=0)), tuple(x.max(axis=0)),
        tuple(r.instance for r in rows), tuple(sorted(counts)), tuple(r.data_hash for r in rows),
        _digest([_row_payload(r) for r in rows]))


def choose_lp_v2(features, model, environment_id, available):
    """Return (candidate, reason, leaf_id); abstention has no qualified leaf."""
    available = set(available)
    fallback = model.baseline if model.baseline in available else None
    if environment_id != model.environment_id:
        return fallback, "environment_mismatch", None
    try:
        features = _features(features)
    except ValueError:
        return fallback, "outside_training_support", None
    if any(v < lo - 1e-12 or v > hi + 1e-12 for v, lo, hi in zip(features, model.lower, model.upper)):
        return fallback, "outside_training_support", None
    index = 0
    while model.nodes[index].feature_index >= 0:
        node = model.nodes[index]
        index = node.left if features[node.feature_index] <= node.threshold else node.right
    candidate = model.nodes[index].candidate
    if candidate not in available:
        return fallback, "candidate_unavailable", None
    return candidate, ("training_baseline" if candidate == model.baseline else "learned_tree_v2"), index


@dataclass(frozen=True)
class LPRouteDecisionV2:
    candidate: str | None
    reason: str
    overhead_s: float
    leaf_id: int | None


def decide_lp_backend_v2(problem, model, *, environment_id, available):
    """Timed shadow decision; solving and production fallback remain caller-owned."""
    start = perf_counter()
    candidate, reason, leaf = choose_lp_v2(lp_features_v2(problem), model, environment_id, available)
    return LPRouteDecisionV2(candidate, reason, perf_counter() - start, leaf)


@dataclass(frozen=True)
class LPLeafGainV2:
    leaf_id: int
    candidate: str
    minimum_family_gain_s: float
    families: tuple[str, ...]

    def __post_init__(self):
        _integer(self.leaf_id, "leaf id", 0, 6)
        if not isinstance(self.candidate, str) or not self.candidate:
            raise ValueError("invalid leaf candidate")
        object.__setattr__(self, "minimum_family_gain_s",
                           _real(self.minimum_family_gain_s, "gain", positive=True))
        object.__setattr__(self, "families", _names(self.families, "calibration families"))


@dataclass(frozen=True)
class LPLeafGainGuardV2:
    model_sha256: str
    implementation_sha256: str
    environment_id: str
    cutoff_s: float
    protocol_sha256: str
    calibration_sha256: str
    overhead_limit_s: float
    setup_charge_s: float
    min_families: int
    certified_leaves: tuple[LPLeafGainV2, ...]
    calibration_instances: tuple[str, ...]
    calibration_groups: tuple[str, ...]
    calibration_hashes: tuple[str, ...]

    def __post_init__(self):
        for field in ("model_sha256", "implementation_sha256", "protocol_sha256", "calibration_sha256"):
            _sha(getattr(self, field))
        if not isinstance(self.environment_id, str) or not self.environment_id:
            raise ValueError("invalid guard environment")
        object.__setattr__(self, "cutoff_s", _cutoff(self.cutoff_s))
        for field in ("overhead_limit_s", "setup_charge_s"):
            object.__setattr__(self, field, _real(getattr(self, field), field))
        _integer(self.min_families, "minimum calibration families", 2)
        for field in ("calibration_instances", "calibration_groups", "calibration_hashes"):
            object.__setattr__(self, field, _names(getattr(self, field), field))
        if len(self.calibration_instances) != len(self.calibration_hashes):
            raise ValueError("unbalanced calibration identities")
        for digest in self.calibration_hashes:
            _sha(digest)
        leaves = tuple(self.certified_leaves)
        if (any(type(leaf) is not LPLeafGainV2 for leaf in leaves)
                or len({leaf.leaf_id for leaf in leaves}) != len(leaves)
                or any(len(leaf.families) < self.min_families
                       or not set(leaf.families) <= set(self.calibration_groups) for leaf in leaves)):
            raise ValueError("invalid calibrated leaves")
        object.__setattr__(self, "certified_leaves", leaves)

    @property
    def candidate_gains(self):
        return tuple((name, min(leaf.minimum_family_gain_s for leaf in self.certified_leaves
                                if leaf.candidate == name))
                     for name in sorted({leaf.candidate for leaf in self.certified_leaves}))

    def permits_bound_candidate(self, candidate, leaf_id):
        return (type(leaf_id) is int and any(leaf.leaf_id == leaf_id and leaf.candidate == candidate
                                           for leaf in self.certified_leaves))

    def permits(self, model, candidate, *, elapsed_s, cutoff_s, leaf_id=None):
        try:
            elapsed = _real(elapsed_s, "elapsed")
            cutoff = _cutoff(cutoff_s)
        except ValueError:
            return False
        return (self.permits_bound_candidate(candidate, leaf_id)
                and self.model_sha256 == model.payload()["sha256"]
                and self.implementation_sha256 == implementation_id()
                and self.environment_id == model.environment_id
                and self.protocol_sha256 == model.protocol_sha256
                and self.cutoff_s == cutoff == model.cutoff_s
                and elapsed <= self.overhead_limit_s)

    def payload(self):
        data = {"schema": GUARD_SCHEMA_V2, **asdict(self)}
        return {**data, "sha256": _digest(data)}

    def save(self, path):
        Path(path).write_text(json.dumps(self.payload(), indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = _read_artifact(path, GUARD_SCHEMA_V2)
        try:
            data["certified_leaves"] = tuple(LPLeafGainV2(**leaf) for leaf in data["certified_leaves"])
            return cls(**data)
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid guard artifact") from exc


def calibrate_lp_guard_v2(model, rows, *, overhead_limit_s=.005, setup_charge_s=0., min_families=4):
    """Certify individual leaves using independent calibration families only."""
    overhead_limit_s = _real(overhead_limit_s, "overhead limit")
    setup_charge_s = _real(setup_charge_s, "setup charge")
    _integer(min_families, "minimum calibration families", 2)
    rows = sorted(_validate_rows(rows, (*model.candidates, "production"), "calibration"),
                  key=lambda r: r.instance)
    _reject_overlap((set(model.training_instances), set(model.training_groups), set(model.training_hashes)),
                    _identity_sets(rows))
    if rows[0].environment_id != model.environment_id:
        raise ValueError("calibration environment mismatch")
    gains, lost = {}, set()
    for row in rows:
        candidate, reason, leaf = choose_lp_v2(row.features, model, model.environment_id, model.candidates)
        if leaf is None:
            continue
        selected, selected_ok = _score(row.samples[candidate], model.cutoff_s,
                                       overhead_limit_s + setup_charge_s)
        production, production_ok = _score(row.samples["production"], model.cutoff_s)
        if selected_ok < production_ok:
            lost.add(leaf)
        gains.setdefault((leaf, candidate), {}).setdefault(row.group, []).append(production - selected)
    allowed = []
    for (leaf, candidate), families in sorted(gains.items()):
        floor = min(float(np.mean(values)) for values in families.values())
        if leaf not in lost and len(families) >= min_families and floor > 0:
            allowed.append(LPLeafGainV2(leaf, candidate, floor, tuple(sorted(families))))
    return LPLeafGainGuardV2(model.payload()["sha256"], implementation_id(), model.environment_id,
        model.cutoff_s, model.protocol_sha256, _digest([_row_payload(r) for r in rows]),
        overhead_limit_s, setup_charge_s, min_families, tuple(allowed),
        tuple(r.instance for r in rows), tuple(sorted({r.group for r in rows})),
        tuple(r.data_hash for r in rows))


def _ratios(numerator, denominator):
    a, b = np.asarray(numerator), np.asarray(denominator)
    return np.where((a == 0) & (b == 0), 1., a / np.maximum(b, 1e-12))


def evaluate_lp_selector_v2(model, rows, *, decisions, guard=None, session_setup_s=0.,
                            reuse_count=1, baselines=("production",), baseline_setup_s=None,
                            min_families=6, bootstrap_draws=2000, seed=314159):
    """Compare frozen routes including cold and fixed-amortization setup costs.

    Candidate observations must include complete solver/verification call walls.
    Timed shadow decisions are frozen before outcomes; their overhead is charged
    even when the guard or support check sends the solve to production.  This is
    development evaluation, not a substitute for measured routed public calls.
    """
    session_setup_s = _real(session_setup_s, "session setup")
    _integer(reuse_count, "fixed reuse count", 1)
    _integer(min_families, "minimum evaluation families", 2)
    _integer(bootstrap_draws, "bootstrap draws", 100, 100_000)
    _integer(seed, "bootstrap seed", 0)
    baselines = _names(baselines, "baselines")
    if "production" not in baselines:
        raise ValueError("production baseline required")
    compared = tuple(dict.fromkeys((model.baseline, *baselines)))
    candidates = tuple(dict.fromkeys((*model.candidates, *baselines)))
    rows = sorted(_validate_rows(rows, candidates, "test"), key=lambda r: r.instance)
    _reject_overlap((set(model.training_instances), set(model.training_groups), set(model.training_hashes)),
                    _identity_sets(rows))
    if rows[0].environment_id != model.environment_id:
        raise ValueError("test environment mismatch")
    if guard is not None:
        if (guard.model_sha256 != model.payload()["sha256"]
                or guard.environment_id != model.environment_id or guard.cutoff_s != model.cutoff_s
                or guard.protocol_sha256 != model.protocol_sha256
                or guard.implementation_sha256 != implementation_id()):
            raise ValueError("guard binding mismatch")
        _reject_overlap((set(guard.calibration_instances), set(guard.calibration_groups),
                         set(guard.calibration_hashes)), _identity_sets(rows))
    setup = dict(baseline_setup_s or {})
    if not set(setup) <= set(compared):
        raise ValueError("unknown baseline setup")
    setup = {name: _real(setup.get(name, 0.), "baseline setup") for name in compared}
    if set(decisions) != {r.instance for r in rows}:
        raise ValueError("incomplete decision coverage")
    selected, overheads = [], []
    for row in rows:
        decision = decisions[row.instance]
        overhead = _real(decision.overhead_s, "decision overhead")
        expected, reason, leaf = choose_lp_v2(row.features, model, model.environment_id, model.candidates)
        if (decision.candidate != expected or decision.leaf_id != leaf
                or (decision.leaf_id is not None and type(decision.leaf_id) is not int)):
            raise ValueError("decision differs from frozen model/leaf")
        routed = expected
        if leaf is None or (guard is not None and (not guard.permits_bound_candidate(expected, leaf)
                                                 or overhead > guard.overhead_limit_s)):
            routed = "production"
        selected.append(routed)
        overheads.append(overhead)
    families = sorted({r.group for r in rows})
    indices = [np.asarray([i for i, r in enumerate(rows) if r.group == name]) for name in families]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(families), size=(bootstrap_draws, len(families)))
    scopes = {}
    for scope, divisor in (("cold", 1), ("amortized", reuse_count)):
        policy_cost, policy_success = zip(*[_score(r.samples[c], model.cutoff_s, o + session_setup_s / divisor)
                                            for r, c, o in zip(rows, selected, overheads)])
        policy_cost = np.asarray(policy_cost)
        family_policy = np.asarray([policy_cost[i].mean() for i in indices])
        comparisons = {}
        for baseline in compared:
            costs, successes = zip(*[_score(r.samples[baseline], model.cutoff_s, setup[baseline] / divisor)
                                     for r in rows])
            costs = np.asarray(costs)
            family_base = np.asarray([costs[i].mean() for i in indices])
            boot = _ratios(family_policy[draws].sum(axis=1), family_base[draws].sum(axis=1))
            ratio = float(_ratios(family_policy.sum(), family_base.sum()))
            upper = float(np.quantile(boot, .975))
            tail = float(np.quantile(_ratios(policy_cost, costs), .9))
            gates = {"three_percent_improvement": ratio <= .97,
                     "bootstrap_upper_below_one": upper < 1,
                     "p90_slowdown_at_most_25_percent": tail <= 1.25,
                     "no_lost_verified_solves": all(a >= b for a, b in zip(policy_success, successes))}
            comparisons[baseline] = {"family_weighted_ratio": ratio,
                "baseline_family_mean_par10_s": float(family_base.mean()),
                "bootstrap_95pct": [float(np.quantile(boot, .025)), upper],
                "p90_instance_ratio": tail, "baseline_verified_repeats": sum(successes), "gates": gates}
        enough = len(families) >= min_families
        scopes[scope] = {"session_setup_charge_s": session_setup_s / divisor,
            "policy_family_mean_par10_s": float(family_policy.mean()),
            "verified_repeats": sum(policy_success), "comparisons": comparisons,
            "enough_families": enough,
            "research_gate_passed": enough and all(all(c["gates"].values()) for c in comparisons.values())}
    return {"schema": "solverpilot.experimental.lp-v2-development-report.v1",
        "model_sha256": model.payload()["sha256"], "protocol_sha256": model.protocol_sha256,
        "guard_sha256": guard.payload()["sha256"] if guard is not None else None,
        "instances": len(rows), "families": len(families), "fixed_reuse_count": reuse_count,
        "bootstrap_draws": bootstrap_draws, "bootstrap_seed": seed, "scopes": scopes,
        "routed_candidates": dict(Counter(selected)),
        "production_authorized": False, "automatic_production_routing_enabled": False,
        "cost_definition": "per-repeat solver/verification wall + routing + setup share; failed or late = 10*cutoff; equal family weights"}
