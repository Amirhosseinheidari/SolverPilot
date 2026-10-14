"""Training-only gain guard for explicit experimental routing.

Calibration is not a promotion test. All outcomes include setup, solve and
independent checking; unsuccessful/late runs pay PAR10. Every calibration group
must benefit after charging routing overhead. Public held-out qualification is
still required before any production promotion.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .learned_lp import _choose, _digest, _validate


def implementation_paths():
    root = Path(__file__).resolve().parents[1]
    names = ('experimental/learned_lp.py', 'experimental/robust_lp.py',
             'experimental/lp_gain.py', 'experimental/lp_session.py',
             'experimental/learned_lp_v2.py',
             'backends/highspy_native.py', 'backends/scipy_highs_lp.py',
             'experimental/lp_environment.py',
             'runtime/auto.py', 'runtime/executor.py',
             'validate/optimality.py', 'validate/lp_dual.py',
             'validate/_certificate_arithmetic.py')
    return tuple((name, root/name) for name in names)


def implementation_id():
    return _digest({name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for name, path in implementation_paths()})


@dataclass(frozen=True)
class LPGainGuard:
    model_sha256: str
    implementation_sha256: str
    environment_id: str
    cutoff_s: float
    calibration_sha256: str
    overhead_limit_s: float
    # Minimum group mean gain for each candidate, after charging overhead.
    candidate_gains: tuple[tuple[str, float], ...]

    def __post_init__(self):
        object.__setattr__(self, 'candidate_gains', tuple(tuple(p) for p in self.candidate_gains))
        for value in (self.model_sha256, self.implementation_sha256, self.calibration_sha256):
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('invalid guard digest')
        if (not self.environment_id or not math.isfinite(self.cutoff_s) or self.cutoff_s <= 0
                or not math.isfinite(self.overhead_limit_s) or self.overhead_limit_s < 0
                or len(dict(self.candidate_gains)) != len(self.candidate_gains)
                or any(not name or not math.isfinite(gain) or gain <= 0 for name, gain in self.candidate_gains)):
            raise ValueError('invalid gain guard')

    def permits(self, model, candidate, *, elapsed_s, cutoff_s):
        return (self.model_sha256 == model.payload()['sha256']
                and self.implementation_sha256 == implementation_id()
                and self.environment_id == model.environment_id
                and self.cutoff_s == cutoff_s
                and math.isfinite(elapsed_s) and 0 <= elapsed_s <= self.overhead_limit_s
                and candidate in dict(self.candidate_gains))

    def payload(self):
        data = {'schema': 'solverpilot.experimental.lp-gain.v1', **asdict(self)}
        return {**data, 'sha256': _digest(data)}

    @classmethod
    def load(cls, path):
        raw = Path(path).read_bytes()
        if len(raw) > 2_000_000: raise ValueError('guard artifact too large')
        data = json.loads(raw)
        digest = data.pop('sha256')
        if digest != _digest(data) or data.pop('schema') != 'solverpilot.experimental.lp-gain.v1':
            raise ValueError('guard digest/schema mismatch')
        data['candidate_gains'] = tuple(tuple(pair) for pair in data['candidate_gains'])
        return cls(**data)


def calibrate_gain_guard(model, rows, *, overhead_limit_s=.005, min_groups=4):
    """Use training observations only; held-out outcomes can never tune the guard.

    Rows include every model candidate plus production. Gains are checked only
    on rows where the frozen model would select that candidate. Even one lost
    verified repeat disallows that candidate. The caller must measure full-call
    times; a schema cannot prove how external measurements were collected.
    """
    candidates = (*model.candidates, 'production')
    _validate(rows, candidates, model.cutoff_s)
    if (any(row.split != 'train' or row.environment_id != model.environment_id for row in rows)
            or not math.isfinite(overhead_limit_s) or overhead_limit_s < 0
            or type(min_groups) is not int or min_groups < 2):
        raise ValueError('training-only calibration with valid overhead/group limits required')
    gains = {}
    lost = set()
    for row in rows:
        candidate, reason = _choose(row.features, model, model.environment_id, set(model.candidates))
        if reason not in {'learned_stump', 'training_baseline'}: continue
        def score(name, overhead=0.):
            return [(wall+overhead if ok and wall+overhead <= model.cutoff_s else 10*model.cutoff_s,
                     bool(ok and wall+overhead <= model.cutoff_s)) for wall, ok in row.samples[name]]
        selected, production = score(candidate, overhead_limit_s), score('production')
        if sum(ok for _, ok in selected) < sum(ok for _, ok in production): lost.add(candidate)
        improvement = float(np.mean([v for v, _ in production])-np.mean([v for v, _ in selected]))
        gains.setdefault(candidate, {}).setdefault(row.group, []).append(improvement)
    allowed = []
    for name, groups in sorted(gains.items()):
        if name in lost or len(groups) < min_groups: continue
        floor = min(float(np.mean(values)) for values in groups.values())
        if floor > 0: allowed.append((name, floor))
    return LPGainGuard(model.payload()['sha256'], implementation_id(), model.environment_id,
                       model.cutoff_s, _digest([asdict(row) for row in rows]),
                       float(overhead_limit_s), tuple(allowed))
