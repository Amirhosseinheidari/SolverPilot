"""Explicit process-local binding for repeated experimental routing.

Content hashing occurs at creation. Warm calls check source metadata instead;
this detects ordinary edits/replacements, not adversarial metadata-preserving
rewrites. Recreate after code reload, dependency/configuration changes, or a new
measured environment. This is not a security boundary or production promotion.
"""
from dataclasses import dataclass, asdict, is_dataclass
import math
from time import perf_counter

from .lp_gain import implementation_id, implementation_paths
from .learned_lp import _digest


def _backend_snapshot(backends):
    result = []
    for name, backend in sorted(backends.items()):
        if not is_dataclass(backend):
            raise ValueError('bound routing requires dataclass backend settings')
        result.append((name, id(backend), type(backend).__module__, type(backend).__qualname__,
                       _digest(asdict(backend))))
    return tuple(result)


def _snapshot(paths):
    result = []
    for _, path in paths:
        s = path.stat()
        result.append((s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns))
    return tuple(result)


@dataclass(frozen=True, slots=True, init=False)
class LPRoutingSession:
    model: object
    guard: object
    environment_id: str
    cutoff_s: float
    preparation_s: float
    _paths: tuple
    _source_snapshot: tuple
    _backend_settings: tuple | None

    @classmethod
    def create(cls, model, guard, *, environment_id, cutoff_s, backends=None):
        start = perf_counter()
        if (isinstance(cutoff_s, bool) or not math.isfinite(cutoff_s) or cutoff_s <= 0
                or environment_id != model.environment_id or environment_id != guard.environment_id
                or cutoff_s != model.cutoff_s or cutoff_s != guard.cutoff_s):
            raise ValueError('session environment/cutoff binding mismatch')
        paths = implementation_paths()
        before = _snapshot(paths)
        if (guard.model_sha256 != model.payload()['sha256']
                or guard.implementation_sha256 != implementation_id()
                or getattr(guard, 'protocol_sha256', model.protocol_sha256) != model.protocol_sha256):
            raise ValueError('session model/implementation binding mismatch')
        if before != _snapshot(paths):
            raise ValueError('implementation changed during session preparation')
        settings = None if backends is None else _backend_snapshot(backends)
        obj = object.__new__(cls)
        for name, value in dict(model=model, guard=guard, environment_id=environment_id,
                cutoff_s=cutoff_s, preparation_s=perf_counter()-start, _paths=paths,
                _source_snapshot=before, _backend_settings=settings).items():
            object.__setattr__(obj, name, value)
        return obj

    def backends_match(self, backends):
        try:
            return self._backend_settings is not None and self._backend_settings == _backend_snapshot(backends)
        except (ValueError, TypeError):
            return False

    def permits(self, model, guard, candidate, *, environment_id, cutoff_s, elapsed_s, leaf_id=None):
        if (model is not self.model or guard is not self.guard
                or environment_id != self.environment_id or isinstance(cutoff_s, bool)
                or cutoff_s != self.cutoff_s or not math.isfinite(elapsed_s)
                or elapsed_s < 0 or elapsed_s > guard.overhead_limit_s
                or candidate not in dict(guard.candidate_gains)):
            return False
        if hasattr(guard, 'permits_bound_candidate') and not guard.permits_bound_candidate(candidate, leaf_id):
            return False
        try:
            return self._source_snapshot == _snapshot(self._paths)
        except OSError:
            return False
