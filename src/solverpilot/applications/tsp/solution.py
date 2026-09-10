from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .errors import TSPValidationError


_INDEPENDENT_PROOF_TOKEN = object()


@dataclass(frozen=True, slots=True)
class TSPSolution:
    route: tuple[int, ...]
    objective: float
    method: str
    is_exact: bool
    optimality_proven: bool
    runtime_seconds: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __init__(
        self,
        route: Sequence[int],
        objective: float,
        *,
        method: str,
        is_exact: bool,
        optimality_proven: bool,
        runtime_seconds: float | None = None,
        metadata: Mapping[str, Any] | None = None,
        _proof_token: object | None = None,
    ) -> None:
        normalized: list[int] = []
        for value in route:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TSPValidationError("TSP route entries must be integer node indices")
            normalized.append(int(value))
        obj = float(objective)
        if not isfinite(obj):
            raise TSPValidationError("TSP objective must be finite")
        if type(is_exact) is not bool or type(optimality_proven) is not bool:
            raise TSPValidationError("is_exact and optimality_proven must be bool values")
        if optimality_proven and not is_exact:
            raise TSPValidationError("optimality_proven requires is_exact=True")
        if optimality_proven and _proof_token is not _INDEPENDENT_PROOF_TOKEN:
            raise TSPValidationError(
                "independent optimality proof may only be issued by SolverPilot verified/reference paths"
            )
        method_value = str(method).strip()
        if not method_value:
            raise TSPValidationError("TSP solution method must be non-empty")
        if runtime_seconds is not None:
            runtime_seconds = float(runtime_seconds)
            if not isfinite(runtime_seconds) or runtime_seconds < 0.0:
                raise TSPValidationError("runtime_seconds must be finite and non-negative")
        object.__setattr__(self, "route", tuple(normalized))
        object.__setattr__(self, "objective", obj)
        object.__setattr__(self, "method", method_value)
        object.__setattr__(self, "is_exact", is_exact)
        object.__setattr__(self, "optimality_proven", optimality_proven)
        object.__setattr__(self, "runtime_seconds", runtime_seconds)
        object.__setattr__(self, "metadata", MappingProxyType(dict(metadata or {})))
