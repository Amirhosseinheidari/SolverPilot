from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .errors import VRPValidationError


_INDEPENDENT_PROOF_TOKEN = object()


@dataclass(frozen=True, slots=True)
class VRPRoute:
    vehicle_id: str
    customer_ids: tuple[str, ...]

    def __init__(self, vehicle_id: str, customer_ids: Sequence[str] = ()) -> None:
        vehicle = str(vehicle_id).strip()
        if not vehicle:
            raise VRPValidationError("route vehicle_id must be non-empty")
        ids = tuple(str(x).strip() for x in customer_ids)
        if any(not x for x in ids):
            raise VRPValidationError("route customer ids must be non-empty strings")
        object.__setattr__(self, "vehicle_id", vehicle)
        object.__setattr__(self, "customer_ids", ids)


@dataclass(frozen=True, slots=True)
class VRPSolution:
    routes: tuple[VRPRoute, ...]
    objective: float
    method: str
    is_exact: bool
    optimality_proven: bool
    unassigned_customer_ids: tuple[str, ...] = ()
    runtime_seconds: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __init__(
        self,
        routes: Sequence[VRPRoute],
        objective: float,
        *,
        method: str,
        is_exact: bool,
        optimality_proven: bool,
        unassigned_customer_ids: Sequence[str] = (),
        runtime_seconds: float | None = None,
        metadata: Mapping[str, Any] | None = None,
        _proof_token: object | None = None,
    ) -> None:
        route_tuple = tuple(routes)
        if any(not isinstance(r, VRPRoute) for r in route_tuple):
            raise VRPValidationError("routes must contain VRPRoute objects")
        if isinstance(objective, bool):
            raise VRPValidationError("VRP objective must be numeric, not bool")
        obj = float(objective)
        if not isfinite(obj) or obj < 0.0:
            raise VRPValidationError("VRP objective must be finite and non-negative")
        if type(is_exact) is not bool or type(optimality_proven) is not bool:
            raise VRPValidationError("is_exact and optimality_proven must be bool")
        if optimality_proven and not is_exact:
            raise VRPValidationError("optimality_proven requires is_exact=True")
        if optimality_proven and _proof_token is not _INDEPENDENT_PROOF_TOKEN:
            raise VRPValidationError(
                "independent optimality proof may only be issued by SolverPilot verified/reference paths"
            )
        method_value = str(method).strip()
        if not method_value:
            raise VRPValidationError("VRP solution method must be non-empty")
        unassigned = tuple(str(x).strip() for x in unassigned_customer_ids)
        if any(not x for x in unassigned) or len(set(unassigned)) != len(unassigned):
            raise VRPValidationError("unassigned customer ids must be unique non-empty strings")
        if runtime_seconds is not None:
            if isinstance(runtime_seconds, bool):
                raise VRPValidationError("runtime_seconds must be numeric, not bool")
            runtime_seconds = float(runtime_seconds)
            if not isfinite(runtime_seconds) or runtime_seconds < 0.0:
                raise VRPValidationError("runtime_seconds must be finite and non-negative")
        object.__setattr__(self, "routes", route_tuple)
        object.__setattr__(self, "objective", obj)
        object.__setattr__(self, "method", method_value)
        object.__setattr__(self, "is_exact", is_exact)
        object.__setattr__(self, "optimality_proven", optimality_proven)
        object.__setattr__(self, "unassigned_customer_ids", unassigned)
        object.__setattr__(self, "runtime_seconds", runtime_seconds)
        object.__setattr__(self, "metadata", MappingProxyType(dict(metadata or {})))
