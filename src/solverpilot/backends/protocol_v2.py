from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from solverpilot.capabilities import BackendCapabilityManifestV2


class CallbackEventKind(str, Enum):
    PROGRESS = "progress"
    INCUMBENT = "incumbent"
    NODE = "node"
    MESSAGE = "message"


@dataclass(frozen=True, slots=True)
class CallbackEvent:
    kind: CallbackEventKind
    runtime_s: float | None = None
    objective: float | None = None
    bound: float | None = None
    gap: float | None = None
    node_count: int | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BackendArtifactBundle:
    """Raw backend artifacts before original-space reconstruction/validation.

    P3 intentionally keeps artifacts optional and independent. A backend may return a
    primal vector without duals, or a basis without rays. Absence is represented by
    ``None`` rather than inferred from solver termination.
    """

    primal: np.ndarray | None = None
    objective: float | None = None
    dual: np.ndarray | None = None
    slacks: np.ndarray | None = None
    reduced_costs: np.ndarray | None = None
    primal_ray: np.ndarray | None = None
    dual_ray: np.ndarray | None = None
    basis: Any | None = None
    iis: Any | None = None
    solution_pool: tuple[np.ndarray, ...] = ()
    infeasibility_certificate: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BackendSolveRequestV2:
    time_limit_s: float | None = None
    thread_limit: int | None = None
    primal_start: np.ndarray | None = None
    dual_start: np.ndarray | None = None
    basis_start: Any | None = None
    mip_start: np.ndarray | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BackendSolveResultV2:
    termination: str
    artifacts: BackendArtifactBundle
    statistics: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class BackendCallbackControllerV2(Protocol):
    def terminate(self) -> None: ...
    def add_lazy_constraint(self, constraint: Any) -> None: ...
    def add_user_cut(self, constraint: Any) -> None: ...
    def submit_heuristic_solution(self, values: np.ndarray) -> None: ...


@runtime_checkable
class BackendCallbackV2(Protocol):
    def __call__(self, event: CallbackEvent, controller: BackendCallbackControllerV2) -> None: ...


@runtime_checkable
class BackendProtocolV2(Protocol):
    """Future adapter protocol consumed by P4/P5.

    Current M-track adapters do not automatically satisfy this protocol. P3 freezes the
    surface so later milestones can implement it capability-by-capability without
    retrofitting callback/mutation semantics after the Bridge Engine is built.
    """

    @property
    def capability_manifest_v2(self) -> BackendCapabilityManifestV2: ...

    def is_available(self) -> bool: ...

    def solve_v2(
        self,
        problem: Any,
        request: BackendSolveRequestV2,
        callback: BackendCallbackV2 | None = None,
    ) -> BackendSolveResultV2: ...


@runtime_checkable
class IncrementalBackendProtocolV2(Protocol):
    def update_variable_bounds(self, lower: np.ndarray, upper: np.ndarray) -> None: ...
    def update_rhs(self, lower: np.ndarray, upper: np.ndarray) -> None: ...
    def update_objective(self, linear: np.ndarray, quadratic: Any | None = None) -> None: ...
    def update_matrix_values(self, values: np.ndarray, indices: np.ndarray | None = None) -> None: ...


@runtime_checkable
class StructuralIncrementalBackendProtocolV2(Protocol):
    def add_variable(self, specification: Any) -> Any: ...
    def delete_variable(self, handle: Any) -> None: ...
    def add_constraint(self, specification: Any) -> Any: ...
    def delete_constraint(self, handle: Any) -> None: ...
