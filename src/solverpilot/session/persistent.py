from __future__ import annotations

from threading import RLock
from solverpilot._synchronization import serialized

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from time import perf_counter
from typing import Any, Callable
import copy

from solverpilot.backends import Backend
from solverpilot.capabilities import (
    BackendCapabilityManifestV2,
    CapabilityKey,
    resolve_backend_capabilities,
)
from solverpilot.model import CompiledModel, Model, Parameter
from solverpilot.problem import LinearProblem, QuadraticProblem
from solverpilot.runtime import SolveResult
from solverpilot.runtime.executor import execute

from .mutations import MutationKind, MutationRecord, classify_mutations


class PersistentSessionError(RuntimeError):
    pass


class ReuseEvidenceMismatchError(PersistentSessionError):
    pass


class MutationExecutionPath(str, Enum):
    COLD_BUILD = "cold_build"
    NATIVE_PATCH = "native_patch"
    SAFE_REBUILD = "safe_rebuild"
    FULL_REBUILD = "full_rebuild"
    NO_MUTATION_REUSE = "no_mutation_reuse"


@dataclass(frozen=True, slots=True)
class SessionPolicy:
    require_verified_native_patch: bool = True
    require_backend_reuse_evidence: bool = True
    fresh_backend_on_rebuild: bool = True
    original_space_validation: bool = True


@dataclass(frozen=True, slots=True)
class MutationDecision:
    path: MutationExecutionPath
    reason: str
    mutation_kinds: tuple[MutationKind, ...] = ()
    required_capabilities: tuple[CapabilityKey, ...] = ()
    missing_capabilities: tuple[CapabilityKey, ...] = ()
    semantic_changed: bool = False
    execution_structure_changed: bool = False
    compilation_plan_changed: bool = False
    certificate_changed: bool = False


@dataclass(frozen=True, slots=True)
class PersistentSessionTrace:
    revision: int
    path: MutationExecutionPath
    decision: MutationDecision
    compiler_cache_status: str
    compile_s: float
    dispatch_s: float
    total_s: float
    backend: str
    backend_reuse_applied: bool | None
    backend_reuse_mode: str | None
    backend_reset: bool
    original_validation_valid: bool | None
    semantic_hash: str
    data_hash: str
    compilation_hash: str
    transformation_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "path": self.path.value,
            "decision": {
                "path": self.decision.path.value,
                "reason": self.decision.reason,
                "mutation_kinds": [k.value for k in self.decision.mutation_kinds],
                "required_capabilities": [k.value for k in self.decision.required_capabilities],
                "missing_capabilities": [k.value for k in self.decision.missing_capabilities],
                "semantic_changed": self.decision.semantic_changed,
                "execution_structure_changed": self.decision.execution_structure_changed,
                "compilation_plan_changed": self.decision.compilation_plan_changed,
                "certificate_changed": self.decision.certificate_changed,
            },
            "compiler_cache_status": self.compiler_cache_status,
            "compile_s": self.compile_s,
            "dispatch_s": self.dispatch_s,
            "total_s": self.total_s,
            "backend": self.backend,
            "backend_reuse_applied": self.backend_reuse_applied,
            "backend_reuse_mode": self.backend_reuse_mode,
            "backend_reset": self.backend_reset,
            "original_validation_valid": self.original_validation_valid,
            "semantic_hash": self.semantic_hash,
            "data_hash": self.data_hash,
            "compilation_hash": self.compilation_hash,
            "transformation_count": self.transformation_count,
        }


@dataclass(frozen=True, slots=True)
class PersistentSolveOutcome:
    result: SolveResult
    compiled: CompiledModel
    trace: PersistentSessionTrace
    original_validation: Any | None = None

    @property
    def status(self):
        return self.result.status

    @property
    def x(self):
        return self.result.x

    @property
    def objective(self):
        return self.result.objective


BackendFactory = Callable[[], Backend]


@dataclass(frozen=True, slots=True)
class PersistentBackendBinding:
    backend: Backend
    capabilities: BackendCapabilityManifestV2
    persistence_verified: bool
    conformance: Any | None = None


def bind_persistent_backend(backend: Backend, *, verify: bool = True) -> PersistentBackendBinding:
    if verify:
        from .conformance_v2 import conform_persistent_backend
        report = conform_persistent_backend(backend)
        return PersistentBackendBinding(backend, report.manifest, report.passed, report)
    manifest = resolve_backend_capabilities(backend, verify=True)
    return PersistentBackendBinding(backend, manifest, False, None)


def _factory_from_dataclass_backend(backend: Backend) -> BackendFactory | None:
    if not is_dataclass(backend):
        return None
    values: dict[str, Any] = {}
    try:
        for item in fields(backend):
            if not item.init or item.name.startswith("_"):
                continue
            values[item.name] = copy.deepcopy(getattr(backend, item.name))
    except Exception:
        return None

    def factory():
        return type(backend)(**copy.deepcopy(values))

    try:
        probe = factory()
    except Exception:
        return None
    close = getattr(probe, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass
    return factory


def _mutation_capability(kind: MutationKind) -> CapabilityKey | None:
    return {
        MutationKind.OBJECTIVE_VECTOR: CapabilityKey.INCREMENTAL_OBJECTIVE,
        MutationKind.OBJECTIVE_OFFSET: CapabilityKey.INCREMENTAL_OBJECTIVE,
        MutationKind.OBJECTIVE_SENSE: CapabilityKey.INCREMENTAL_OBJECTIVE,
        MutationKind.OBJECTIVE_QUADRATIC_VALUES: CapabilityKey.INCREMENTAL_OBJECTIVE,
        MutationKind.VARIABLE_BOUNDS: CapabilityKey.INCREMENTAL_VARIABLE_BOUNDS,
        MutationKind.CONSTRAINT_BOUNDS: CapabilityKey.INCREMENTAL_RHS,
        MutationKind.MATRIX_VALUES_SAME_SPARSITY: CapabilityKey.INCREMENTAL_MATRIX_VALUES,
    }.get(kind)


_STRUCTURAL_KINDS = {
    MutationKind.MATRIX_SPARSITY_CHANGED,
    MutationKind.QUADRATIC_SPARSITY_CHANGED,
    MutationKind.VARIABLES_ADDED_REMOVED,
    MutationKind.CONSTRAINTS_ADDED_REMOVED,
    MutationKind.INTEGRALITY_CHANGED,
}


def decide_mutation_path(
    previous: CompiledModel | None,
    current: CompiledModel,
    capabilities: BackendCapabilityManifestV2,
    *,
    require_verified: bool = True,
) -> tuple[MutationDecision, MutationRecord | None]:
    if previous is None:
        return MutationDecision(MutationExecutionPath.COLD_BUILD, "first solve has no bound backend state"), None

    semantic_changed = previous.semantic_hash != current.semantic_hash
    compilation_changed = previous.compilation_hash != current.compilation_hash
    certificate_changed = previous.precondition_certificate_hashes != current.precondition_certificate_hashes

    old = previous.execution_ir
    new = current.execution_ir
    if type(old) is not type(new):
        return MutationDecision(
            MutationExecutionPath.FULL_REBUILD,
            "execution IR class changed",
            semantic_changed=semantic_changed,
            execution_structure_changed=True,
            compilation_plan_changed=compilation_changed,
            certificate_changed=certificate_changed,
        ), None

    record = classify_mutations(old, new, revision_before=0, revision_after=1)
    kinds = record.kinds
    if record.is_noop:
        return MutationDecision(
            MutationExecutionPath.NO_MUTATION_REUSE,
            "execution data is unchanged; existing backend model may be solved again",
            mutation_kinds=(),
            semantic_changed=semantic_changed,
            execution_structure_changed=False,
            compilation_plan_changed=compilation_changed,
            certificate_changed=certificate_changed,
        ), record

    if semantic_changed or any(kind in _STRUCTURAL_KINDS for kind in kinds):
        return MutationDecision(
            MutationExecutionPath.FULL_REBUILD,
            "semantic or execution structure changed; P5 does not emulate structural incremental edits",
            mutation_kinds=kinds,
            semantic_changed=semantic_changed,
            execution_structure_changed=record.structural_change,
            compilation_plan_changed=compilation_changed,
            certificate_changed=certificate_changed,
        ), record

    required_list: list[CapabilityKey] = [CapabilityKey.LIFECYCLE_PERSISTENT]
    for kind in kinds:
        cap = _mutation_capability(kind)
        if cap is None:
            return MutationDecision(
                MutationExecutionPath.SAFE_REBUILD,
                f"mutation {kind.value} has no verified P5 native-patch mapping",
                mutation_kinds=kinds,
                semantic_changed=semantic_changed,
                execution_structure_changed=record.structural_change,
                compilation_plan_changed=compilation_changed,
                certificate_changed=certificate_changed,
            ), record
        if cap not in required_list:
            required_list.append(cap)
    required = tuple(required_list)
    missing = tuple(
        key for key in required
        if not capabilities.check(key, require_verified=require_verified).usable
    )
    if missing:
        return MutationDecision(
            MutationExecutionPath.SAFE_REBUILD,
            "same-structure update is correct but native patch lacks verified capability evidence",
            mutation_kinds=kinds,
            required_capabilities=required,
            missing_capabilities=missing,
            semantic_changed=semantic_changed,
            execution_structure_changed=record.structural_change,
            compilation_plan_changed=compilation_changed,
            certificate_changed=certificate_changed,
        ), record
    return MutationDecision(
        MutationExecutionPath.NATIVE_PATCH,
        "all same-structure mutation operations and persistent lifecycle are runtime-verified",
        mutation_kinds=kinds,
        required_capabilities=required,
        semantic_changed=semantic_changed,
        execution_structure_changed=record.structural_change,
        compilation_plan_changed=compilation_changed,
        certificate_changed=certificate_changed,
    ), record


class PersistentSession:
    """Bind a semantic model to one backend lifecycle with fail-closed mutation routing.

    P5 owns the *execution decision* only. P2 remains responsible for semantic/parameter
    recompilation, P4 remains responsible for reformulation certificates, and the backend
    remains responsible for the native numerical update implementation.
    """

    def __init__(
        self,
        model: Model,
        backend: Backend,
        *,
        capabilities: BackendCapabilityManifestV2 | None = None,
        backend_factory: BackendFactory | None = None,
        bridge_policy=None,
        policy: SessionPolicy | None = None,
    ) -> None:
        if not isinstance(model, Model):
            raise TypeError("PersistentSession requires an solverpilot.model.Model")
        self._lock = RLock()
        self.model = model
        self.backend = backend
        self.capabilities = capabilities or resolve_backend_capabilities(backend, verify=True)
        if self.capabilities.backend != backend.manifest.name:
            raise PersistentSessionError("capability manifest backend identity does not match bound backend")
        if (
            self.capabilities.backend_version is not None
            and backend.manifest.version is not None
            and self.capabilities.backend_version != backend.manifest.version
        ):
            raise PersistentSessionError("capability manifest backend version does not match bound backend")
        self.backend_factory = backend_factory or _factory_from_dataclass_backend(backend)
        self.bridge_policy = bridge_policy
        self.policy = policy or SessionPolicy()
        self._compiled: CompiledModel | None = None
        self._last_outcome: PersistentSolveOutcome | None = None
        self._revision = 0
        self._history: list[PersistentSessionTrace] = []
        self._closed = False

    @classmethod
    def from_binding(cls, model: Model, binding: PersistentBackendBinding, **kwargs):
        return cls(model, binding.backend, capabilities=binding.capabilities, **kwargs)

    @classmethod
    def bind(cls, model: Model, backend: Backend, *, verify: bool = True, **kwargs):
        return cls.from_binding(model, bind_persistent_backend(backend, verify=verify), **kwargs)

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def history(self) -> tuple[PersistentSessionTrace, ...]:
        return tuple(self._history)

    @property
    def last_outcome(self) -> PersistentSolveOutcome | None:
        return self._last_outcome

    @property
    def compiled(self) -> CompiledModel | None:
        return self._compiled

    @serialized
    def set_parameter(self, parameter: Parameter, value: Any) -> None:
        if not isinstance(parameter, Parameter) or parameter._model is not self.model:
            raise TypeError("parameter must belong to this PersistentSession model")
        parameter.value = value

    @serialized
    def set_parameter_by_name(self, name: str, value: Any) -> None:
        matches = [p for p in self.model.parameters if p.name == name]
        if len(matches) != 1:
            raise KeyError(f"parameter name must resolve uniquely: {name!r}")
        matches[0].value = value

    def _replace_backend_fresh(self) -> bool:
        old = self.backend
        if self.backend_factory is not None:
            close = getattr(old, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            self.backend = self.backend_factory()
            return True
        close = getattr(old, "close", None)
        if callable(close):
            close()
            return True
        if self.policy.fresh_backend_on_rebuild:
            raise PersistentSessionError(
                "safe rebuild requires a backend factory or a backend exposing close(); "
                "provide backend_factory explicitly"
            )
        return False

    @serialized
    def solve(self) -> PersistentSolveOutcome:
        if self._closed:
            raise PersistentSessionError("PersistentSession is closed")
        total_t0 = perf_counter()
        compile_t0 = perf_counter()
        compiled = self.model.compile(
            use_cache=True,
            bridge_policy=self.bridge_policy,
            capabilities=self.capabilities,
        )
        compile_s = perf_counter() - compile_t0
        decision, _ = decide_mutation_path(
            self._compiled,
            compiled,
            self.capabilities,
            require_verified=self.policy.require_verified_native_patch,
        )

        backend_reset = False
        if decision.path in {MutationExecutionPath.SAFE_REBUILD, MutationExecutionPath.FULL_REBUILD}:
            backend_reset = self._replace_backend_fresh()
            # A fresh backend object may have identical runtime identity; capabilities remain
            # valid only for the exact version signature already bound to this session.
            fresh_caps = resolve_backend_capabilities(self.backend, verify=False)
            if (
                fresh_caps.backend != self.capabilities.backend
                or fresh_caps.backend_version != self.capabilities.backend_version
            ):
                raise PersistentSessionError("fresh backend runtime identity drifted from bound capability evidence")

        dispatch_t0 = perf_counter()
        result = execute(compiled.execution_ir, self.backend)
        dispatch_s = perf_counter() - dispatch_t0
        raw = result.raw_statistics or {}
        reuse_raw = raw.get("reuse_applied")
        reuse_applied = reuse_raw if isinstance(reuse_raw, bool) else None
        reuse_mode = None if raw.get("reuse_mode") is None else str(raw.get("reuse_mode"))

        if decision.path is MutationExecutionPath.NATIVE_PATCH and self.policy.require_backend_reuse_evidence:
            if reuse_applied is not True:
                raise ReuseEvidenceMismatchError(
                    "P5 selected native_patch from verified capabilities, but backend did not report applied reuse"
                )
        if decision.path in {MutationExecutionPath.COLD_BUILD, MutationExecutionPath.SAFE_REBUILD, MutationExecutionPath.FULL_REBUILD}:
            if backend_reset or decision.path is MutationExecutionPath.COLD_BUILD:
                if reuse_applied is True:
                    raise ReuseEvidenceMismatchError(
                        "fresh/cold backend unexpectedly reported reuse; lifecycle evidence is inconsistent"
                    )

        original_validation = None
        original_valid = None
        if self.policy.original_space_validation and result.x is not None:
            original_validation = compiled.validate_original(self.model, result.x)
            original_valid = bool(original_validation.valid)
            if not original_valid:
                raise PersistentSessionError("backend candidate failed P5 original-space semantic validation")

        previous_data = None if self._compiled is None else self._compiled.data_hash
        if previous_data is not None and previous_data != compiled.data_hash:
            self._revision += 1
        elif self._compiled is not None and self._compiled.semantic_hash != compiled.semantic_hash:
            self._revision += 1

        trace = PersistentSessionTrace(
            revision=self._revision,
            path=decision.path,
            decision=decision,
            compiler_cache_status=compiled.compilation_report.cache_status,
            compile_s=float(compile_s),
            dispatch_s=float(dispatch_s),
            total_s=float(perf_counter() - total_t0),
            backend=self.backend.manifest.name,
            backend_reuse_applied=reuse_applied,
            backend_reuse_mode=reuse_mode,
            backend_reset=backend_reset,
            original_validation_valid=original_valid,
            semantic_hash=compiled.semantic_hash,
            data_hash=compiled.data_hash,
            compilation_hash=compiled.compilation_hash,
            transformation_count=len(compiled.transformation_tape),
        )
        outcome = PersistentSolveOutcome(result=result, compiled=compiled, trace=trace, original_validation=original_validation)
        self._compiled = compiled
        self._last_outcome = outcome
        self._history.append(trace)
        return outcome

    @serialized
    def close(self) -> None:
        if self._closed:
            return
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
