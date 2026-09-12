from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Iterable, Mapping
from types import MappingProxyType

from .enums import Capability, SupportLevel
from .manifest import BackendManifest


CAPABILITY_SCHEMA_VERSION = "2.0"


class CapabilityStatus(str, Enum):
    UNSUPPORTED = "unsupported"
    SUPPORTED = "supported"
    RESTRICTED = "restricted"
    UNVERIFIED = "unverified"


class CapabilityMode(str, Enum):
    NATIVE = "native"
    EMULATED_SAFE = "emulated_safe"
    EMULATED_RISKY = "emulated_risky"
    NONE = "none"
    UNKNOWN = "unknown"


class VerificationLevel(str, Enum):
    VERIFIED = "verified"
    PROJECTED_LEGACY = "projected_legacy"
    UNVERIFIED = "unverified"


class CapabilityKey(str, Enum):
    # Problem classes
    PROBLEM_LP = "problem.lp"
    PROBLEM_MILP = "problem.milp"
    PROBLEM_CONVEX_QP = "problem.convex_qp"
    PROBLEM_NONCONVEX_QP = "problem.nonconvex_qp"
    PROBLEM_MIQP = "problem.miqp"
    PROBLEM_CONIC = "problem.conic"
    PROBLEM_CONIC_QUADRATIC = "problem.conic_quadratic"
    PROBLEM_NLP = "problem.nlp"
    PROBLEM_MINLP = "problem.minlp"
    PROBLEM_CP = "problem.cp"

    # Native semantic/constraint families that P4 may query.
    CONSTRAINT_LINEAR = "constraint.linear"
    CONSTRAINT_INDICATOR = "constraint.indicator"
    CONSTRAINT_SOS1 = "constraint.sos1"
    CONSTRAINT_SOS2 = "constraint.sos2"
    CONSTRAINT_EXPONENTIAL = "constraint.exponential"
    CONSTRAINT_POWER = "constraint.power"
    CONSTRAINT_GENERALIZED_POWER = "constraint.generalized_power"
    CONSTRAINT_SOC = "constraint.soc"
    CONSTRAINT_ROTATED_SOC = "constraint.rotated_soc"
    CONSTRAINT_PSD = "constraint.psd"
    CONSTRAINT_NONLINEAR = "constraint.nonlinear"
    CONSTRAINT_ALL_DIFFERENT = "constraint.all_different"
    CONSTRAINT_EXACTLY_ONE = "constraint.exactly_one"
    CONSTRAINT_TABLE = "constraint.table"
    CONSTRAINT_ELEMENT = "constraint.element"
    CONSTRAINT_CIRCUIT = "constraint.circuit"
    CONSTRAINT_NO_OVERLAP = "constraint.no_overlap"
    CONSTRAINT_CUMULATIVE = "constraint.cumulative"

    # Derivative surface.
    DERIVATIVE_GRADIENT = "derivative.gradient"
    DERIVATIVE_JACOBIAN = "derivative.jacobian"
    DERIVATIVE_HESSIAN = "derivative.hessian"

    # Incremental modification surface.
    INCREMENTAL_VARIABLE_BOUNDS = "incremental.variable_bounds"
    INCREMENTAL_RHS = "incremental.rhs"
    INCREMENTAL_OBJECTIVE = "incremental.objective"
    INCREMENTAL_MATRIX_VALUES = "incremental.matrix_values"
    INCREMENTAL_MATRIX_STRUCTURE = "incremental.matrix_structure"
    INCREMENTAL_ADD_VARIABLE = "incremental.add_variable"
    INCREMENTAL_DELETE_VARIABLE = "incremental.delete_variable"
    INCREMENTAL_ADD_CONSTRAINT = "incremental.add_constraint"
    INCREMENTAL_DELETE_CONSTRAINT = "incremental.delete_constraint"

    # Starts / reuse artifacts.
    START_PRIMAL = "start.primal"
    START_DUAL = "start.dual"
    START_BASIS = "start.basis"
    START_MIP = "start.mip"

    # Callback surface.
    CALLBACK_PROGRESS = "callback.progress"
    CALLBACK_INCUMBENT = "callback.incumbent"
    CALLBACK_NODE = "callback.node"
    CALLBACK_LAZY_CONSTRAINT = "callback.lazy_constraint"
    CALLBACK_USER_CUT = "callback.user_cut"
    CALLBACK_HEURISTIC = "callback.heuristic"

    # Result artifacts.
    RESULT_PRIMAL = "result.primal"
    RESULT_OBJECTIVE = "result.objective"
    RESULT_DUAL = "result.dual"
    RESULT_SLACK = "result.slack"
    RESULT_REDUCED_COST = "result.reduced_cost"
    RESULT_PRIMAL_RAY = "result.primal_ray"
    RESULT_DUAL_RAY = "result.dual_ray"
    RESULT_BASIS = "result.basis"
    RESULT_IIS = "result.iis"
    RESULT_SOLUTION_POOL = "result.solution_pool"
    RESULT_INFEASIBILITY_CERTIFICATE = "result.infeasibility_certificate"

    # Lifecycle / execution surface.
    LIFECYCLE_PERSISTENT = "lifecycle.persistent"
    LIFECYCLE_INTERRUPT = "lifecycle.interrupt"
    LIFECYCLE_CLONE = "lifecycle.clone"
    LIFECYCLE_THREAD_SAFE = "lifecycle.thread_safe"
    LIFECYCLE_THREAD_CONTROL = "lifecycle.thread_control"


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze_value(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(v) for v in value)
    return value


@dataclass(frozen=True, slots=True)
class CapabilityEvidence:
    evidence_id: str
    kind: str
    backend_versions: tuple[str, ...] = ()
    binding_versions: tuple[str, ...] = ()
    adapter_versions: tuple[str, ...] = ()
    verified_on: str | None = None
    notes: tuple[str, ...] = ()

    def matches_runtime(
        self,
        *,
        backend_version: str | None,
        binding_version: str | None,
        adapter_version: str | None,
    ) -> bool:
        def exact_or_unconstrained(allowed: tuple[str, ...], actual: str | None) -> bool:
            if not allowed:
                return True
            return actual is not None and actual in allowed

        return (
            exact_or_unconstrained(self.backend_versions, backend_version)
            and exact_or_unconstrained(self.binding_versions, binding_version)
            and exact_or_unconstrained(self.adapter_versions, adapter_version)
        )


@dataclass(frozen=True, slots=True)
class CapabilityClaim:
    status: CapabilityStatus
    mode: CapabilityMode = CapabilityMode.UNKNOWN
    verification: VerificationLevel = VerificationLevel.UNVERIFIED
    restrictions: Mapping[str, Any] = field(default_factory=dict)
    evidence: tuple[CapabilityEvidence, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "restrictions", _freeze_value(self.restrictions))
        if self.status is CapabilityStatus.UNSUPPORTED and self.mode not in {
            CapabilityMode.NONE,
            CapabilityMode.UNKNOWN,
        }:
            raise ValueError("unsupported capability cannot declare an active implementation mode")
        if self.status is CapabilityStatus.UNVERIFIED and self.verification is VerificationLevel.VERIFIED:
            raise ValueError("unverified capability cannot have verified evidence state")
        if self.status is CapabilityStatus.RESTRICTED and not self.restrictions:
            raise ValueError("restricted capability requires machine-readable restrictions")

    def runtime_verified(
        self,
        *,
        backend_version: str | None,
        binding_version: str | None,
        adapter_version: str | None,
    ) -> bool:
        if self.verification is not VerificationLevel.VERIFIED:
            return False
        if not self.evidence:
            return False
        return any(
            item.matches_runtime(
                backend_version=backend_version,
                binding_version=binding_version,
                adapter_version=adapter_version,
            )
            for item in self.evidence
        )

    def usable(
        self,
        *,
        backend_version: str | None,
        binding_version: str | None,
        adapter_version: str | None,
        allow_restricted: bool = True,
        allow_safe_emulation: bool = False,
        allow_risky_emulation: bool = False,
        require_verified: bool = True,
    ) -> bool:
        if self.status is CapabilityStatus.UNSUPPORTED or self.status is CapabilityStatus.UNVERIFIED:
            return False
        if self.status is CapabilityStatus.RESTRICTED and not allow_restricted:
            return False
        if self.mode is CapabilityMode.EMULATED_SAFE and not allow_safe_emulation:
            return False
        if self.mode is CapabilityMode.EMULATED_RISKY and not allow_risky_emulation:
            return False
        if self.mode in {CapabilityMode.NONE, CapabilityMode.UNKNOWN}:
            return False
        if require_verified and not self.runtime_verified(
            backend_version=backend_version,
            binding_version=binding_version,
            adapter_version=adapter_version,
        ):
            return False
        return True


@dataclass(frozen=True, slots=True)
class CapabilityCheck:
    key: CapabilityKey
    usable: bool
    status: CapabilityStatus
    mode: CapabilityMode
    verification: VerificationLevel
    reason: str
    restrictions: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BackendCapabilityManifestV2:
    backend: str
    backend_version: str | None
    binding: str
    binding_version: str | None
    adapter_version: str | None
    claims: Mapping[CapabilityKey, CapabilityClaim] = field(default_factory=dict)
    numeric_limits: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = CAPABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "claims", MappingProxyType(dict(self.claims)))
        object.__setattr__(self, "numeric_limits", _freeze_value(self.numeric_limits))
        object.__setattr__(self, "metadata", _freeze_value(self.metadata))
        if not self.backend:
            raise ValueError("backend must be non-empty")
        if not self.binding:
            raise ValueError("binding must be non-empty")
        if self.schema_version != CAPABILITY_SCHEMA_VERSION:
            raise ValueError(f"unsupported capability schema version {self.schema_version!r}")

    def claim(self, key: CapabilityKey) -> CapabilityClaim:
        return self.claims.get(
            key,
            CapabilityClaim(
                status=CapabilityStatus.UNVERIFIED,
                mode=CapabilityMode.UNKNOWN,
                verification=VerificationLevel.UNVERIFIED,
                notes=("capability absent from manifest",),
            ),
        )

    def check(
        self,
        key: CapabilityKey,
        *,
        allow_restricted: bool = True,
        allow_safe_emulation: bool = False,
        allow_risky_emulation: bool = False,
        require_verified: bool = True,
    ) -> CapabilityCheck:
        claim = self.claim(key)
        usable = claim.usable(
            backend_version=self.backend_version,
            binding_version=self.binding_version,
            adapter_version=self.adapter_version,
            allow_restricted=allow_restricted,
            allow_safe_emulation=allow_safe_emulation,
            allow_risky_emulation=allow_risky_emulation,
            require_verified=require_verified,
        )
        if usable:
            reason = "verified capability is usable under the requested policy"
        elif claim.status is CapabilityStatus.UNSUPPORTED:
            reason = "capability is explicitly unsupported"
        elif claim.status is CapabilityStatus.UNVERIFIED:
            reason = "capability is unverified"
        elif require_verified and not claim.runtime_verified(
            backend_version=self.backend_version,
            binding_version=self.binding_version,
            adapter_version=self.adapter_version,
        ):
            reason = "capability evidence does not verify this runtime version"
        elif claim.mode is CapabilityMode.EMULATED_SAFE and not allow_safe_emulation:
            reason = "safe emulation is disabled by policy"
        elif claim.mode is CapabilityMode.EMULATED_RISKY and not allow_risky_emulation:
            reason = "risky emulation is disabled by policy"
        elif claim.status is CapabilityStatus.RESTRICTED and not allow_restricted:
            reason = "restricted support is disabled by policy"
        else:
            reason = "capability is not usable under the requested policy"
        return CapabilityCheck(
            key=key,
            usable=usable,
            status=claim.status,
            mode=claim.mode,
            verification=claim.verification,
            reason=reason,
            restrictions=claim.restrictions,
        )

    def supports_verified(self, key: CapabilityKey, **policy: Any) -> bool:
        return self.check(key, require_verified=True, **policy).usable

    def require_verified(self, keys: Iterable[CapabilityKey], **policy: Any) -> tuple[CapabilityCheck, ...]:
        checks = tuple(self.check(key, require_verified=True, **policy) for key in keys)
        return checks

    @property
    def signature(self) -> str:
        raw = json.dumps(self.to_canonical_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return hashlib.sha256(raw).hexdigest()

    def to_canonical_dict(self) -> dict[str, Any]:
        claims: dict[str, Any] = {}
        for key in sorted(self.claims, key=lambda item: item.value):
            claim = self.claims[key]
            claims[key.value] = {
                "status": claim.status.value,
                "mode": claim.mode.value,
                "verification": claim.verification.value,
                "restrictions": _jsonable(claim.restrictions),
                "evidence": [
                    {
                        "evidence_id": item.evidence_id,
                        "kind": item.kind,
                        "backend_versions": list(item.backend_versions),
                        "binding_versions": list(item.binding_versions),
                        "adapter_versions": list(item.adapter_versions),
                        "verified_on": item.verified_on,
                        "notes": list(item.notes),
                    }
                    for item in claim.evidence
                ],
                "notes": list(claim.notes),
            }
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "binding": self.binding,
            "binding_version": self.binding_version,
            "adapter_version": self.adapter_version,
            "claims": claims,
            "numeric_limits": _jsonable(self.numeric_limits),
            "metadata": _jsonable(self.metadata),
        }

    def to_p0_schema_dict(self) -> dict[str, Any]:
        groups: dict[str, dict[str, str]] = {
            "problem_classes": {},
            "native_constraints": {},
            "incremental": {},
            "starts": {},
            "callbacks": {},
            "result_artifacts": {},
        }
        restrictions: dict[str, Any] = {}
        for key, claim in self.claims.items():
            group, leaf = _p0_group_leaf(key)
            if group is None:
                continue
            p0_status = (
                "unsupported"
                if claim.status is CapabilityStatus.UNSUPPORTED
                else "restricted"
                if claim.status in {CapabilityStatus.RESTRICTED, CapabilityStatus.UNVERIFIED}
                or claim.mode is not CapabilityMode.NATIVE
                or claim.verification is not VerificationLevel.VERIFIED
                else "supported"
            )
            groups[group][leaf] = p0_status
            if p0_status == "restricted":
                restrictions[key.value] = {
                    **_jsonable(claim.restrictions),
                    "mode": claim.mode.value,
                    "verification": claim.verification.value,
                }
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "backend_version": self.backend_version or "unknown",
            "binding_version": self.binding_version or "unknown",
            **groups,
            "restrictions": restrictions,
            "verified_on": {
                "adapter_version": self.adapter_version,
                "manifest_signature": self.signature,
            },
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _p0_group_leaf(key: CapabilityKey) -> tuple[str | None, str]:
    prefix, leaf = key.value.split(".", 1)
    group = {
        "problem": "problem_classes",
        "constraint": "native_constraints",
        "incremental": "incremental",
        "start": "starts",
        "callback": "callbacks",
        "result": "result_artifacts",
    }.get(prefix)
    return group, leaf


def _mode_from_legacy(level: SupportLevel) -> CapabilityMode:
    return {
        SupportLevel.NATIVE: CapabilityMode.NATIVE,
        SupportLevel.EMULATED_SAFE: CapabilityMode.EMULATED_SAFE,
        SupportLevel.EMULATED_RISKY: CapabilityMode.EMULATED_RISKY,
        SupportLevel.UNSUPPORTED: CapabilityMode.NONE,
        SupportLevel.UNKNOWN: CapabilityMode.UNKNOWN,
    }[level]


def _status_from_legacy(level: SupportLevel, *, coarse: bool = False) -> CapabilityStatus:
    if level is SupportLevel.UNSUPPORTED:
        return CapabilityStatus.UNSUPPORTED
    if level is SupportLevel.UNKNOWN:
        return CapabilityStatus.UNVERIFIED
    if coarse or level is not SupportLevel.NATIVE:
        return CapabilityStatus.RESTRICTED
    return CapabilityStatus.SUPPORTED


_LEGACY_DIRECT: dict[Capability, CapabilityKey] = {
    Capability.LP: CapabilityKey.PROBLEM_LP,
    Capability.MILP: CapabilityKey.PROBLEM_MILP,
    Capability.CONVEX_QP: CapabilityKey.PROBLEM_CONVEX_QP,
    Capability.MIP_START: CapabilityKey.START_MIP,
    Capability.PRIMAL_START: CapabilityKey.START_PRIMAL,
    Capability.DUAL_START: CapabilityKey.START_DUAL,
    Capability.BASIS_START: CapabilityKey.START_BASIS,
    Capability.CALLBACK_PROGRESS: CapabilityKey.CALLBACK_PROGRESS,
    Capability.IIS: CapabilityKey.RESULT_IIS,
    Capability.INFEASIBILITY_CERTIFICATE: CapabilityKey.RESULT_INFEASIBILITY_CERTIFICATE,
}


def project_legacy_manifest(
    manifest: BackendManifest,
    *,
    binding: str = "legacy-adapter",
    binding_version: str | None = None,
    adapter_version: str | None = None,
) -> BackendCapabilityManifestV2:
    """Project the coarse M-track manifest into v2 without upgrading evidence.

    The projection is intentionally conservative. It preserves legacy declarations for
    observability, but projected claims are *not* considered verified v2 capabilities.
    Coarse same-sparsity/structural-update flags become restricted granular claims.
    """

    claims: dict[CapabilityKey, CapabilityClaim] = {}
    for legacy_key, v2_key in _LEGACY_DIRECT.items():
        level = manifest.support(legacy_key)
        claims[v2_key] = CapabilityClaim(
            status=_status_from_legacy(level),
            mode=_mode_from_legacy(level),
            verification=(
                VerificationLevel.UNVERIFIED
                if level is SupportLevel.UNKNOWN
                else VerificationLevel.PROJECTED_LEGACY
            ),
            restrictions=(
                {"legacy_emulation": level.value}
                if level in {SupportLevel.EMULATED_SAFE, SupportLevel.EMULATED_RISKY}
                else {}
            ),
            notes=("projected from BackendManifest v1; native v2 conformance not implied",),
        )

    same = manifest.support(Capability.SAME_SPARSITY_DATA_UPDATE)
    for key in (
        CapabilityKey.INCREMENTAL_VARIABLE_BOUNDS,
        CapabilityKey.INCREMENTAL_RHS,
        CapabilityKey.INCREMENTAL_OBJECTIVE,
        CapabilityKey.INCREMENTAL_MATRIX_VALUES,
    ):
        claims[key] = CapabilityClaim(
            status=_status_from_legacy(same, coarse=same not in {SupportLevel.UNSUPPORTED, SupportLevel.UNKNOWN}),
            mode=_mode_from_legacy(same),
            verification=(
                VerificationLevel.UNVERIFIED
                if same is SupportLevel.UNKNOWN
                else VerificationLevel.PROJECTED_LEGACY
            ),
            restrictions=(
                {"legacy_scope": "same_sparsity_data_update", "granular_operation_not_individually_verified": True}
                if same not in {SupportLevel.UNSUPPORTED, SupportLevel.UNKNOWN}
                else {}
            ),
            notes=("coarse M-track capability projected conservatively",),
        )

    structural = manifest.support(Capability.STRUCTURAL_INCREMENTAL_UPDATE)
    for key in (
        CapabilityKey.INCREMENTAL_MATRIX_STRUCTURE,
        CapabilityKey.INCREMENTAL_ADD_VARIABLE,
        CapabilityKey.INCREMENTAL_DELETE_VARIABLE,
        CapabilityKey.INCREMENTAL_ADD_CONSTRAINT,
        CapabilityKey.INCREMENTAL_DELETE_CONSTRAINT,
    ):
        claims[key] = CapabilityClaim(
            status=_status_from_legacy(structural, coarse=structural not in {SupportLevel.UNSUPPORTED, SupportLevel.UNKNOWN}),
            mode=_mode_from_legacy(structural),
            verification=(
                VerificationLevel.UNVERIFIED
                if structural is SupportLevel.UNKNOWN
                else VerificationLevel.PROJECTED_LEGACY
            ),
            restrictions=(
                {"legacy_scope": "structural_incremental_update", "granular_operation_not_individually_verified": True}
                if structural not in {SupportLevel.UNSUPPORTED, SupportLevel.UNKNOWN}
                else {}
            ),
            notes=("coarse M-track structural capability projected conservatively",),
        )

    reopt = manifest.support(Capability.NATIVE_REOPTIMIZATION)
    claims[CapabilityKey.LIFECYCLE_PERSISTENT] = CapabilityClaim(
        status=_status_from_legacy(reopt, coarse=reopt not in {SupportLevel.UNSUPPORTED, SupportLevel.UNKNOWN}),
        mode=_mode_from_legacy(reopt),
        verification=(VerificationLevel.UNVERIFIED if reopt is SupportLevel.UNKNOWN else VerificationLevel.PROJECTED_LEGACY),
        restrictions=(
            {"legacy_scope": "native_reoptimization", "persistent_lifecycle_not_individually_verified": True}
            if reopt not in {SupportLevel.UNSUPPORTED, SupportLevel.UNKNOWN}
            else {}
        ),
        notes=("native_reoptimization does not automatically prove the full v2 persistent lifecycle",),
    )

    # Current BackendSolveResult always has a primal/objective channel, but whether a
    # backend produces a valid candidate is verified separately by the conformance kit.
    for key in (CapabilityKey.RESULT_PRIMAL, CapabilityKey.RESULT_OBJECTIVE):
        claims.setdefault(
            key,
            CapabilityClaim(
                status=CapabilityStatus.UNVERIFIED,
                mode=CapabilityMode.NATIVE,
                verification=VerificationLevel.UNVERIFIED,
                notes=("adapter result channel exists; successful native production requires conformance",),
            ),
        )

    return BackendCapabilityManifestV2(
        backend=manifest.name,
        backend_version=manifest.version,
        binding=binding,
        binding_version=binding_version,
        adapter_version=adapter_version,
        claims=claims,
        metadata={"legacy_manifest_metadata": dict(manifest.metadata)},
    )


def with_verified_claims(
    manifest: BackendCapabilityManifestV2,
    verified: Mapping[CapabilityKey, CapabilityEvidence],
    *,
    status_overrides: Mapping[CapabilityKey, CapabilityStatus] | None = None,
    restriction_overrides: Mapping[CapabilityKey, Mapping[str, Any]] | None = None,
) -> BackendCapabilityManifestV2:
    claims = dict(manifest.claims)
    status_overrides = status_overrides or {}
    restriction_overrides = restriction_overrides or {}
    for key, evidence in verified.items():
        prior = claims.get(key)
        status = status_overrides.get(
            key,
            prior.status if prior and prior.status not in {CapabilityStatus.UNVERIFIED, CapabilityStatus.UNSUPPORTED} else CapabilityStatus.SUPPORTED,
        )
        restrictions = restriction_overrides.get(key, prior.restrictions if prior else {})
        if status is CapabilityStatus.RESTRICTED and not restrictions:
            restrictions = {"verified_restriction": "see evidence"}
        mode = prior.mode if prior and prior.mode not in {CapabilityMode.UNKNOWN, CapabilityMode.NONE} else CapabilityMode.NATIVE
        claims[key] = CapabilityClaim(
            status=status,
            mode=mode,
            verification=VerificationLevel.VERIFIED,
            restrictions=restrictions,
            evidence=(evidence,),
            notes=prior.notes if prior else (),
        )
    return BackendCapabilityManifestV2(
        backend=manifest.backend,
        backend_version=manifest.backend_version,
        binding=manifest.binding,
        binding_version=manifest.binding_version,
        adapter_version=manifest.adapter_version,
        claims=claims,
        numeric_limits=manifest.numeric_limits,
        metadata=manifest.metadata,
    )

@dataclass(frozen=True, slots=True)
class CapabilityRequirementsV2:
    required: tuple[CapabilityKey, ...]


def requirements_v2_for(problem) -> CapabilityRequirementsV2:
    # Import lazily so capability protocol remains usable during package bootstrap.
    from solverpilot.problem import LinearProblem, QuadraticProblem
    try:
        from solverpilot.conic import ConeKind, ConicProblem
    except Exception:  # pragma: no cover - package bootstrap
        ConeKind = ConicProblem = None
    try:
        from solverpilot.nlp import NLPProblem
    except Exception:  # pragma: no cover
        NLPProblem = None
    try:
        from solverpilot.minlp import MINLPProblem
    except Exception:  # pragma: no cover
        MINLPProblem = None
    try:
        from solverpilot.cp import CPProblem, CPAllDifferentIR, CPExactlyOneIR, CPTableIR, CPElementIR, CPCircuitIR, CPNoOverlapIR, CPCumulativeIR
    except Exception:  # pragma: no cover
        CPProblem = CPAllDifferentIR = CPExactlyOneIR = CPTableIR = CPElementIR = CPCircuitIR = CPNoOverlapIR = CPCumulativeIR = None

    if CPProblem is not None and isinstance(problem, CPProblem):
        keys = [CapabilityKey.PROBLEM_CP]
        for constraint in problem.constraints:
            if CPAllDifferentIR is not None and isinstance(constraint, CPAllDifferentIR): keys.append(CapabilityKey.CONSTRAINT_ALL_DIFFERENT)
            elif CPExactlyOneIR is not None and isinstance(constraint, CPExactlyOneIR): keys.append(CapabilityKey.CONSTRAINT_EXACTLY_ONE)
            elif CPTableIR is not None and isinstance(constraint, CPTableIR): keys.append(CapabilityKey.CONSTRAINT_TABLE)
            elif CPElementIR is not None and isinstance(constraint, CPElementIR): keys.append(CapabilityKey.CONSTRAINT_ELEMENT)
            elif CPCircuitIR is not None and isinstance(constraint, CPCircuitIR): keys.append(CapabilityKey.CONSTRAINT_CIRCUIT)
            elif CPNoOverlapIR is not None and isinstance(constraint, CPNoOverlapIR): keys.append(CapabilityKey.CONSTRAINT_NO_OVERLAP)
            elif CPCumulativeIR is not None and isinstance(constraint, CPCumulativeIR): keys.append(CapabilityKey.CONSTRAINT_CUMULATIVE)
            else: keys.append(CapabilityKey.CONSTRAINT_LINEAR)
        return CapabilityRequirementsV2(tuple(dict.fromkeys(keys)))

    if ConicProblem is not None and isinstance(problem, ConicProblem):
        keys = [CapabilityKey.PROBLEM_CONIC, CapabilityKey.CONSTRAINT_LINEAR]
        if problem.P.nnz:
            keys.append(CapabilityKey.PROBLEM_CONIC_QUADRATIC)
        kinds = {c.kind for c in problem.cones}
        if ConeKind.EXPONENTIAL in kinds:
            keys.append(CapabilityKey.CONSTRAINT_EXPONENTIAL)
        if ConeKind.POWER in kinds:
            keys.append(CapabilityKey.CONSTRAINT_POWER)
        if ConeKind.GENERALIZED_POWER in kinds:
            keys.append(CapabilityKey.CONSTRAINT_GENERALIZED_POWER)
        if ConeKind.SECOND_ORDER in kinds:
            keys.append(CapabilityKey.CONSTRAINT_SOC)
        if ConeKind.ROTATED_SECOND_ORDER in kinds:
            keys.append(CapabilityKey.CONSTRAINT_ROTATED_SOC)
        if ConeKind.POSITIVE_SEMIDEFINITE in kinds:
            keys.append(CapabilityKey.CONSTRAINT_PSD)
        return CapabilityRequirementsV2(tuple(keys))

    if MINLPProblem is not None and isinstance(problem, MINLPProblem):
        return CapabilityRequirementsV2((CapabilityKey.PROBLEM_MINLP, CapabilityKey.CONSTRAINT_NONLINEAR))

    if NLPProblem is not None and isinstance(problem, NLPProblem):
        return CapabilityRequirementsV2((CapabilityKey.PROBLEM_NLP, CapabilityKey.CONSTRAINT_NONLINEAR))

    if isinstance(problem, QuadraticProblem):
        return CapabilityRequirementsV2((CapabilityKey.PROBLEM_CONVEX_QP, CapabilityKey.CONSTRAINT_LINEAR))
    if isinstance(problem, LinearProblem):
        problem_key = CapabilityKey.PROBLEM_MILP if problem.has_integer_variables else CapabilityKey.PROBLEM_LP
        return CapabilityRequirementsV2((problem_key, CapabilityKey.CONSTRAINT_LINEAR))
    raise TypeError(f"unsupported problem type: {type(problem)!r}")


def compatible_v2(
    manifest: BackendCapabilityManifestV2,
    requirements: CapabilityRequirementsV2,
    *,
    allow_restricted: bool = True,
    allow_safe_emulation: bool = False,
    allow_risky_emulation: bool = False,
    require_verified: bool = True,
) -> tuple[bool, tuple[CapabilityCheck, ...]]:
    checks = tuple(
        manifest.check(
            key,
            allow_restricted=allow_restricted,
            allow_safe_emulation=allow_safe_emulation,
            allow_risky_emulation=allow_risky_emulation,
            require_verified=require_verified,
        )
        for key in requirements.required
    )
    return all(check.usable for check in checks), checks
