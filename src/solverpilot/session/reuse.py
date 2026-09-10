from __future__ import annotations

from dataclasses import dataclass

from solverpilot.capabilities import BackendManifest, Capability
from solverpilot.problem import LinearProblem, QuadraticProblem

from .mutations import MutationKind, MutationRecord


@dataclass(frozen=True, slots=True)
class ReuseAssessment:
    backend: str
    considered: tuple[Capability, ...]
    supported: tuple[Capability, ...]
    safe_candidate: bool
    applied: bool
    reason: str


def assess_reuse(
    problem: LinearProblem | QuadraticProblem,
    manifest: BackendManifest,
    mutations: tuple[MutationRecord, ...],
) -> ReuseAssessment:
    """Assess *potential* reuse from declared capabilities.

    This function never claims reuse was applied. Applying solver state requires a
    backend-specific stateful interface, which M1 deliberately does not fake.
    """

    kinds = {kind for record in mutations for kind in record.kinds}
    if not mutations or not kinds:
        return ReuseAssessment(
            backend=manifest.name,
            considered=(),
            supported=(),
            safe_candidate=False,
            applied=False,
            reason="no model mutation since previous solve",
        )

    structural_kinds = {
        MutationKind.VARIABLES_ADDED_REMOVED,
        MutationKind.CONSTRAINTS_ADDED_REMOVED,
        MutationKind.MATRIX_SPARSITY_CHANGED,
        MutationKind.QUADRATIC_SPARSITY_CHANGED,
        MutationKind.INTEGRALITY_CHANGED,
    }
    if kinds & structural_kinds:
        considered = (
            Capability.STRUCTURAL_INCREMENTAL_UPDATE,
            Capability.NATIVE_REOPTIMIZATION,
        )
    else:
        candidates: list[Capability] = [Capability.SAME_SPARSITY_DATA_UPDATE]
        if isinstance(problem, QuadraticProblem):
            candidates.extend([Capability.PRIMAL_START, Capability.DUAL_START])
        elif problem.has_integer_variables:
            candidates.append(Capability.MIP_START)
        else:
            candidates.extend([Capability.BASIS_START, Capability.PRIMAL_START])
        candidates.append(Capability.NATIVE_REOPTIMIZATION)
        considered = tuple(candidates)

    supported = tuple(
        cap for cap in considered if manifest.supports_without_risky_emulation(cap)
    )
    return ReuseAssessment(
        backend=manifest.name,
        considered=considered,
        supported=supported,
        safe_candidate=bool(supported),
        applied=False,
        reason=(
            "backend declares potentially reusable mechanisms; execution evidence is required "
            "before reuse is marked applied"
            if supported
            else "backend declares no compatible reuse mechanism for these mutations"
        ),
    )
