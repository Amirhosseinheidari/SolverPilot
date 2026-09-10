from __future__ import annotations

from dataclasses import dataclass

from solverpilot.problem import ConvexityStatus, LinearProblem, QuadraticProblem

from .enums import Capability, SupportLevel
from .manifest import BackendManifest


@dataclass(frozen=True, slots=True)
class CapabilityRequirements:
    required: frozenset[Capability]


def requirements_for(problem: LinearProblem | QuadraticProblem) -> CapabilityRequirements:
    if isinstance(problem, QuadraticProblem):
        if problem.convexity_status is not ConvexityStatus.CONFIRMED:
            raise ValueError("unverified quadratic problem cannot be routed as convex_qp")
        return CapabilityRequirements(frozenset({Capability.CONVEX_QP}))
    if isinstance(problem, LinearProblem):
        return CapabilityRequirements(
            frozenset({Capability.MILP if problem.has_integer_variables else Capability.LP})
        )
    raise TypeError(f"unsupported problem type: {type(problem)!r}")


def compatible(
    manifest: BackendManifest,
    requirements: CapabilityRequirements,
    *,
    allow_risky_emulation: bool = False,
) -> bool:
    allowed = {SupportLevel.NATIVE, SupportLevel.EMULATED_SAFE}
    if allow_risky_emulation:
        allowed.add(SupportLevel.EMULATED_RISKY)
    return all(manifest.support(cap) in allowed for cap in requirements.required)
