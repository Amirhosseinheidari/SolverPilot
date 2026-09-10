from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from solverpilot.problem import ConvexityStatus, LinearProblem, QuadraticProblem


class ProblemClass(str, Enum):
    LP="LP"; MILP="MILP"; CONVEX_QP="CONVEX_QP"; QP_UNVERIFIED="QP_UNVERIFIED"; CONIC="CONIC"; NLP="NLP"; MINLP="MINLP"; CP="CP"; UNKNOWN="UNKNOWN"
class ConvexityClass(str, Enum):
    CONFIRMED_CONVEX="confirmed_convex"; CERTIFIED_CONVEX="certified_convex"; UNKNOWN="unknown"; NOT_APPLICABLE="not_applicable"
class IntegralityClass(str, Enum):
    CONTINUOUS="continuous"; MIXED_INTEGER="mixed_integer"; DISCRETE="discrete"; UNKNOWN="unknown"
class ProofRequirement(str, Enum):
    FEASIBILITY="feasibility"; OPTIMALITY="optimality"; GLOBAL_OPTIMALITY="global_optimality"


@dataclass(frozen=True, slots=True)
class ProblemDescriptor:
    problem_class: ProblemClass
    convexity: ConvexityClass
    integrality: IntegralityClass
    objective_family: str
    constraint_families: tuple[str, ...]
    proof_requirement: ProofRequirement
    source_type: str
    def to_dict(self) -> dict[str,Any]:
        p=asdict(self)
        for k in ("problem_class","convexity","integrality","proof_requirement"): p[k]=getattr(self,k).value
        return p


def describe_problem(problem: object) -> ProblemDescriptor:
    if isinstance(problem,QuadraticProblem):
        confirmed=problem.convexity_status is ConvexityStatus.CONFIRMED
        return ProblemDescriptor(ProblemClass.CONVEX_QP if confirmed else ProblemClass.QP_UNVERIFIED,ConvexityClass.CONFIRMED_CONVEX if confirmed else ConvexityClass.UNKNOWN,IntegralityClass.CONTINUOUS,"quadratic",("linear",),ProofRequirement.OPTIMALITY,type(problem).__qualname__)
    if isinstance(problem,LinearProblem):
        integer=bool(problem.has_integer_variables)
        return ProblemDescriptor(ProblemClass.MILP if integer else ProblemClass.LP,ConvexityClass.CONFIRMED_CONVEX,IntegralityClass.MIXED_INTEGER if integer else IntegralityClass.CONTINUOUS,"linear",("linear",),ProofRequirement.GLOBAL_OPTIMALITY if integer else ProofRequirement.OPTIMALITY,type(problem).__qualname__)
    try:
        from solverpilot.conic.ir import ConicProblem
        if isinstance(problem,ConicProblem): return ProblemDescriptor(ProblemClass.CONIC,ConvexityClass.CONFIRMED_CONVEX,IntegralityClass.CONTINUOUS,"linear_or_conic",("affine","cone"),ProofRequirement.OPTIMALITY,type(problem).__qualname__)
    except ImportError: pass
    try:
        from solverpilot.minlp.ir import MINLPProblem
        if isinstance(problem,MINLPProblem): return ProblemDescriptor(ProblemClass.MINLP,ConvexityClass.CERTIFIED_CONVEX,IntegralityClass.MIXED_INTEGER,"nonlinear",("nonlinear","integrality"),ProofRequirement.GLOBAL_OPTIMALITY,type(problem).__qualname__)
    except ImportError: pass
    try:
        from solverpilot.nlp.ir import NLPProblem
        if isinstance(problem,NLPProblem): return ProblemDescriptor(ProblemClass.NLP,ConvexityClass.UNKNOWN,IntegralityClass.CONTINUOUS,"nonlinear",("nonlinear",),ProofRequirement.OPTIMALITY,type(problem).__qualname__)
    except ImportError: pass
    try:
        from solverpilot.cp.ir import CPProblem
        if isinstance(problem,CPProblem):
            fam=tuple(sorted({getattr(c,"kind",type(c).__name__) for c in problem.constraints}))
            return ProblemDescriptor(ProblemClass.CP,ConvexityClass.NOT_APPLICABLE,IntegralityClass.DISCRETE,"linear_integer_or_none",fam,ProofRequirement.GLOBAL_OPTIMALITY if problem.objective is not None else ProofRequirement.FEASIBILITY,type(problem).__qualname__)
    except ImportError: pass
    return ProblemDescriptor(ProblemClass.UNKNOWN,ConvexityClass.UNKNOWN,IntegralityClass.UNKNOWN,"unknown",(),ProofRequirement.FEASIBILITY,type(problem).__qualname__)


def capability_requirements_for_descriptor(descriptor: ProblemDescriptor):
    from .v2 import CapabilityKey, CapabilityRequirementsV2
    mapping={ProblemClass.LP:CapabilityKey.PROBLEM_LP,ProblemClass.MILP:CapabilityKey.PROBLEM_MILP,ProblemClass.CONVEX_QP:CapabilityKey.PROBLEM_CONVEX_QP,ProblemClass.CONIC:CapabilityKey.PROBLEM_CONIC,ProblemClass.NLP:CapabilityKey.PROBLEM_NLP,ProblemClass.MINLP:CapabilityKey.PROBLEM_MINLP,ProblemClass.CP:CapabilityKey.PROBLEM_CP}
    if descriptor.problem_class is ProblemClass.QP_UNVERIFIED: raise ValueError("unverified QP cannot be mapped to a verified convex-QP capability")
    key=mapping.get(descriptor.problem_class)
    if key is None: raise ValueError(f"no capability mapping for problem class {descriptor.problem_class.value}")
    keys=[key]
    family_map={"linear":CapabilityKey.CONSTRAINT_LINEAR,"nonlinear":CapabilityKey.CONSTRAINT_NONLINEAR,"all_different":CapabilityKey.CONSTRAINT_ALL_DIFFERENT,"exactly_one":CapabilityKey.CONSTRAINT_EXACTLY_ONE,"table":CapabilityKey.CONSTRAINT_TABLE,"element":CapabilityKey.CONSTRAINT_ELEMENT,"circuit":CapabilityKey.CONSTRAINT_CIRCUIT,"no_overlap":CapabilityKey.CONSTRAINT_NO_OVERLAP,"cumulative":CapabilityKey.CONSTRAINT_CUMULATIVE}
    for family in descriptor.constraint_families:
        if family in family_map: keys.append(family_map[family])
    return CapabilityRequirementsV2(tuple(dict.fromkeys(keys)))
