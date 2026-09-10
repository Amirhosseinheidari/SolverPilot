from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping
import numpy as np
from solverpilot.nlp.ir import NLPProblem

@dataclass(frozen=True, slots=True)
class ConvexityCertificate:
    schema: str
    objective_curvature: str
    constraint_curvatures: tuple[str, ...]
    supported_atom_policy: str
    data_hash: str
    notes: tuple[str, ...] = ()

@dataclass(frozen=True, slots=True)
class MINLPProblem:
    relaxation: NLPProblem
    domains: tuple[str, ...]
    integer_indices: tuple[int, ...]
    binary_indices: tuple[int, ...]
    convexity_certificate: ConvexityCertificate
    semantic_hash: str
    data_hash: str
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if len(self.domains) != self.relaxation.n_variables:
            raise ValueError('domain vector length mismatch')
        object.__setattr__(self, 'metadata', MappingProxyType(dict(self.metadata)))

    @property
    def n_variables(self) -> int:
        return self.relaxation.n_variables
