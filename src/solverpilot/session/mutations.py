from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import sparse

from solverpilot.problem import LinearProblem, QuadraticProblem


class MutationKind(str, Enum):
    OBJECTIVE_VECTOR = "objective_vector"
    OBJECTIVE_OFFSET = "objective_offset"
    OBJECTIVE_SENSE = "objective_sense"
    OBJECTIVE_QUADRATIC_VALUES = "objective_quadratic_values"
    QUADRATIC_SPARSITY_CHANGED = "quadratic_sparsity_changed"
    VARIABLE_BOUNDS = "variable_bounds"
    CONSTRAINT_BOUNDS = "constraint_bounds"
    MATRIX_VALUES_SAME_SPARSITY = "matrix_values_same_sparsity"
    MATRIX_SPARSITY_CHANGED = "matrix_sparsity_changed"
    VARIABLES_ADDED_REMOVED = "variables_added_removed"
    CONSTRAINTS_ADDED_REMOVED = "constraints_added_removed"
    INTEGRALITY_CHANGED = "integrality_changed"


@dataclass(frozen=True, slots=True)
class MutationRecord:
    revision_before: int
    revision_after: int
    kinds: tuple[MutationKind, ...]
    old_structural_hash: str
    new_structural_hash: str
    old_data_hash: str
    new_data_hash: str

    @property
    def is_noop(self) -> bool:
        return not self.kinds

    @property
    def structural_change(self) -> bool:
        return self.old_structural_hash != self.new_structural_hash


def _same_sparse_pattern(a: sparse.csr_matrix, b: sparse.csr_matrix) -> bool:
    return (
        a.shape == b.shape
        and np.array_equal(a.indptr, b.indptr)
        and np.array_equal(a.indices, b.indices)
    )


def classify_mutations(
    old: LinearProblem | QuadraticProblem,
    new: LinearProblem | QuadraticProblem,
    *,
    revision_before: int,
    revision_after: int,
) -> MutationRecord:
    if type(old) is not type(new):
        raise TypeError("session update cannot change problem class")

    old_linear = old.linear if isinstance(old, QuadraticProblem) else old
    new_linear = new.linear if isinstance(new, QuadraticProblem) else new
    kinds: list[MutationKind] = []

    if old_linear.n_variables != new_linear.n_variables:
        kinds.append(MutationKind.VARIABLES_ADDED_REMOVED)
    if old_linear.n_constraints != new_linear.n_constraints:
        kinds.append(MutationKind.CONSTRAINTS_ADDED_REMOVED)

    if not np.array_equal(old_linear.c, new_linear.c):
        kinds.append(MutationKind.OBJECTIVE_VECTOR)
    if old_linear.objective_offset != new_linear.objective_offset:
        kinds.append(MutationKind.OBJECTIVE_OFFSET)
    if old_linear.objective_sense != new_linear.objective_sense:
        kinds.append(MutationKind.OBJECTIVE_SENSE)
    if (
        not np.array_equal(old_linear.variable_lower, new_linear.variable_lower)
        or not np.array_equal(old_linear.variable_upper, new_linear.variable_upper)
    ):
        kinds.append(MutationKind.VARIABLE_BOUNDS)
    if (
        not np.array_equal(old_linear.constraint_lower, new_linear.constraint_lower)
        or not np.array_equal(old_linear.constraint_upper, new_linear.constraint_upper)
    ):
        kinds.append(MutationKind.CONSTRAINT_BOUNDS)
    if not np.array_equal(old_linear.domains, new_linear.domains):
        kinds.append(MutationKind.INTEGRALITY_CHANGED)

    same_a_pattern = _same_sparse_pattern(old_linear.A, new_linear.A)
    if not same_a_pattern:
        kinds.append(MutationKind.MATRIX_SPARSITY_CHANGED)
    elif not np.array_equal(old_linear.A.data, new_linear.A.data):
        kinds.append(MutationKind.MATRIX_VALUES_SAME_SPARSITY)

    if isinstance(old, QuadraticProblem):
        same_p_pattern = _same_sparse_pattern(old.P, new.P)
        if not same_p_pattern:
            kinds.append(MutationKind.QUADRATIC_SPARSITY_CHANGED)
        elif not np.array_equal(old.P.data, new.P.data):
            kinds.append(MutationKind.OBJECTIVE_QUADRATIC_VALUES)

    # Keep enum order deterministic and avoid duplicates caused by broad changes.
    deduped = tuple(dict.fromkeys(kinds))
    return MutationRecord(
        revision_before=revision_before,
        revision_after=revision_after,
        kinds=deduped,
        old_structural_hash=old.structural_hash,
        new_structural_hash=new.structural_hash,
        old_data_hash=old.data_hash,
        new_data_hash=new.data_hash,
    )
