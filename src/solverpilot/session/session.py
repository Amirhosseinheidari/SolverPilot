from __future__ import annotations

from threading import RLock
from solverpilot._synchronization import serialized

from dataclasses import replace
from typing import Any

import numpy as np
from scipy import sparse

from solverpilot.backends import Backend, BackendRegistry
from solverpilot.plan import PlannerContext, SolveBudget, SolveIntent
from solverpilot.problem import LinearProblem, QuadraticProblem
from solverpilot.runtime import SolveResult, solve as runtime_solve

from .mutations import MutationRecord, classify_mutations
from .reuse import ReuseAssessment, assess_reuse


class Session:
    """Versioned repeated-solve session with explicit mutation tracking.

    M1 tracks and assesses reuse safely but does not pretend to apply native solver
    state. Actual basis/factorization/incumbent injection is reserved for native
    backend adapters that expose a stateful contract.
    """

    def __init__(
        self,
        problem: LinearProblem | QuadraticProblem,
        *,
        registry: BackendRegistry | None = None,
    ) -> None:
        self._lock = RLock()
        self._problem = problem
        if registry is None:
            from solverpilot.runtime import default_registry
            registry = default_registry()
        self._registry = registry
        self._revision = 0
        self._history: list[MutationRecord] = []
        self._last_result: SolveResult | None = None
        self._last_solve_revision: int | None = None
        self._last_reuse_assessment: ReuseAssessment | None = None

    @property
    def problem(self) -> LinearProblem | QuadraticProblem:
        return self._problem

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def history(self) -> tuple[MutationRecord, ...]:
        return tuple(self._history)

    @property
    def last_result(self) -> SolveResult | None:
        return self._last_result

    @property
    def last_reuse_assessment(self) -> ReuseAssessment | None:
        return self._last_reuse_assessment

    @serialized
    def update(self, **changes: Any) -> MutationRecord:
        old = self._problem
        new = _updated_problem(old, changes)
        next_revision = self._revision if old.data_hash == new.data_hash else self._revision + 1
        record = classify_mutations(
            old,
            new,
            revision_before=self._revision,
            revision_after=next_revision,
        )
        # Descriptive updates do not change the mathematical hash/revision,
        # but must still be visible to the caller.
        self._problem = new
        if not record.is_noop:
            self._revision = next_revision
            self._history.append(record)
        return record

    @serialized
    def solve(
        self,
        *,
        backend: str | Backend | None = None,
        intent: SolveIntent | str = SolveIntent.BALANCED,
        budget: SolveBudget | None = None,
        diagnose_infeasible: bool = False,
    ) -> SolveResult:
        previous_backend = None if self._last_result is None else self._last_result.trace.backend
        context = PlannerContext(
            previous_backend=previous_backend,
            session_revision=self._revision,
        )
        result = runtime_solve(
            self._problem,
            registry=self._registry,
            backend=backend,
            intent=intent,
            budget=budget,
            context=context,
            diagnose_infeasible=diagnose_infeasible,
        )
        result = replace(result, session_revision=self._revision)

        if self._last_result is not None and result.trace.backend is not None:
            since = self._mutations_since_last_solve()
            if backend is not None and not isinstance(backend, str):
                manifest = backend.manifest
            else:
                manifest = self._registry.get(result.trace.backend).manifest
            assessment = assess_reuse(self._problem, manifest, since)
            if result.trace.reuse_applied is True:
                mode = result.trace.reuse_mode or "backend-reported reuse"
                assessment = replace(
                    assessment,
                    applied=True,
                    safe_candidate=True,
                    reason=f"backend reported applied reuse: {mode}",
                )
            self._last_reuse_assessment = assessment
        else:
            self._last_reuse_assessment = None

        self._last_result = result
        self._last_solve_revision = self._revision
        return result

    def _mutations_since_last_solve(self) -> tuple[MutationRecord, ...]:
        if self._last_solve_revision is None:
            return ()
        return tuple(
            record for record in self._history if record.revision_after > self._last_solve_revision
        )


def _updated_problem(
    problem: LinearProblem | QuadraticProblem,
    changes: dict[str, Any],
) -> LinearProblem | QuadraticProblem:
    allowed_linear = {
        "A",
        "c",
        "variable_lower",
        "variable_upper",
        "constraint_lower",
        "constraint_upper",
        "domains",
        "objective_sense",
        "objective_offset",
        "name",
        "metadata",
    }
    allowed_qp = {
        "P",
        "A",
        "q",
        "variable_lower",
        "variable_upper",
        "constraint_lower",
        "constraint_upper",
        "objective_sense",
        "objective_offset",
        "metadata",
    }

    if isinstance(problem, LinearProblem):
        unknown = set(changes) - allowed_linear
        if unknown:
            raise TypeError(f"unknown LinearProblem update fields: {sorted(unknown)}")
        data = {
            "A": problem.A,
            "c": problem.c,
            "variable_lower": problem.variable_lower,
            "variable_upper": problem.variable_upper,
            "constraint_lower": problem.constraint_lower,
            "constraint_upper": problem.constraint_upper,
            "domains": problem.domains,
            "objective_sense": problem.objective_sense,
            "objective_offset": problem.objective_offset,
            "name": problem.name,
            "metadata": problem.metadata,
        }
        data.update(changes)
        return LinearProblem.from_data(**data)

    if isinstance(problem, QuadraticProblem):
        unknown = set(changes) - allowed_qp
        if unknown:
            raise TypeError(f"unknown QuadraticProblem update fields: {sorted(unknown)}")
        linear = problem.linear
        P = changes.get("P", problem.P)
        A = changes.get("A", linear.A)
        q = changes.get("q", linear.c)
        vl = changes.get("variable_lower", linear.variable_lower)
        vu = changes.get("variable_upper", linear.variable_upper)
        cl = changes.get("constraint_lower", linear.constraint_lower)
        cu = changes.get("constraint_upper", linear.constraint_upper)
        sense = changes.get("objective_sense", linear.objective_sense)
        offset = changes.get("objective_offset", linear.objective_offset)
        metadata = changes.get("metadata", problem.metadata)

        # Avoid an unnecessary eigenvalue calculation when P is unchanged.
        if "P" not in changes:
            new_linear = LinearProblem.from_data(
                A=A,
                c=q,
                variable_lower=vl,
                variable_upper=vu,
                constraint_lower=cl,
                constraint_upper=cu,
                objective_sense=sense,
                objective_offset=offset,
                metadata=metadata,
            )
            return QuadraticProblem(
                linear=new_linear,
                P=problem.P,
                convexity_status=problem.convexity_status,
                metadata=metadata,
            )

        return QuadraticProblem.from_data(
            P=P,
            A=A,
            q=q,
            variable_lower=vl,
            variable_upper=vu,
            constraint_lower=cl,
            constraint_upper=cu,
            objective_sense=sense,
            objective_offset=offset,
            verify_convexity=True,
            metadata=metadata,
        )

    raise TypeError(f"unsupported problem type: {type(problem)!r}")
