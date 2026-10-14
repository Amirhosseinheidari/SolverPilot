"""Cached primal evidence must refer to an owned, unchanged candidate."""

import numpy as np
import pytest

from solverpilot import LinearProblem
from solverpilot.backends.base import BackendSolveResult
from solverpilot.capabilities import BackendManifest, Capability, SupportLevel
from solverpilot.runtime import executor
from solverpilot.validate import core, optimality


class SharedBufferBackend:
    manifest = BackendManifest(
        name="shared-buffer", capabilities={Capability.LP: SupportLevel.NATIVE},
    )

    def __init__(self, dual):
        self.x = np.array([1.])
        self.dual = dual
        self.raw = {"canonical_dual": self.dual, "nested": {"history": [1., 2.]}}

    def is_available(self):
        return True

    def solve(self, problem):
        return BackendSolveResult("optimal", self.x, float(problem.c @ self.x), self.raw)


@pytest.mark.parametrize("dual_container", [list, np.array])
@pytest.mark.parametrize("objective", [0., 1.])
def test_validation_certificate_and_result_share_an_owned_snapshot(monkeypatch, dual_container, objective):
    problem = LinearProblem.from_data(
        A=[[1.]], c=[objective], variable_lower=[0.], variable_upper=[2.],
        constraint_lower=[1.], constraint_upper=[np.inf],
    )
    original_dual = [-objective, 0.]
    backend = SharedBufferBackend(dual_container(original_dual))
    seen = {}
    verify = optimality._verify_optimality_with_primal_report

    def validate(problem, candidate, **kwargs):
        seen["candidate"] = candidate.x
        report = core.validate_solution(problem, candidate, **kwargs)
        # Simulate a subsequent backend solve overwriting the returned native
        # workspace after primal validation but before certificate evaluation.
        backend.x[0] = 0.
        backend.dual[0] = 9.
        backend.raw["canonical_dual"] = [8., 8.]
        backend.raw["nested"]["history"][0] = 99.
        return report

    def verify_snapshot(problem, x, dual, report, **kwargs):
        assert x is seen["candidate"]
        assert not x.flags.writeable
        assert not np.shares_memory(x, backend.x)
        assert list(dual) == original_dual
        return verify(problem, x, dual, report, **kwargs)

    monkeypatch.setattr(executor, "validate_solution", validate)
    monkeypatch.setattr(optimality, "_verify_optimality_with_primal_report", verify_snapshot)
    result = executor.execute(problem, backend)
    assert result.x.tolist() == [1.]
    assert result.validation.valid and result.optimality_evidence.independently_verified_optimal
    assert result.objective == pytest.approx(objective)
    monkeypatch.setattr(optimality, "_verify_optimality_with_primal_report", verify)
    assert optimality.verify_optimality(problem, result.x, original_dual, fast_reject=True).verified
    assert not optimality.verify_optimality(problem, backend.x, original_dual, fast_reject=True).verified
    assert list(result.raw_statistics["canonical_dual"]) == original_dual
    assert result.raw_statistics["nested"]["history"] == (1., 2.)
    backend.x[0] = 2.
    assert result.x.tolist() == [1.]


def test_overwritten_dual_cannot_promote_an_originally_invalid_certificate(monkeypatch):
    problem = LinearProblem.from_data(
        A=[[1.]], c=[1.], variable_lower=[0.], variable_upper=[2.],
        constraint_lower=[1.], constraint_upper=[np.inf],
    )
    backend = SharedBufferBackend(np.array([0., 0.]))

    def validate(problem, candidate, **kwargs):
        report = core.validate_solution(problem, candidate, **kwargs)
        backend.dual[0] = -1.  # A later workspace now happens to hold a valid dual.
        return report

    monkeypatch.setattr(executor, "validate_solution", validate)
    result = executor.execute(problem, backend)
    assert result.validation.valid
    assert not result.optimality_evidence.independently_verified_optimal
    assert list(result.raw_statistics["canonical_dual"]) == [0., 0.]
