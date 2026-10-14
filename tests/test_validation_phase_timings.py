"""Stage accounting and evidence equivalence when reusing primal validation."""
from dataclasses import asdict, replace
import inspect
import json

import numpy as np
import pytest

from solverpilot import LinearProblem, QuadraticProblem
from solverpilot.backends.base import BackendSolveResult
from solverpilot.capabilities import BackendManifest, Capability, SupportLevel
from solverpilot.runtime import executor
from solverpilot.trace import PhaseTimings, SolveTrace
from solverpilot.validate import CandidateSolution, PublicStatus, ValidationTolerances
from solverpilot.validate import core, lp_dual, optimality


def _lp(*, sense="minimize", integer=False):
    return LinearProblem.from_data(
        A=[[1.]], c=[1.], variable_lower=[-np.inf], variable_upper=[np.inf],
        constraint_lower=[1.], constraint_upper=[2.],
        objective_sense=sense, domains=["integer" if integer else "continuous"],
    )


class _Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class _Backend:
    manifest = BackendManifest(
        name="stage-accounting-test",
        capabilities={cap: SupportLevel.NATIVE for cap in (
            Capability.LP, Capability.MILP, Capability.CONVEX_QP,
        )},
    )

    def __init__(self, clock, *, x, dual, objective, status="optimal"):
        self.clock = clock
        self.x = None if x is None else np.array(x, dtype=float)
        self.dual = dual
        self.objective = objective
        self.status = status

    def is_available(self):
        return True

    def solve(self, problem):
        self.clock.advance(13.)
        return BackendSolveResult(self.status, self.x, self.objective, {
            "canonical_dual": self.dual,
            "phase_timings": {"backend_build_s": 2., "backend_update_s": 3., "solve_s": 7.},
        })


@pytest.mark.parametrize("case,x,dual,objective,primal_calls,proof_calls,repair_calls,verified", [
    ("verified", [1.], [-1., 0.], 1., 1, 1, 0, True),
    ("failed_dual", [1.], [0., 0.], 1., 1, 1, 0, False),
    ("no_dual", [1.], None, 1., 1, 0, 0, False),
    ("invalid_primal", [0.], [-1., 0.], 0., 1, 1, 0, False),
    ("repaired_dual", [1.], [-1., 1e-12], 1., 1, 2, 1, True),
    ("no_candidate", None, [-1., 0.], None, 0, 0, 0, False),
    ("nonfinite_candidate", [np.inf], [-1., 0.], None, 1, 1, 0, False),
    ("wrong_objective", [1.], [-1., 0.], 99., 2, 1, 0, False),
    ("no_reported_objective", [1.], [-1., 0.], None, 1, 1, 0, True),
])
def test_executor_accounts_for_each_stage_once(
    monkeypatch, case, x, dual, objective, primal_calls, proof_calls, repair_calls, verified,
):
    clock = _Clock()
    problem = _lp()
    seen_reports = []
    calls = {"primal": 0, "proof": 0, "repair": 0}
    real_check = optimality._verify_optimality_with_primal_report
    real_repair = lp_dual.prepare_lp_dual

    def primal(*args, **kwargs):
        calls["primal"] += 1
        clock.advance(2.)
        return core.validate_solution(*args, **kwargs)

    def certificate(problem, x, dual, report, **kwargs):
        calls["proof"] += 1
        seen_reports.append(report)
        clock.advance(3.)
        return real_check(problem, x, dual, report, **kwargs)

    def repair(*args, **kwargs):
        calls["repair"] += 1
        clock.advance(5.)
        return real_repair(*args, **kwargs)

    def duplicate_primal(*args, **kwargs):
        pytest.fail("the private executor path repeated canonical primal validation")

    monkeypatch.setattr(executor, "perf_counter", clock)
    monkeypatch.setattr(optimality, "perf_counter", clock)
    monkeypatch.setattr(executor, "validate_solution", primal)
    monkeypatch.setattr(optimality, "validate_solution", duplicate_primal)
    monkeypatch.setattr(optimality, "_verify_optimality_with_primal_report", certificate)
    monkeypatch.setattr(lp_dual, "prepare_lp_dual", repair)
    status = "infeasible" if x is None else "optimal"
    result = executor.execute(problem, _Backend(clock, x=x, dual=dual, objective=objective, status=status))

    assert calls == {"primal": primal_calls, "proof": proof_calls, "repair": repair_calls}
    timings = result.trace.timings
    assert timings.backend_build_s == 2.
    assert timings.backend_update_s == 3.
    assert timings.solve_s == 7.
    assert timings.backend_total_s == 13.
    assert timings.primal_validate_s == 2. * primal_calls
    assert timings.certificate_s == 3. * proof_calls + 5. * repair_calls
    assert timings.validate_s == timings.primal_validate_s + timings.certificate_s
    assert timings.total_s == 13. + timings.validate_s
    evidence = result.optimality_evidence
    assert evidence.independently_verified_optimal is verified
    assert not evidence.certificate_verified
    if repair_calls:
        assert seen_reports[0] is seen_reports[1] is result.validation
        assert tuple(result.raw_statistics["canonical_dual"]) == (-1., 1e-12)
        assert tuple(result.raw_statistics["prepared_canonical_dual"]) == (-1., 0.)
        assert not result.raw_statistics["original_optimality_check"]["verified"]
    if case == "wrong_objective":
        assert result.status is PublicStatus.INVALID_SOLUTION
        assert not result.validation.valid
        assert seen_reports[0].valid and seen_reports[0] is not result.validation
        assert evidence.dual_verified and evidence.gap_verified
        assert not evidence.primal_validated
    if case == "invalid_primal":
        assert result.status is PublicStatus.INVALID_SOLUTION
        assert result.raw_statistics["optimality_check"]["reason"] == "primal candidate outside tolerances"
    if x is None:
        assert result.status is PublicStatus.INFEASIBLE
        assert "optimality_check" not in result.raw_statistics


@pytest.mark.parametrize("fast_reject", [False, True])
@pytest.mark.parametrize("kind,x,dual", [
    ("lp", [1.], [-1., 0.]),
    ("lp", [0.], [-1., 0.]),
    ("lp", [1.], [0., 0.]),
    ("lp", [1.], [-1., 1e-12]),
    ("lp", [1.], [np.nan, 0.]),
    ("lp", [1.], [0.]),
    ("lp", [1.], ["bad", 0.]),
    ("max", [2.], [1., 0.]),
    ("milp", [1.], [-1., 0.]),
    ("qp", [1.], [0.]),
    ("qp", [0.], [0.]),
])
def test_cached_and_public_checks_have_identical_evidence(kind, x, dual, fast_reject):
    if kind == "qp":
        problem = QuadraticProblem.from_data(
            P=[[2.]], A=np.empty((0, 1)), q=[-2.],
            variable_lower=[-2.], variable_upper=[2.],
            constraint_lower=[], constraint_upper=[], objective_offset=7.,
        )
    else:
        problem = _lp(sense="maximize" if kind == "max" else "minimize", integer=kind == "milp")
    tol = ValidationTolerances(feasibility=1e-6, objective_abs=1e-6)
    report = core.validate_solution(problem, CandidateSolution(x), tolerances=tol)
    public = optimality.verify_optimality(problem, x, dual, tolerances=tol, fast_reject=fast_reject)
    cached = optimality._verify_optimality_with_primal_report(
        problem, x, dual, report, tolerances=tol, fast_reject=fast_reject,
    )
    assert asdict(cached) == asdict(public)


def test_public_verifier_revalidates_each_candidate_and_has_no_report_parameter(monkeypatch):
    problem = _lp()
    calls = []

    def checked(problem, candidate, **kwargs):
        calls.append(np.array(candidate.x))
        return core.validate_solution(problem, candidate, **kwargs)

    monkeypatch.setattr(optimality, "validate_solution", checked)
    assert optimality.verify_optimality(problem, [1.], [-1., 0.], fast_reject=True).verified
    assert not optimality.verify_optimality(problem, [0.], [-1., 0.], fast_reject=True).verified
    assert len(calls) == 2
    report = core.validate_solution(problem, CandidateSolution([1.]))
    with pytest.raises(TypeError, match="primal_report"):
        optimality.verify_optimality(problem, [0.], [-1., 0.], primal_report=report)


def test_executor_reuses_the_actual_custom_tolerance_report():
    clock = _Clock()
    tol = ValidationTolerances(feasibility=2e-6, objective_abs=2e-6, objective_rel=0.)
    problem = _lp()
    x = [1. - 1e-6]
    assert not core.validate_solution(problem, CandidateSolution(x)).valid
    result = executor.execute(problem, _Backend(clock, x=x, dual=[-1., 0.], objective=x[0]), tolerances=tol)
    assert result.validation.valid
    assert result.optimality_evidence.independently_verified_optimal
    assert result.raw_statistics["optimality_check"] == asdict(
        optimality.verify_optimality(problem, x, [-1., 0.], tolerances=tol, fast_reject=True)
    )


@pytest.mark.parametrize("recover", [False, True])
def test_solve_preserves_split_and_adds_termination_recovery_to_certificate(monkeypatch, recover):
    from solverpilot.runtime import auto
    from solverpilot.validate import recovery

    problem = _lp()
    clock = _Clock()
    backend = _Backend(
        clock, x=None if recover else [1.], dual=None if recover else [-1., 0.],
        objective=None if recover else 1., status="infeasible" if recover else "optimal",
    )
    original = executor.execute(problem, backend)
    primal_s, proof_s = (0., 0.) if recover else (2., 3.)
    old = PhaseTimings(
        backend_total_s=13., solve_s=7., backend_build_s=2., backend_update_s=3.,
        primal_validate_s=primal_s, certificate_s=proof_s,
        validate_s=primal_s + proof_s, total_s=13. + primal_s + proof_s,
    )
    original = replace(original, trace=replace(original.trace, timings=old))
    clock.now = 0.

    def execute(*args, **kwargs):
        clock.advance(old.total_s)
        return original

    def recover_certificate(*args, **kwargs):
        clock.advance(11.)
        return recovery.LPCertificate(problem.data_hash, "unknown", False, "test recovery")

    monkeypatch.setattr(auto, "perf_counter", clock)
    monkeypatch.setattr(auto, "execute", execute)
    monkeypatch.setattr(recovery, "recover_lp_certificate", recover_certificate)
    result = auto.solve(problem, backend=backend, certificate_recovery=20. if recover else None)
    timings = result.trace.timings
    assert timings.primal_validate_s == primal_s
    assert timings.certificate_s == proof_s + (11. if recover else 0.)
    assert timings.validate_s == timings.primal_validate_s + timings.certificate_s
    assert timings.total_s == old.total_s + (11. if recover else 0.)
    assert timings.backend_total_s == old.backend_total_s
    assert result.trace.execution_id == original.trace.execution_id
    assert result.optimality_evidence == original.optimality_evidence


def test_phase_timings_keep_old_positional_and_serialized_contracts():
    old_fields = [
        "inspect_s", "plan_s", "backend_build_s", "solve_s", "validate_s",
        "diagnose_s", "total_s", "backend_update_s", "backend_total_s",
    ]
    old_record = dict(zip(old_fields, range(1, 10)))
    restored = PhaseTimings(**old_record)
    assert restored == PhaseTimings(*range(1, 10))
    assert restored.primal_validate_s == restored.certificate_s == 0.
    signature = inspect.signature(PhaseTimings)
    for name in ("primal_validate_s", "certificate_s"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters[name].default == 0.
    with pytest.raises(TypeError):
        PhaseTimings(*range(11))

    timings = PhaseTimings(validate_s=5., primal_validate_s=2., certificate_s=3.)
    trace = SolveTrace("structure", "data", "lp", timings=timings, source_sha256=None)
    encoded = json.loads(json.dumps(trace.to_dict()))
    assert encoded["timings"]["primal_validate_s"] == 2.
    assert encoded["timings"]["certificate_s"] == 3.
    assert PhaseTimings(**encoded["timings"]) == timings
