from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from solverpilot import LinearProblem, QuadraticProblem, PublicStatus
from solverpilot.backends import BundledHighsCAPIBackend, BundledOSQPCAPIBackend
from solverpilot.benchmark import RunSpec, benchmark_registry, run_benchmark
from solverpilot.benchmark.runner import _clean_pythonpath, _protocol_id
from solverpilot.runtime.executor import execute


def _qp(*, p_scale: float = 1.0, q=(-2.0, -4.0), a_scale: float = 1.0) -> QuadraticProblem:
    return QuadraticProblem.from_data(
        P=np.array([[2.0 * p_scale, 0.0], [0.0, 2.0 * p_scale]]),
        q=np.asarray(q, dtype=float),
        A=np.array([[1.0 * a_scale, 1.0 * a_scale]]),
        variable_lower=[0.0, 0.0],
        variable_upper=[10.0, 10.0],
        constraint_lower=[-np.inf],
        constraint_upper=[10.0 * a_scale],
    )


def _available() -> tuple[BundledOSQPCAPIBackend, BundledHighsCAPIBackend]:
    osqp = BundledOSQPCAPIBackend()
    highs = BundledHighsCAPIBackend()
    if not (osqp.is_available() and highs.is_available()):
        pytest.skip("verified CasADi 3.7.2 bundled C-API solver libraries are unavailable")
    return osqp, highs


def test_bundled_capi_manifest_provenance_is_exact():
    osqp, highs = _available()
    assert osqp.manifest.version == "0.6.3"
    assert highs.manifest.version == "1.10.0"
    for backend in (osqp, highs):
        md = backend.manifest.metadata
        assert md["verification_only"] is True
        assert md["direct_c_api"] is True
        assert md["bridge_package_version"] == "3.7.2"
        assert backend.manifest.version in md["verified_solver_versions"]
        assert len(md["library_sha256"]) == 64
        assert len(md["header_sha256"]) == 64


def test_benchmark_registry_accepts_exactly_verified_bundled_capi_versions():
    _available()
    reg = benchmark_registry(("bundled-osqp-capi", "bundled-highs-capi"))
    assert {b.manifest.name for b in reg.all()} == {"bundled-osqp-capi", "bundled-highs-capi"}


def test_direct_highs_capi_lp_and_basis_reuse():
    _, backend = _available()
    p1 = LinearProblem.from_data(
        A=[[1.0, 1.0]], c=[1.0, 2.0],
        variable_lower=[0.0, 0.0], variable_upper=[1.0, 1.0],
        constraint_lower=[1.0], constraint_upper=[np.inf],
    )
    p2 = LinearProblem.from_data(
        A=[[1.0, 1.0]], c=[1.1, 2.0],
        variable_lower=[0.0, 0.0], variable_upper=[1.0, 1.0],
        constraint_lower=[1.0], constraint_upper=[np.inf],
    )
    r1 = execute(p1, backend)
    r2 = execute(p2, backend)
    assert r1.status is PublicStatus.VALID_OPTIMAL and r1.validation.valid
    assert r2.status is PublicStatus.VALID_OPTIMAL and r2.validation.valid
    assert not r1.trace.reuse_applied
    assert r2.trace.reuse_applied
    assert "explicit_basis_hot_start" in (r2.trace.reuse_mode or "")
    assert r2.raw_statistics["basis_applied"] is True
    backend.close()


def test_direct_highs_capi_binary_milp_matches_known_optimum():
    _, backend = _available()
    problem = LinearProblem.from_data(
        A=[[1.0, 1.0]], c=[-3.0, -2.0],
        variable_lower=[0.0, 0.0], variable_upper=[1.0, 1.0],
        constraint_lower=[-np.inf], constraint_upper=[1.0],
        domains=["binary", "binary"],
    )
    result = execute(problem, backend)
    assert result.status is PublicStatus.VALID_OPTIMAL
    assert result.validation.valid
    assert np.isclose(result.objective, -3.0, atol=1e-9)
    assert np.allclose(result.x, [1.0, 0.0], atol=1e-9)
    backend.close()


def test_direct_osqp_capi_qp_and_vector_reuse():
    backend, _ = _available()
    p1 = _qp()
    p2 = _qp(q=(-2.2, -4.0))
    r1 = execute(p1, backend)
    r2 = execute(p2, backend)
    assert r1.status is PublicStatus.VALID_OPTIMAL and r1.validation.valid
    assert r2.status is PublicStatus.VALID_OPTIMAL and r2.validation.valid
    assert np.allclose(r1.x, [1.0, 2.0], atol=1e-5)
    assert r2.trace.reuse_applied
    assert r2.trace.reuse_mode == "persistent_workspace+vector_update+automatic_warm_start"
    assert r2.raw_statistics["factorization_reuse_claimed"] is False
    backend.close()


def test_direct_osqp_capi_same_sparsity_matrix_update():
    backend, _ = _available()
    r1 = execute(_qp(), backend)
    r2 = execute(_qp(p_scale=1.2, a_scale=1.1), backend)
    assert r1.validation.valid and r2.validation.valid
    assert r2.trace.reuse_applied
    assert "same_sparsity_matrix_value_update" in (r2.trace.reuse_mode or "")
    assert r2.raw_statistics["matrix_values_updated"] is True
    backend.close()


def test_direct_osqp_capi_limit_candidate_is_not_promoted_when_infeasible_not_detected():
    backend, _ = _available()
    problem = QuadraticProblem.from_data(
        P=np.eye(1), q=[0.0], A=[[1.0], [1.0]],
        variable_lower=[-np.inf], variable_upper=[np.inf],
        constraint_lower=[1.0, -np.inf], constraint_upper=[np.inf, 0.0],
    )
    result = execute(problem, backend)
    # The bundled OSQP 0.6.3 build reaches its iteration limit on this tiny
    # contradictory example.  Crucially, the invalid candidate is rejected by
    # independent validation instead of being promoted to a feasible solution.
    assert result.status is PublicStatus.INVALID_SOLUTION
    assert result.validation is not None and result.validation.valid is False
    assert result.raw_statistics["osqp_status"] == "maximum iterations reached"
    assert result.raw_statistics["certificate_kind"] is None
    backend.close()


BASIC = """NAME          BASIC
ROWS
 N  COST
 G  DEMAND
COLUMNS
    X1        COST       1        DEMAND     1
    X2        COST       2        DEMAND     1
RHS
    RHS1      DEMAND     1
BOUNDS
 UP BND1      X1         1
 UP BND1      X2         1
ENDATA
"""


def test_no_site_worker_runs_direct_highs_capi_end_to_end(tmp_path: Path):
    _available()
    data = tmp_path / "data"; data.mkdir()
    (data / "a.mps").write_text(BASIC)
    manifest = tmp_path / "bench.test"; manifest.write_text("a.mps\n")
    reference = tmp_path / "bench.solu"; reference.write_text("=opt= a 1\n")
    output = tmp_path / "rows.jsonl"
    backends = ("bundled-highs-capi",)
    spec = RunSpec(
        dataset_dir=data, manifest=manifest, reference=reference,
        backends=backends, output_jsonl=output,
        hard_timeout_s=20, worker_python_mode="no_site",
    )
    report = run_benchmark(spec, registry=benchmark_registry(backends))
    row = json.loads(output.read_text().strip())
    assert report["counters"]["completed"] == 1
    assert row["state"] == "solved"
    assert row["validated"] is True
    assert row["reference_check"] == "objective_matches_optimum"
    assert row["backend"] == "bundled-highs-capi"
    assert row["thread_policy"]["worker_python_mode"] == "no_site"


def test_worker_python_mode_is_protocol_versioned(tmp_path: Path):
    _available()
    data = tmp_path / "data"; data.mkdir()
    manifest = tmp_path / "bench.test"; manifest.write_text("a.mps\n")
    common = dict(dataset_dir=data, manifest=manifest, reference=None, backends=("bundled-highs-capi",), output_jsonl=tmp_path/"x.jsonl")
    reg = benchmark_registry(("bundled-highs-capi",))
    normal = RunSpec(**common, worker_python_mode="normal")
    clean = RunSpec(**common, worker_python_mode="no_site")
    assert _protocol_id(normal, reg) != _protocol_id(clean, reg)


def test_clean_pythonpath_excludes_platform_hook_paths():
    value = _clean_pythonpath()
    assert "python-hooks" not in value
    assert "pyvenv-overrides" not in value


def test_direct_highs_capi_survives_repeated_mixed_health_probes():
    _available()
    from solverpilot import builtin_backend_candidates
    from solverpilot.backends import BackendProbeStatus, probe_backends

    for _ in range(3):
        reports = probe_backends(builtin_backend_candidates())
        by_name = {report.backend: report for report in reports}
        assert by_name["bundled-highs-capi"].status is BackendProbeStatus.HEALTHY
