from __future__ import annotations

import inspect
import json
from pathlib import Path
import tomllib
import warnings

import numpy as np
import pytest

import solverpilot
from solverpilot import (
    BackendManifest,
    BackendSolveResult,
    BackendUnavailableError,
    BudgetNotSupportedError,
    Capability,
    CapabilityMismatchError,
    LinearProblem,
    NoCompatibleBackendError,
    SolverPilotError,
    QuadraticProblem,
    SolveBudget,
    SupportLevel,
    UnknownBackendError,
    default_registry,
    execute,
    solve,
    solve_production,
)
from solverpilot.backends import ScipyHighsLPBackend

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_MANIFEST = ROOT / "docs/history/frozen-m30/PUBLIC-API-M30.json"

RESEARCH_ONLY = {
    "M22_OFFICIAL_EVIDENCE",
    "SelectiveLPDecision",
    "decide_selective_lp_backend",
    "production_evidence_from_m22_gate",
    "production_evidence_from_m24_public_ood",
    "production_evidence_from_m25_opportunity",
    "production_evidence_from_m26_validation",
    "production_evidence_from_m27_heldout",
    "production_evidence_from_m28_native_choose",
    "production_evidence_from_m29_value_audit",
}


def _signature(obj):
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def test_public_api_manifest_exactly_matches_runtime_all_and_signatures():
    payload = json.loads(PUBLIC_MANIFEST.read_text())
    assert payload["schema"] == "optimind.public_api.v1"
    assert payload["package_version"] == "0.0.35"
    rows = payload["symbols"]
    expected_names = ["SolverPilotError" if row["name"] == "OptiMindError" else row["name"] for row in rows]
    assert expected_names == solverpilot.__all__
    for row in rows:
        current_name = "SolverPilotError" if row["name"] == "OptiMindError" else row["name"]
        obj = getattr(solverpilot, current_name)
        expected_module = row["module"].replace("optimind.", "solverpilot.") if row["module"] else None
        expected_qualname = "SolverPilotError" if row["qualname"] == "OptiMindError" else row["qualname"]
        assert expected_module == getattr(obj, "__module__", None)
        assert expected_qualname == getattr(obj, "__qualname__", None)
        assert row["signature"] == _signature(obj)


def test_research_milestone_symbols_are_not_in_frozen_top_level_api():
    assert RESEARCH_ONLY.isdisjoint(solverpilot.__all__)
    import solverpilot.experimental as experimental
    assert RESEARCH_ONLY.issubset(set(experimental.__all__))


def test_pre_m30_research_alias_warns_and_points_to_experimental_namespace():
    import solverpilot.experimental as experimental
    # Access through getattr so the module-level compatibility shim is exercised.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old = getattr(solverpilot, "production_evidence_from_m29_value_audit")
    assert old is experimental.production_evidence_from_m29_value_audit
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_public_exception_hierarchy_preserves_builtin_categories():
    assert issubclass(BackendUnavailableError, (SolverPilotError, RuntimeError))
    assert issubclass(CapabilityMismatchError, (SolverPilotError, ValueError))
    assert issubclass(NoCompatibleBackendError, (SolverPilotError, RuntimeError))
    assert issubclass(UnknownBackendError, (SolverPilotError, KeyError))
    assert issubclass(BudgetNotSupportedError, (SolverPilotError, NotImplementedError))


def test_unknown_backend_uses_stable_exception():
    with pytest.raises(UnknownBackendError, match="unknown backend"):
        default_registry().get("definitely-not-a-backend")


def test_execute_unavailable_backend_uses_stable_exception():
    class NeverAvailable:
        manifest = BackendManifest(
            name="never-available",
            capabilities={Capability.LP: SupportLevel.NATIVE},
        )

        def is_available(self):
            return False

        def solve(self, problem):  # pragma: no cover - must never run
            return BackendSolveResult("solver_error", None)

    problem = LinearProblem.from_data(A=[[1.0]], c=[1.0], variable_lower=[0.0], variable_upper=[1.0], constraint_lower=[0.0], constraint_upper=[1.0])
    with pytest.raises(BackendUnavailableError, match="never-available"):
        execute(problem, NeverAvailable())


def test_execute_capability_mismatch_uses_stable_exception():
    qp = QuadraticProblem.from_data(P=[[2.0]], A=[[1.0]], q=[-2.0], variable_lower=[0.0], variable_upper=[2.0], constraint_lower=[0.0], constraint_upper=[2.0])
    with pytest.raises(CapabilityMismatchError):
        execute(qp, ScipyHighsLPBackend(method="highs-ds"))


def test_unenforceable_budget_uses_stable_exception():
    qp = QuadraticProblem.from_data(P=[[2.0]], A=[[1.0]], q=[-2.0], variable_lower=[0.0], variable_upper=[2.0], constraint_lower=[0.0], constraint_upper=[2.0])
    with pytest.raises(BudgetNotSupportedError):
        solve(qp, backend="scipy-slsqp-qp-bridge", budget=SolveBudget(threads=2))


def test_production_solve_remains_conservative_and_auditable():
    problem = LinearProblem.from_data(
        A=[[1.0, 1.0]],
        c=[1.0, 2.0],
        variable_lower=[0.0, 0.0],
        variable_upper=[1.0, 1.0],
        constraint_lower=[1.0],
        constraint_upper=[np.inf],
    )
    result, decision = solve_production(problem)
    assert result.validation and result.validation.valid
    assert decision.plan.selected_backend == "scipy-highs-ds"
    assert decision.auto_performance_ranking_enabled is False


def test_pyproject_metadata_matches_frozen_runtime_version_and_readme():
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = payload["project"]
    assert project["version"] == solverpilot.__version__
    assert project["readme"] == "README.md"
    assert project["requires-python"].startswith(">=3.")
    assert project["optional-dependencies"]["native"] == project["optional-dependencies"]["open-source"]


def test_package_tree_does_not_embed_generated_distribution_metadata():
    # pip/build legitimately create sibling egg-info, build and dist folders.
    # They must not be embedded in the importable package. The release manifest
    # checker separately inspects the contents of the actual built wheel.
    package = ROOT / "src" / "solverpilot"
    assert not list(package.rglob("*.egg-info"))
    assert not (package / "build").exists()
    assert not (package / "dist").exists()


def test_prepublic_optimind_namespace_is_not_shipped_in_current_source():
    assert not (ROOT / "src/optimind").exists()
    assert not hasattr(solverpilot, "OptiMindError")
    assert (ROOT / "docs/history/frozen-rc6/SOLVERPILOT-BRAND-MIGRATION-RC6.md").is_file()
