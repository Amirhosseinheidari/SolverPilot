from dataclasses import dataclass

from solverpilot.backends import BackendRegistry
from solverpilot.capabilities import BackendManifest
from solverpilot.trace import SolveTrace


@dataclass
class FakeBackend:
    _name: str
    available_flag: bool = True

    @property
    def manifest(self):
        return BackendManifest(self._name)

    def is_available(self):
        return self.available_flag

    def solve(self, problem):
        raise NotImplementedError


def test_registry_filters_availability():
    r = BackendRegistry()
    r.register(FakeBackend("a", True))
    r.register(FakeBackend("b", False))
    assert r.names() == ("a", "b")
    assert [x.manifest.name for x in r.available()] == ["a"]


def test_duplicate_backend_rejected():
    r = BackendRegistry()
    r.register(FakeBackend("a"))
    try:
        r.register(FakeBackend("a"))
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("duplicate registration should fail")


def test_trace_serializes():
    t = SolveTrace(
        problem_structural_hash="s",
        problem_data_hash="d",
        problem_class="lp",
        backend="fake",
        validation_valid=True,
    )
    d = t.to_dict()
    assert d["schema_version"] == "0.3"
    assert d["backend"] == "fake"


def test_trace_captures_backend_reuse_evidence():
    from solverpilot.backends import ScipyVendoredHighsDevBackend
    from solverpilot import LinearProblem, execute
    import numpy as np

    backend = ScipyVendoredHighsDevBackend(solver="simplex")
    if not backend.is_available():
        return
    p1 = LinearProblem.from_data(
        A=[[1.0, 1.0]], c=[1.0, 2.0],
        variable_lower=[0.0, 0.0], variable_upper=[10.0, 10.0],
        constraint_lower=[1.0], constraint_upper=[np.inf],
    )
    p2 = LinearProblem.from_data(
        A=[[1.0, 1.0]], c=[2.0, 1.0],
        variable_lower=[0.0, 0.0], variable_upper=[10.0, 10.0],
        constraint_lower=[1.1], constraint_upper=[np.inf],
    )
    execute(p1, backend)
    r2 = execute(p2, backend)
    assert r2.trace.reuse_applied is True
    assert r2.trace.reuse_mode == "explicit_basis_hot_start"
