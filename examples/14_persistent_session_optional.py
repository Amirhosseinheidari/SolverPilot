"""Optional persistent semantic-model session with verified native patch routing."""
from solverpilot.model import Model
from solverpilot.session import PersistentSession, MutationExecutionPath, conform_persistent_backend
from solverpilot.backends import BundledHighsCAPIBackend

backend = BundledHighsCAPIBackend(threads=1, solver="simplex")
if not backend.is_available():
    print("PersistentSession example skipped: bundled HiGHS verification backend is unavailable.")
else:
    m = Model("persistent-lp")
    x = m.variable(2, lower=0.0, upper=5.0)
    cost = m.parameter(2, value=[1.0, 2.0])
    m.add(x.sum() >= 1.0)
    m.minimize(cost @ x)

    report = conform_persistent_backend(backend)
    assert report.passed
    with PersistentSession(m, backend, capabilities=report.manifest) as session:
        first = session.solve()
        cost.value = [3.0, 1.0]
        second = session.solve()

        print("first_path:", first.trace.path.value)
        print("second_path:", second.trace.path.value)
        print("backend_reuse_applied:", second.trace.backend_reuse_applied)

        assert first.trace.path is MutationExecutionPath.COLD_BUILD
        assert second.trace.path in {MutationExecutionPath.NATIVE_PATCH, MutationExecutionPath.SAFE_REBUILD}
