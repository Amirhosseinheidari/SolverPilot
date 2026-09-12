import numpy as np
import pytest
from solverpilot.model import Model, ExponentialCone
from solverpilot.conic import ClarabelBackend, verify_conic_optimality
from solverpilot.runtime import solve_verified


@pytest.mark.parametrize("kind", ["soc", "rsoc", "psd"])
def test_native_independent_conic_bound(kind):
    pytest.importorskip("clarabel")
    m = Model()
    if kind == "soc":
        x = m.variable(2, lower=[3, 4], upper=[3, 4])
        t = m.variable(lower=0, upper=10)
        m.soc(t, x)
        m.minimize(t)
        expected = 5.0
    elif kind == "rsoc":
        u = m.variable(lower=0, upper=5)
        v = m.variable(lower=0, upper=5)
        z = m.variable(1, lower=1, upper=1)
        m.rotated_soc(u, v, z)
        m.minimize(u + v)
        expected = np.sqrt(2.0)
    else:
        t = m.variable(lower=0, upper=5)
        m.psd(t * m.constant(np.eye(2)) + m.constant([[0, 1], [1, 0]]))
        m.minimize(t)
        expected = 1.0
    compiled = m.compile()
    summary, r = solve_verified(compiled, backend=ClarabelBackend())
    assert summary.objective == pytest.approx(expected, abs=1e-6)
    assert summary.optimality == "independent_numerical_bound"
    assert r.raw_statistics["optimality_check"]["dual_bound"] <= expected + 1e-12
    bad = np.ones_like(r.x) * 100
    assert not verify_conic_optimality(
        compiled.execution_ir,
        bad,
        r.raw_statistics["canonical_linear_dual"],
        r.raw_statistics["canonical_cone_duals"],
    ).verified


def test_transcendental_membership_not_falsely_verified():
    pytest.importorskip("clarabel")
    m = Model()
    x = m.variable(3, lower=[1, 1, 0], upper=[1, 1, 10])
    m.add_in_set(x, ExponentialCone())
    m.minimize(x[2])
    r = m.solve(backend=ClarabelBackend())
    assert r.validation.valid and not r.optimality_evidence.independently_verified_optimal
    assert "transcendental" in r.raw_statistics["optimality_check"]["reason"]


def test_malformed_or_unbounded_domain_dual_cannot_be_verified():
    m = Model()
    x = m.variable(2, lower=-2, upper=2)
    t = m.variable(lower=0, upper=10)
    m.soc(t, x)
    m.minimize(t)
    p = m.compile().execution_ir
    point = np.zeros(3)
    y = np.zeros(3)
    for z in ([1, 2, 0], [np.nan, 0, 0], [-1, 0, 0], [1, 0]):
        assert not verify_conic_optimality(p, point, y, [z]).verified
    assert not verify_conic_optimality(p, ["bad"], y, [[0, 0, 0]]).verified
    assert not verify_conic_optimality(p, point, [0], [[0, 0, 0]]).verified
    m = Model()
    x = m.variable(2)
    t = m.variable()
    m.soc(t, x)
    m.minimize(t)
    p = m.compile().execution_ir
    r = verify_conic_optimality(p, point, y, [[0.9, 0, 0]])
    assert not r.verified and "original-domain" in r.reason


@pytest.mark.parametrize(
    "kind,value",
    [
        ("soc", [0.9999999999999999, 1, 0]),
        ("rsoc", [0.5, 1, 1.0000000000000002]),
        ("psd", [[1, 1.0000000000000002], [1.0000000000000002, 1]]),
    ],
)
def test_repaired_duals_still_require_exact_membership(kind, value):
    from solverpilot.conic.ir import ConeKind
    from solverpilot.conic.optimality import repair_dual, _dual_member

    kinds = {
        "soc": ConeKind.SECOND_ORDER,
        "rsoc": ConeKind.ROTATED_SECOND_ORDER,
        "psd": ConeKind.POSITIVE_SEMIDEFINITE,
    }
    k = kinds[kind]
    a = np.asarray(value, dtype=float)
    assert not _dual_member(k, a)
    assert _dual_member(k, repair_dual(k, a))
