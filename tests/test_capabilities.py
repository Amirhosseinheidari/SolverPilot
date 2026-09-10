from solverpilot import (
    BackendManifest,
    Capability,
    LinearProblem,
    QuadraticProblem,
    SupportLevel,
    VariableDomain,
    compatible,
    requirements_for,
)


def lp(integer=False):
    return LinearProblem.from_data(
        A=[[1.0]],
        c=[1.0],
        variable_lower=[0.0],
        variable_upper=[1.0],
        constraint_lower=[0.0],
        constraint_upper=[1.0],
        domains=[VariableDomain.INTEGER if integer else VariableDomain.CONTINUOUS],
    )


def test_lp_requirement():
    assert requirements_for(lp()).required == frozenset({Capability.LP})


def test_milp_requirement():
    assert requirements_for(lp(integer=True)).required == frozenset({Capability.MILP})


def test_qp_requirement():
    q = QuadraticProblem.from_data(
        P=[[1.0]], A=[[1.0]], q=[1.0],
        variable_lower=[0.0], variable_upper=[1.0],
        constraint_lower=[0.0], constraint_upper=[1.0],
    )
    assert requirements_for(q).required == frozenset({Capability.CONVEX_QP})


def test_risky_emulation_is_not_compatible_by_default():
    m = BackendManifest("fake", capabilities={Capability.MILP: SupportLevel.EMULATED_RISKY})
    req = requirements_for(lp(integer=True))
    assert not compatible(m, req)
    assert compatible(m, req, allow_risky_emulation=True)


def test_unknown_is_not_compatible():
    m = BackendManifest("fake")
    assert not compatible(m, requirements_for(lp()))
