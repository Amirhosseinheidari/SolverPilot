from __future__ import annotations

import itertools

import numpy as np
import pytest

from solverpilot.backends import ScipyHighsBackend
from solverpilot.bridges import BridgePolicy, ExactnessClass, MappingAvailability, choose_indicator_path
from solverpilot.capabilities.v2 import (
    BackendCapabilityManifestV2,
    CapabilityClaim,
    CapabilityEvidence,
    CapabilityKey,
    CapabilityMode,
    CapabilityStatus,
    VerificationLevel,
)
from solverpilot.model import CompileError, DomainError, IndicatorConstraint, Model
from solverpilot.runtime import execute


def _linear_feasible(compiled, x, tol=1e-10):
    p = compiled.execution_ir
    xx = np.asarray(x, dtype=float)
    ax = np.asarray(p.A @ xx, dtype=float)
    return (
        np.all(xx >= p.variable_lower - tol)
        and np.all(xx <= p.variable_upper + tol)
        and np.all(ax >= p.constraint_lower - tol)
        and np.all(ax <= p.constraint_upper + tol)
    )


def _semantic_indicator_feasible(z, x, *, active, relation, rhs):
    if z != active:
        return True
    if relation == "le": return x <= rhs
    if relation == "ge": return x >= rhs
    if relation == "eq": return x == rhs
    raise AssertionError(relation)


def test_indicator_is_preserved_semantically_until_compile():
    m=Model(); z=m.binary(name="z"); x=m.integer(lower=-2,upper=4,name="x")
    c=m.indicator(z, x <= 1, name="guard")
    m.minimize(x)
    assert isinstance(c, IndicatorConstraint)
    payload=m._semantic_payload(include_parameter_values=False)
    assert payload["constraints"][0]["kind"]=="IndicatorConstraint"
    assert payload["constraints"][0]["indicator"]==z.id.value
    compiled=m.compile(use_cache=False)
    assert compiled.compilation_report.n_transformations==1
    assert compiled.transformation_tape[0]["bridge_id"]=="solverpilot.bridge.indicator.certified-big-m"


def test_indicator_requires_scalar_binary_variable_and_scalar_body():
    m=Model(); x=m.variable(); y=m.variable(2); z=m.variable(lower=0,upper=1)
    with pytest.raises((DomainError, Exception), match="binary"):
        m.indicator(z, x <= 1)
    zz=m.binary(2)
    with pytest.raises(Exception, match="scalar"):
        m.indicator(zz, x <= 1)
    zs=m.binary()
    with pytest.raises(Exception, match="scalar"):
        m.indicator(zs, y <= 1)


@pytest.mark.parametrize("active", [0,1])
@pytest.mark.parametrize("relation", ["le","ge","eq"])
def test_certified_big_m_is_exact_by_exhaustive_integer_enumeration(active, relation):
    m=Model(); z=m.binary(); x=m.integer(lower=-3,upper=4)
    rhs=1
    rel={"le": x<=rhs, "ge": x>=rhs, "eq": x==rhs}[relation]
    m.indicator(z, rel, active_value=active)
    m.minimize(x)
    compiled=m.compile(use_cache=False)
    for zv,xv in itertools.product([0,1], range(-3,5)):
        sem=_semantic_indicator_feasible(zv,xv,active=active,relation=relation,rhs=rhs)
        low=_linear_feasible(compiled,[zv,xv])
        assert sem==low, (active,relation,zv,xv,compiled.transformation_tape)


def test_big_m_certificate_is_tight_box_bound_not_arbitrary_constant():
    m=Model(); z=m.binary(); x=m.integer(lower=0,upper=5)
    m.indicator(z, x <= 2)
    m.minimize(-x)
    c=m.compile(use_cache=False)
    cert=c.transformation_tape[0]["preconditions"][0]
    assert cert["kind"]=="certified-affine-big-m"
    assert cert["payload"]["m_values"]["upper"]==pytest.approx(3.0)
    assert cert["payload"]["body_min"]==pytest.approx(-2.0)
    assert cert["payload"]["body_max"]==pytest.approx(3.0)


def test_unbounded_needed_range_rejects_instead_of_guessing_big_m():
    m=Model(); z=m.binary(); x=m.variable(lower=0,upper=np.inf)
    m.indicator(z, x <= 2); m.minimize(x)
    with pytest.raises(CompileError, match="finite Big-M"):
        m.compile(use_cache=False)


def test_fixed_active_indicator_materializes_body_without_big_m():
    m=Model(); z=m.variable(lower=1,upper=1,domain="binary"); x=m.integer(lower=0,upper=5)
    m.indicator(z, x <= 2); m.minimize(x)
    c=m.compile(use_cache=False)
    step=c.transformation_tape[0]
    assert step["bridge_id"]=="solverpilot.bridge.indicator.fixed"
    assert step["preconditions"][0]["kind"]=="fixed-indicator-active"
    np.testing.assert_allclose(c.execution_ir.A.toarray(), [[0,1]])
    assert c.execution_ir.constraint_upper[0]==pytest.approx(2)


def test_fixed_inactive_indicator_drops_constraint_exactly():
    m=Model(); z=m.variable(lower=0,upper=0,domain="binary"); x=m.integer(lower=0,upper=5)
    m.indicator(z, x <= 2, active_value=1); m.minimize(x)
    c=m.compile(use_cache=False)
    assert c.execution_ir.n_constraints==0
    step=c.transformation_tape[0]
    assert step["generated_target_ids"]==[]
    assert step["preconditions"][0]["kind"]=="fixed-indicator-inactive"


def test_parameterized_indicator_is_plan_sensitive_and_revalidates_certificate():
    m=Model(); z=m.binary(); x=m.integer(lower=0,upper=10); p=m.parameter(value=3.0)
    m.indicator(z, p*x <= 6); m.minimize(x)
    a=m.compile(); sem=a.semantic_hash; data=a.data_hash; plan=a.compilation_hash
    assert p.id.value in a.plan_sensitive_dependencies
    assert a.parameter_map[p.id.value]["plan_sensitive"] is True
    p.value=1.0
    b=m.compile()
    assert b.compilation_report.cache_status=="plan_sensitive_recompile"
    assert b.semantic_hash==sem and b.data_hash!=data
    assert b.compilation_hash!=plan
    assert b.precondition_certificate_hashes!=a.precondition_certificate_hashes
    p.value=3.0
    c=m.compile()
    assert c.compilation_report.cache_status=="snapshot_hit"
    assert c.compilation_hash==a.compilation_hash


def test_tape_and_source_map_are_auditable_and_dual_is_explicitly_unavailable():
    m=Model(); z=m.binary(); x=m.integer(lower=-2,upper=2)
    src=m.indicator(z, x==0); m.minimize(x)
    c=m.compile(use_cache=False)
    step=c.transformation_tape[0]
    assert step["exactness"]==ExactnessClass.EXACT_EQUIVALENT.value
    assert step["primal_mapping"]==MappingAvailability.IDENTITY.value
    assert step["dual_mapping"]==MappingAvailability.UNAVAILABLE.value
    assert len(step["generated_target_ids"])==2
    sm=c.source_map["constraints"][src.entity_id.value]
    assert sm["execution_rows"]==[0,1]
    assert sm["generated_target_ids"]==step["generated_target_ids"]


def test_original_space_validation_checks_indicator_semantics():
    m=Model(); z=m.binary(); x=m.integer(lower=0,upper=5)
    m.indicator(z, x<=2); m.minimize(x)
    c=m.compile(use_cache=False)
    assert c.validate_original(m,[1,2]).valid
    bad=c.validate_original(m,[1,4])
    assert not bad.valid
    assert any("indicator" in v for v in bad.violations)
    # Inactive premise allows the body to be violated.
    assert c.validate_original(m,[0,4]).valid


def test_real_solve_reconstructs_and_validates_original_indicator_model():
    m=Model(); z=m.binary(); x=m.integer(lower=0,upper=5)
    m.indicator(z, x<=2); m.add(z>=1); m.minimize(-x)
    c=m.compile(use_cache=False)
    r=execute(c.execution_ir,ScipyHighsBackend())
    assert r.x is not None
    original=c.reconstruct_primal(r.x)
    assert c.validate_original(m,original).valid
    assert original[1]==pytest.approx(2.0,abs=1e-7)


def _verified_indicator_manifest():
    ev=CapabilityEvidence(
        evidence_id="p4-mock-indicator",kind="unit",backend_versions=("1",),binding_versions=("1",),adapter_versions=("0.0.35",)
    )
    claim=CapabilityClaim(
        status=CapabilityStatus.SUPPORTED,mode=CapabilityMode.NATIVE,verification=VerificationLevel.VERIFIED,evidence=(ev,)
    )
    return BackendCapabilityManifestV2(
        backend="mock",backend_version="1",binding="mock",binding_version="1",adapter_version="0.0.35",
        claims={CapabilityKey.CONSTRAINT_INDICATOR:claim},
    )


def test_bridge_path_prefers_native_only_when_transport_exists():
    caps=_verified_indicator_manifest(); policy=BridgePolicy()
    native=choose_indicator_path(caps,native_transport_available=True,policy=policy)
    assert native.selected_path=="native" and native.native_usable
    current=choose_indicator_path(caps,native_transport_available=False,policy=policy)
    assert current.selected_path=="certified_big_m"
    assert current.native_usable
    assert any("no native-indicator transport" in r for r in current.rejected_paths)


def test_compiler_records_rejected_native_path_when_capability_exists_but_ir_cannot_transport_it():
    m=Model(); z=m.binary(); x=m.integer(lower=0,upper=5); m.indicator(z,x<=2); m.minimize(x)
    c=m.compile(use_cache=False,capabilities=_verified_indicator_manifest())
    assert any("no native-indicator transport" in r for r in c.compilation_report.rejected_bridge_paths)
    assert c.capability_signature.startswith("backend-capability-v2:")


def test_clone_preserves_indicator_semantics_but_uses_new_entity_ids():
    m=Model(); z=m.binary(); x=m.integer(lower=0,upper=3); m.indicator(z,x<=1,active_value=0); m.minimize(x)
    clone=m.clone()
    assert isinstance(clone.constraints[0],IndicatorConstraint)
    assert clone.constraints[0].active_value==0
    assert clone.semantic_hash!=m.semantic_hash  # IDs are ownership-local by contract.
    a=m.compile(use_cache=False).execution_ir
    b=clone.compile(use_cache=False).execution_ir
    np.testing.assert_allclose(a.A.toarray(),b.A.toarray())
    np.testing.assert_allclose(a.constraint_lower,b.constraint_lower)
    np.testing.assert_allclose(a.constraint_upper,b.constraint_upper)


def test_ordinary_p2_parameter_cache_semantics_remain_unchanged():
    m=Model(); x=m.variable(); p=m.parameter(value=1.0); m.add(x>=p); m.minimize(x)
    a=m.compile(); p.value=2.0; b=m.compile()
    assert a.compilation_hash==b.compilation_hash
    assert b.compilation_report.cache_status=="partial_recompile"
    assert b.transformation_tape==()


def test_default_bridge_registry_is_deterministic_and_safe_only():
    from solverpilot.bridges import default_bridge_registry
    reg=default_bridge_registry(); policy=BridgePolicy()
    ids=[s.bridge_id for s in reg.candidates("indicator",policy=policy)]
    assert ids==[
        "solverpilot.bridge.indicator.native",
        "solverpilot.bridge.indicator.fixed",
        "solverpilot.bridge.indicator.certified-big-m",
    ]
    assert all(s.exactness is ExactnessClass.EXACT_EQUIVALENT for s in reg.candidates("indicator",policy=policy))


def test_transformation_tape_schema_validates_real_compilation():
    import json, importlib.resources
    import jsonschema
    from solverpilot.bridges import TransformationTape
    m=Model(); z=m.binary(); x=m.integer(lower=0,upper=5); m.indicator(z,x<=2); m.minimize(x)
    c=m.compile(use_cache=False)
    raw=importlib.resources.files("solverpilot.bridges").joinpath("transformation-tape.schema.json").read_text()
    schema=json.loads(raw)
    jsonschema.validate({"schema_version":"solverpilot.transformation-tape.p4.v1","steps":list(c.transformation_tape)},schema)


def test_indicator_variable_may_appear_in_affine_body_and_lowering_remains_exact():
    m=Model(); z=m.binary(); x=m.integer(lower=-2,upper=3)
    m.indicator(z, x + 2*z <= 1, active_value=1); m.minimize(x)
    c=m.compile(use_cache=False)
    for zv in (0,1):
        for xv in range(-2,4):
            sem=(zv!=1) or (xv+2*zv<=1)
            assert _linear_feasible(c,[zv,xv])==sem


def test_indicator_bridge_preserves_maximization_objective():
    m=Model(); z=m.binary(); x=m.integer(lower=0,upper=5)
    m.indicator(z,x<=2); m.add(z>=1); m.maximize(x)
    c=m.compile(use_cache=False); r=execute(c.execution_ir,ScipyHighsBackend())
    assert r.x is not None and r.objective==pytest.approx(2.0,abs=1e-7)
    assert c.validate_original(m,r.x).valid
