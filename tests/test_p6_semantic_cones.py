from __future__ import annotations

import numpy as np
import pytest

import solverpilot as om
from solverpilot.model.errors import CompileError


def test_soc_model_compiles_to_conic_ir_and_source_map():
    m=om.Model(); x=m.variable(2); t=m.variable(lower=0); c=m.soc(t,x,name='soc'); m.minimize(t)
    compiled=m.compile(use_cache=False)
    assert isinstance(compiled.execution_ir, om.ConicProblem)
    assert compiled.execution_ir.cones[0].kind is om.ConeKind.SECOND_ORDER
    assert compiled.source_map['constraints'][c.entity_id.value]['cone_block'] == 0
    assert compiled.schema_version == 'solverpilot.compiled-model.p6.v1'


def test_rotated_soc_model_compiles():
    m=om.Model(); z=m.variable(3); u=m.variable(lower=0); v=m.variable(lower=0); m.rotated_soc(u,v,z); m.minimize(u+v)
    p=m.compile(use_cache=False).execution_ir
    assert p.cones[0].kind is om.ConeKind.ROTATED_SECOND_ORDER
    assert p.cones[0].output_shape == (5,)


def test_psd_model_requires_structural_symmetry():
    m=om.Model(); X=m.variable((2,2)); m.psd(X); m.minimize(X.sum())
    with pytest.raises(CompileError, match='not structurally symmetric'):
        m.compile(use_cache=False)


def test_psd_symmetric_expression_compiles():
    m=om.Model(); t=m.variable(lower=0); M=t*m.constant(np.eye(2))+m.constant([[0.,1.],[1.,0.]])
    m.psd(M); m.minimize(t)
    p=m.compile(use_cache=False).execution_ir
    assert p.cones[0].kind is om.ConeKind.POSITIVE_SEMIDEFINITE


def test_conic_integer_variables_fail_closed():
    m=om.Model(); x=m.integer(1,lower=0,upper=2); t=m.variable(lower=0); m.soc(t,x); m.minimize(t)
    with pytest.raises(CompileError, match='continuous variables only'):
        m.compile(use_cache=False)


def test_indicator_plus_conic_fails_closed():
    m=om.Model(); z=m.binary(); x=m.variable(1,lower=-1,upper=1); t=m.variable(lower=0); m.soc(t,x); m.indicator(z,x[0] <= 0.5); m.minimize(t)
    with pytest.raises(CompileError, match='does not combine indicator'):
        m.compile(use_cache=False)


def test_parameter_update_relowers_and_snapshot_revisit():
    m=om.Model(); p=m.parameter(value=1.0); x=m.variable(1); t=m.variable(lower=0); m.soc(t,p*x); m.minimize(t)
    c1=m.compile(); sh=c1.semantic_hash; ch=c1.compilation_hash; dh1=c1.data_hash
    p.value=2.; c2=m.compile(); dh2=c2.data_hash
    assert c2.compilation_report.cache_status == 'full_conic_relower'
    assert c2.semantic_hash==sh and c2.compilation_hash==ch and dh2!=dh1
    p.value=1.; c3=m.compile()
    assert c3.compilation_report.cache_status=='snapshot_hit'
    assert c3.data_hash==dh1


def test_clone_preserves_cone_semantics():
    m=om.Model(); x=m.variable(2); t=m.variable(lower=0); m.soc(t,x); m.minimize(t)
    clone=m.clone()
    assert clone.semantic_hash == m.semantic_hash or clone._semantic_payload(include_parameter_values=False)['constraints'][0]['set']['kind']=='SecondOrderCone'
    assert isinstance(clone.compile(use_cache=False).execution_ir, om.ConicProblem)


def test_original_space_validator_understands_cones():
    m=om.Model(); x=m.variable(2); t=m.variable(lower=0); m.soc(t,x); m.minimize(t)
    c=m.compile(use_cache=False)
    assert c.validate_original(m,np.array([3.,4.,5.]),atol=1e-8).valid
    assert not c.validate_original(m,np.array([3.,4.,4.]),atol=1e-8).valid
