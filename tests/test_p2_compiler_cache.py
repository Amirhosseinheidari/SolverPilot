from __future__ import annotations

import numpy as np
import pytest

from solverpilot.model import Model
from solverpilot.problem import LinearProblem, QuadraticProblem


def _assert_linear_equal(a: LinearProblem, b: LinearProblem) -> None:
    np.testing.assert_allclose(a.A.toarray(), b.A.toarray(), atol=1e-12)
    np.testing.assert_allclose(a.c, b.c, atol=1e-12)
    np.testing.assert_allclose(a.variable_lower, b.variable_lower, atol=0)
    np.testing.assert_allclose(a.variable_upper, b.variable_upper, atol=0)
    np.testing.assert_allclose(a.constraint_lower, b.constraint_lower, atol=1e-12)
    np.testing.assert_allclose(a.constraint_upper, b.constraint_upper, atol=1e-12)
    assert np.array_equal(a.domains, b.domains)
    assert a.objective_sense == b.objective_sense
    assert a.objective_offset == pytest.approx(b.objective_offset)


def _assert_problem_equal(a, b) -> None:
    if isinstance(a, QuadraticProblem):
        assert isinstance(b, QuadraticProblem)
        _assert_linear_equal(a.linear, b.linear)
        np.testing.assert_allclose(a.P.toarray(), b.P.toarray(), atol=1e-12)
    else:
        assert isinstance(a, LinearProblem) and isinstance(b, LinearProblem)
        _assert_linear_equal(a, b)


def test_second_compile_is_snapshot_hit_and_reuses_execution_ir():
    m = Model(); x = m.variable(); p = m.parameter(value=2.0)
    m.add(x >= p); m.minimize(x)
    first = m.compile()
    second = m.compile()
    assert first.compilation_report.cache_status == "miss"
    assert second.compilation_report.cache_status == "snapshot_hit"
    assert second.execution_ir is first.execution_ir
    assert second.semantic_hash == first.semantic_hash
    assert second.data_hash == first.data_hash
    assert second.compilation_hash == first.compilation_hash
    info = m.compiler_cache_info
    assert info.full_compiles == 1
    assert info.partial_compiles == 0
    assert info.snapshot_hits == 1


def test_parameter_update_partial_recompile_preserves_semantic_and_compilation_hash():
    m = Model(); x = m.variable(2, lower=0)
    rhs = m.parameter(value=1.0); q = m.parameter(2, value=[1.0, 2.0])
    m.add(x[0] + x[1] >= rhs); m.minimize(q @ x)
    a = m.compile()
    sem = a.semantic_hash; plan = a.compilation_hash; data = a.data_hash
    rhs.value = 3.0
    b = m.compile()
    assert b.compilation_report.cache_status == "partial_recompile"
    assert b.compilation_report.changed_parameters == (rhs.id.value,)
    assert b.semantic_hash == sem
    assert b.compilation_hash == plan
    assert b.data_hash != data
    assert b.compilation_report.recompiled_execution_rows == 1
    assert b.compilation_report.objective_recompiled is False
    assert "constraint_bounds" in b.compilation_report.execution_mutations


def test_dependency_graph_recompiles_only_affected_constraint_rows():
    m=Model(); x=m.variable(2)
    p1=m.parameter(value=1.0); p2=m.parameter(value=2.0)
    c1=m.add(x[0] <= p1); c2=m.add(x[1] <= p2); m.minimize(x[0]+x[1])
    first=m.compile(); p1.value=4.0; second=m.compile()
    assert second.compilation_report.recompiled_execution_rows == 1
    assert second.compilation_report.reused_execution_rows == 1
    assert second.parameter_map[p1.id.value]["constraint_ids"] == [c1.entity_id.value]
    assert second.parameter_map[p2.id.value]["constraint_ids"] == [c2.entity_id.value]
    assert second.parameter_map[p1.id.value]["objective"] is False
    np.testing.assert_allclose(second.execution_ir.A.toarray(), [[-1.0,0.0],[0.0,-1.0]])
    np.testing.assert_allclose(second.execution_ir.constraint_lower, [-4.0,-2.0])
    assert np.isposinf(second.execution_ir.constraint_upper).all()


def test_objective_only_parameter_update_reuses_all_rows():
    m=Model(); x=m.variable(3, lower=0); q=m.parameter(3, value=[1,2,3])
    m.add(np.ones(3) @ x >= 1); m.minimize(q @ x)
    m.compile(); q.value=[4,5,6]; c=m.compile()
    assert c.compilation_report.recompiled_execution_rows == 0
    assert c.compilation_report.reused_execution_rows == 1
    assert c.compilation_report.objective_recompiled is True
    assert "objective_vector" in c.compilation_report.execution_mutations
    np.testing.assert_allclose(c.execution_ir.c,[4,5,6])


def test_parameterized_matrix_zero_crossing_is_classified_as_sparsity_change():
    m=Model(); x=m.variable(); a=m.parameter(value=1.0)
    m.add(a*x <= 2); m.minimize(x)
    first=m.compile(); assert first.execution_ir.A.nnz == 1
    a.value=0.0; second=m.compile()
    assert second.execution_ir.A.nnz == 0
    assert "matrix_sparsity_changed" in second.compilation_report.execution_mutations
    a.value=3.0; third=m.compile()
    assert third.execution_ir.A.nnz == 1
    assert "matrix_sparsity_changed" in third.compilation_report.execution_mutations


def test_revisiting_old_parameter_snapshot_hits_execution_snapshot_cache():
    m=Model(); x=m.variable(); p=m.parameter(value=1.0)
    m.add(x >= p); m.minimize(x)
    a=m.compile(); p.value=2.0; b=m.compile(); p.value=1.0; c=m.compile()
    assert b.compilation_report.cache_status == "partial_recompile"
    assert c.compilation_report.cache_status == "snapshot_hit"
    assert c.execution_ir.data_hash == a.execution_ir.data_hash
    assert m.compiler_cache_info.snapshots == 2
    assert m.compiler_cache_info.snapshot_hits == 1


def test_unused_parameter_changes_data_hash_but_not_execution_ir_semantics():
    m=Model(); x=m.variable(lower=0); unused=m.parameter(value=1.0)
    m.minimize(x)
    a=m.compile(); unused.value=9.0; b=m.compile()
    assert b.data_hash != a.data_hash
    assert b.semantic_hash == a.semantic_hash
    assert b.compilation_hash == a.compilation_hash
    assert b.compilation_report.recompiled_execution_rows == 0
    assert b.compilation_report.objective_recompiled is False
    assert b.compilation_report.execution_mutations == ()
    _assert_problem_equal(a.execution_ir,b.execution_ir)


def test_structural_degree_controls_qp_target_even_when_parameter_makes_quadratic_zero():
    m=Model(); x=m.variable(lower=-2, upper=2); p=m.parameter(value=0.0)
    m.minimize(0.5*p*x*x + x)
    a=m.compile()
    assert isinstance(a.execution_ir, QuadraticProblem)
    assert a.execution_ir.P.nnz == 0
    plan=a.compilation_hash
    p.value=2.0; b=m.compile()
    assert isinstance(b.execution_ir, QuadraticProblem)
    assert b.compilation_hash == plan
    np.testing.assert_allclose(b.execution_ir.P.toarray(), [[2.0]])


def test_structurally_quadratic_constraint_routes_to_p7_even_if_current_parameter_is_zero():
    from solverpilot.nlp import NLPProblem
    m=Model(); x=m.variable(); p=m.parameter(value=0.0)
    m.add(p*x*x <= 1); m.minimize(x)
    compiled=m.compile()
    assert isinstance(compiled.execution_ir,NLPProblem)
    assert compiled.schema_version == "solverpilot.compiled-model.p7.v1"


def test_qp_parameter_update_revalidates_convexity_and_fail_closed_on_nonconvex_snapshot():
    m=Model(); x=m.variable(); p=m.parameter(value=1.0)
    m.minimize(0.5*p*x*x)
    good=m.compile(); assert isinstance(good.execution_ir,QuadraticProblem)
    p.value=-1.0
    with pytest.raises(ValueError, match="nonconvex|positive semidefinite|not positive"):
        m.compile()
    p.value=1.0
    recovered=m.compile()
    assert recovered.compilation_report.cache_status == "snapshot_hit"
    _assert_problem_equal(recovered.execution_ir,good.execution_ir)


def test_cache_disabled_always_full_compiles_without_populating_model_cache():
    m=Model(); x=m.variable(); p=m.parameter(value=1); m.add(x>=p); m.minimize(x)
    a=m.compile(use_cache=False); p.value=2; b=m.compile(use_cache=False)
    assert a.compilation_report.cache_status == "disabled"
    assert b.compilation_report.cache_status == "disabled"
    assert m.compiler_cache_info.full_compiles == 0
    assert m.compiler_cache_info.snapshots == 0


def test_semantic_change_invalidates_template_and_forces_miss():
    m=Model(); x=m.variable(); m.minimize(x); first=m.compile()
    y=m.variable(); m.minimize(x+y); second=m.compile()
    assert first.semantic_hash != second.semantic_hash
    assert second.compilation_report.cache_status == "miss"
    assert second.execution_ir.n_variables == 2


def test_clear_compile_cache_forces_new_full_compile():
    m=Model(); x=m.variable(); m.minimize(x); m.compile(); m.compile()
    assert m.compiler_cache_info.snapshot_hits == 1
    m.clear_compile_cache(); assert m.compiler_cache_info.snapshots == 0
    c=m.compile(); assert c.compilation_report.cache_status == "miss"


def test_compiled_refresh_uses_model_cache():
    m=Model(); x=m.variable(); p=m.parameter(value=1); m.add(x>=p); m.minimize(x)
    c=m.compile(); p.value=3; d=c.refresh(m)
    assert d.compilation_report.cache_status == "partial_recompile"
    np.testing.assert_allclose(d.execution_ir.constraint_lower,[3])


def test_parameter_same_value_is_noop_and_snapshot_hit():
    m=Model(); x=m.variable(); p=m.parameter(value=1.0); m.add(x>=p); m.minimize(x)
    a=m.compile(); p.value=1.0; b=m.compile()
    assert a.data_hash == b.data_hash
    assert b.compilation_report.cache_status == "snapshot_hit"
    assert b.compilation_report.changed_parameters == ()


def test_multiple_parameter_update_union_of_dependencies():
    m=Model(); x=m.variable(2); a=m.parameter(value=1); b=m.parameter(value=2); q=m.parameter(2,value=[1,1])
    m.add(x[0] <= a); m.add(x[1] <= b); m.minimize(q@x)
    m.compile(); a.value=3; b.value=4; q.value=[2,5]; c=m.compile()
    assert set(c.compilation_report.changed_parameters)=={a.id.value,b.id.value,q.id.value}
    assert c.compilation_report.recompiled_execution_rows==2
    assert c.compilation_report.objective_recompiled is True
    np.testing.assert_allclose(c.execution_ir.constraint_lower,[-3,-4]); assert np.isposinf(c.execution_ir.constraint_upper).all(); np.testing.assert_allclose(c.execution_ir.c,[2,5])


def test_execution_snapshot_cache_is_lru_bounded():
    m=Model(); x=m.variable(); p=m.parameter(value=0.0); m.add(x>=p); m.minimize(x)
    m.compile()
    for value in range(1,13):
        p.value=float(value); m.compile()
    assert m.compiler_cache_info.snapshots == 8
    hits_before=m.compiler_cache_info.snapshot_hits
    p.value=0.0; c=m.compile()
    assert c.compilation_report.cache_status == "partial_recompile"
    assert m.compiler_cache_info.snapshot_hits == hits_before
