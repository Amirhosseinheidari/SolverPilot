from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

import solverpilot as om
from solverpilot.model import CompileError, DomainError, Model, OwnershipError, ShapeError, SignDomain, SymbolicTruthValueError
from solverpilot.problem import LinearProblem, QuadraticProblem, VariableDomain
from solverpilot.validate import PublicStatus
from solverpilot.backends import BundledHighsCAPIBackend, BundledOSQPCAPIBackend


def test_scalar_model_compiles_exactly_to_linear_problem():
    m = Model("lp")
    x = m.variable(2, lower=0, upper=[2, 3], name="x")
    m.add(x[0] + 2 * x[1] >= 1)
    m.minimize(np.array([3.0, 4.0]) @ x + 5)
    c = m.compile()
    p = c.execution_ir
    assert isinstance(p, LinearProblem)
    np.testing.assert_allclose(p.A.toarray(), [[1, 2]])
    np.testing.assert_allclose(p.c, [3, 4])
    np.testing.assert_allclose(p.constraint_lower, [1])
    assert np.isposinf(p.constraint_upper[0])
    assert p.objective_offset == 5
    assert c.source_map["variables"][x.id.value]["execution_slice"] == [0, 2]


def test_vector_relation_expands_rows():
    m = Model()
    x = m.variable(3)
    m.add(x <= [1, 2, 3])
    m.minimize(om.symbolic_sum(x))
    p = m.compile().execution_ir
    np.testing.assert_allclose(p.A.toarray(), np.eye(3))
    np.testing.assert_allclose(p.constraint_upper, [1, 2, 3])
    assert p.n_constraints == 3


def test_binary_and_integer_domains_preserved():
    m = Model()
    z = m.binary(2)
    y = m.integer(lower=0, upper=4)
    m.add(z[0] + z[1] + y >= 1)
    m.minimize(z[0] + 2*z[1] + y)
    p = m.compile().execution_ir
    assert isinstance(p, LinearProblem)
    assert list(p.domains) == ["binary", "binary", "integer"]


def test_parameter_update_preserves_semantic_hash_changes_data_hash_and_compile_result():
    m = Model()
    x = m.variable(2, lower=0)
    b = m.parameter(2, value=[1, 2], name="b")
    m.add(x >= b)
    m.minimize(np.array([1.0, 1.0]) @ x)
    sh1, dh1 = m.semantic_hash, m.data_hash
    p1 = m.compile().execution_ir
    b.value = [3, 4]
    sh2, dh2 = m.semantic_hash, m.data_hash
    p2 = m.compile().execution_ir
    assert sh1 == sh2
    assert dh1 != dh2
    np.testing.assert_allclose(p1.constraint_lower, [1, 2])
    np.testing.assert_allclose(p2.constraint_lower, [3, 4])


def test_parameter_shape_and_sign_validation():
    m = Model()
    p = m.parameter(2, value=[1, 2], sign=SignDomain.NONNEGATIVE)
    with pytest.raises(ShapeError):
        p.value = [1, 2, 3]
    with pytest.raises(DomainError):
        p.value = [-1, 2]
    with pytest.raises(DomainError):
        m.parameter(value=np.inf)


def test_expression_metadata_dependencies_and_degree():
    m = Model()
    x = m.variable(2, lower=0)
    p = m.parameter(2, value=[2, 3])
    e = (p @ x) + x[0] * x[1]
    assert e.shape == ()
    assert e.polynomial_degree == 2
    assert e.variable_dependencies == {x.id.value}
    assert e.parameter_dependencies == {p.id.value}
    assert x.sign is SignDomain.NONNEGATIVE


def test_symbolic_truth_value_fails():
    m = Model(); x = m.variable()
    with pytest.raises(SymbolicTruthValueError):
        bool(x <= 1)
    with pytest.raises(SymbolicTruthValueError):
        bool(x)


def test_cross_model_references_fail():
    m1, m2 = Model(), Model()
    x = m1.variable(); y = m2.variable()
    with pytest.raises(OwnershipError):
        _ = x + y
    with pytest.raises(OwnershipError):
        m1.minimize(y)


def test_shape_mismatch_fails():
    m = Model(); x = m.variable(2); y = m.variable(3)
    with pytest.raises(ShapeError):
        _ = x + y
    with pytest.raises(ShapeError):
        _ = x @ y


def test_numpy_left_matmul_and_sum_work():
    m = Model(); x = m.variable(3)
    e = np.array([1.0, 2.0, 3.0]) @ x
    assert e.shape == ()
    s = np.sum(x)
    assert s.shape == ()
    m.minimize(e + s)
    p = m.compile().execution_ir
    np.testing.assert_allclose(p.c, [2, 3, 4])


def test_quadratic_objective_compiles_with_legacy_half_xpx_convention():
    m = Model()
    x = m.variable(2, lower=-10, upper=10)
    m.minimize(x[0]*x[0] + 2*x[1]*x[1] - 2*x[0] - 8*x[1])
    compiled = m.compile()
    p = compiled.execution_ir
    assert isinstance(p, QuadraticProblem)
    np.testing.assert_allclose(p.P.toarray(), [[2, 0], [0, 4]])
    np.testing.assert_allclose(p.linear.c, [-2, -8])
    from solverpilot.backends import ScipySLSQPQPBackend
    r = compiled.solve(backend=ScipySLSQPQPBackend())
    assert r.status is PublicStatus.VALID_FEASIBLE
    np.testing.assert_allclose(r.x, [1, 2], atol=1e-7)
    assert r.objective == pytest.approx(-9.0, abs=1e-7)


def test_matrix_quadratic_expression():
    m = Model(); x = m.variable(2, lower=-10, upper=10)
    Q = np.array([[2.0, 0.5], [0.5, 4.0]])
    m.minimize(0.5 * (x @ Q @ x))
    p = m.compile().execution_ir
    assert isinstance(p, QuadraticProblem)
    np.testing.assert_allclose(p.P.toarray(), Q)


def test_nonconvex_quadratic_rejected_by_legacy_qp_boundary():
    m = Model(); x = m.variable(2)
    m.minimize(x[0] * x[0] - x[1] * x[1])
    with pytest.raises(ValueError, match="positive semidefinite|not positive"):
        m.compile()


def test_quadratic_constraint_routes_to_p7_nlp_after_scope_extension():
    from solverpilot.nlp import NLPProblem
    m = Model(); x = m.variable()
    m.add(x*x <= 1); m.minimize(x)
    compiled = m.compile()
    assert isinstance(compiled.execution_ir, NLPProblem)
    assert compiled.schema_version == "solverpilot.compiled-model.p7.v1"


def test_quadratic_integer_rejected_in_p1():
    m = Model(); x = m.integer(lower=0, upper=4)
    m.minimize(x*x)
    with pytest.raises(CompileError, match="binary discrete variables"):
        m.compile()


def test_model_without_objective_rejected():
    m = Model(); m.variable()
    with pytest.raises(CompileError, match="no objective"):
        m.compile()


def test_objective_must_be_scalar():
    m = Model(); x = m.variable(2)
    with pytest.raises(ShapeError):
        m.minimize(x)


def test_constant_nan_inf_rejected_but_variable_infinite_bounds_allowed():
    m = Model(); x = m.variable(lower=-np.inf, upper=np.inf)
    assert np.isneginf(x.lower)
    with pytest.raises(DomainError):
        _ = x + np.inf
    with pytest.raises(DomainError):
        _ = x + np.nan


def test_compiled_artifact_contract_fields_present():
    m = Model(); x = m.variable(); p = m.parameter(value=2.0)
    m.add(x >= p); m.minimize(x)
    c = m.compile()
    assert c.schema_version == "solverpilot.compiled-model.p2.v1"
    assert c.capability_signature == "legacy-ir:p2:v1"
    assert c.transformation_tape == ()
    assert c.parameter_map[p.id.value]["patchable"] is True
    assert c.compilation_report.n_execution_variables == 1
    assert c.compilation_hash


def test_clone_new_ids_and_same_semantic_shape():
    m = Model(); x = m.variable(2, lower=0); p = m.parameter(2, value=[1,2])
    m.add(x >= p); m.minimize(np.array([1.0,2.0]) @ x)
    clone = m.clone()
    assert clone is not m
    assert clone.variables[0].id.value == "v000001"
    assert clone.parameters[0].id.value == "p000001"
    assert clone.variables[0].id != x.id
    assert clone.parameters[0].id != p.id
    np.testing.assert_allclose(clone.compile().execution_ir.c, m.compile().execution_ir.c)
    p.value = [9,9]
    assert not np.array_equal(clone.parameters[0].value, p.value)


def test_semantic_hash_ignores_names_and_parameter_values_but_includes_fixed_literals():
    def build(name, pval, coef):
        m = Model(name)
        x = m.variable(name="different")
        p = m.parameter(value=pval, name="p")
        m.add(coef*x >= p, name="c")
        m.minimize(x, name="o")
        return m
    a = build("A", 1, 2)
    b = build("B", 7, 2)
    c = build("C", 1, 3)
    assert a.semantic_hash == b.semantic_hash
    assert a.data_hash != b.data_hash
    assert a.semantic_hash != c.semantic_hash


def test_semantic_hash_is_process_deterministic(tmp_path):
    code = '''\nfrom solverpilot.model import Model\nimport numpy as np\nm=Model(); x=m.variable(2,lower=0); p=m.parameter(2,value=[1,2]); m.add(np.array([1.,2.])@x>=p[0]); m.minimize(x[0]+3*x[1]); print(m.semantic_hash)\n'''
    env = dict(__import__('os').environ); env['PYTHONPATH'] = str(Path(__file__).parents[1]/'src')
    h1 = subprocess.check_output([sys.executable, "-c", code], env=env, text=True).strip()
    h2 = subprocess.check_output([sys.executable, "-c", code], env=env, text=True).strip()
    assert h1 == h2


def test_expression_nodes_are_immutable():
    m=Model(); x=m.variable()
    with pytest.raises(Exception):
        x._node.shape = (2,)


def test_direct_legacy_api_remains_available():
    p = LinearProblem.from_data(A=[[1]], c=[1], variable_lower=[0], variable_upper=[1], constraint_lower=[0], constraint_upper=[1])
    assert p.n_variables == 1
    assert om.LinearProblem is LinearProblem


def test_model_solve_convenience_linear():
    m=Model(); x=m.variable(lower=1,upper=3); m.minimize(x)
    from solverpilot.backends import ScipyHighsBackend
    r=m.solve(backend=ScipyHighsBackend())
    assert r.status is PublicStatus.VALID_OPTIMAL
    assert r.x[0] == pytest.approx(1)


def test_legacy_linear_roundtrip_migration_preserves_execution_data():
    legacy = LinearProblem.from_data(
        A=[[1,2],[-3,4]], c=[5,-2],
        variable_lower=[0,-1], variable_upper=[1,7],
        constraint_lower=[-np.inf,2], constraint_upper=[3,9],
        domains=["binary","integer"], objective_sense="maximize", objective_offset=1.5,
    )
    p = Model.from_problem(legacy).compile().execution_ir
    assert isinstance(p, LinearProblem)
    np.testing.assert_allclose(p.A.toarray(), legacy.A.toarray())
    np.testing.assert_allclose(p.c, legacy.c)
    np.testing.assert_allclose(p.variable_lower, legacy.variable_lower)
    np.testing.assert_allclose(p.variable_upper, legacy.variable_upper)
    np.testing.assert_allclose(p.constraint_lower, legacy.constraint_lower)
    np.testing.assert_allclose(p.constraint_upper, legacy.constraint_upper)
    assert list(p.domains) == list(legacy.domains)
    assert p.objective_sense == legacy.objective_sense
    assert p.objective_offset == legacy.objective_offset


def test_legacy_quadratic_roundtrip_migration_preserves_p_q_and_bounds():
    legacy = QuadraticProblem.from_data(
        P=[[2,0.5],[0.5,4]], A=[[1,1]], q=[-2,-8],
        variable_lower=[-10,-10], variable_upper=[10,10],
        constraint_lower=[-np.inf], constraint_upper=[5], objective_offset=3,
    )
    p = Model.from_problem(legacy).compile().execution_ir
    assert isinstance(p, QuadraticProblem)
    np.testing.assert_allclose(p.P.toarray(), legacy.P.toarray())
    np.testing.assert_allclose(p.linear.c, legacy.linear.c)
    np.testing.assert_allclose(p.linear.A.toarray(), legacy.linear.A.toarray())
    np.testing.assert_allclose(p.linear.constraint_lower, legacy.linear.constraint_lower)
    np.testing.assert_allclose(p.linear.constraint_upper, legacy.linear.constraint_upper)
    assert p.objective_offset == legacy.objective_offset


def test_top_level_symbolic_sum_alias():
    m=Model(); x=m.variable(2); m.minimize(om.sum(x)); p=m.compile().execution_ir
    np.testing.assert_allclose(p.c,[1,1])
