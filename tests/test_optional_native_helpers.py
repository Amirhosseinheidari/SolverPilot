import numpy as np
from scipy import sparse

from solverpilot import LinearProblem, QuadraticProblem
from solverpilot.backends.highspy_native import _highs_status_from_strings, _linear_csc, _lower_hessian_csc
from solverpilot.backends.osqp_native import OSQPNativeBackend, _osqp_data, _osqp_status


def test_osqp_translation_appends_variable_bounds():
    p = QuadraticProblem.from_data(
        P=[[2.0, 0.0], [0.0, 4.0]],
        A=[[1.0, 2.0]],
        q=[-1.0, 3.0],
        variable_lower=[0.0, -2.0],
        variable_upper=[5.0, 2.0],
        constraint_lower=[-1.0],
        constraint_upper=[7.0],
    )
    d = _osqp_data(p)
    assert d.P.shape == (2, 2)
    assert d.A.shape == (3, 2)
    assert np.allclose(d.l, [-1.0, 0.0, -2.0])
    assert np.allclose(d.u, [7.0, 5.0, 2.0])
    assert np.allclose(d.q, [-1.0, 3.0])


def test_osqp_status_normalization_is_conservative():
    assert _osqp_status("solved", True) == "optimal"
    assert _osqp_status("solved inaccurate", True) == "converged_candidate"
    assert _osqp_status("primal infeasible", False) == "infeasible"
    assert _osqp_status("dual infeasible", False) == "unbounded"
    assert _osqp_status("primal infeasible inaccurate", False) == "infeasible_candidate"
    assert _osqp_status("dual infeasible inaccurate", False) == "unbounded_candidate"
    assert _osqp_status("maximum iterations reached", True) == "limit_feasible"
    assert _osqp_status("maximum iterations reached", False) == "limit_no_solution"


def test_highs_status_normalization():
    assert _highs_status_from_strings("Optimal", "Feasible") == "optimal"
    assert _highs_status_from_strings("Infeasible", "None") == "infeasible"
    assert _highs_status_from_strings("Unbounded", "None") == "unbounded"
    assert _highs_status_from_strings("Time limit reached", "Feasible") == "limit_feasible"
    assert _highs_status_from_strings("Time limit reached", "None") == "limit_no_solution"


def test_highs_sparse_helpers_have_expected_orientation():
    lp = LinearProblem.from_data(
        A=sparse.csr_matrix([[1.0, 0.0], [2.0, 3.0]]),
        c=[1.0, 2.0],
        variable_lower=[0.0, 0.0],
        variable_upper=[1.0, 1.0],
        constraint_lower=[0.0, 0.0],
        constraint_upper=[2.0, 5.0],
    )
    A = _linear_csc(lp)
    assert sparse.isspmatrix_csc(A)
    assert np.allclose(A.toarray(), lp.A.toarray())

    qp = QuadraticProblem.from_data(
        P=[[2.0, 1.0], [1.0, 3.0]],
        A=np.zeros((0, 2)),
        q=[0.0, 0.0],
        variable_lower=[-1.0, -1.0],
        variable_upper=[1.0, 1.0],
        constraint_lower=[],
        constraint_upper=[],
    )
    H = _lower_hessian_csc(qp)
    assert sparse.isspmatrix_csc(H)
    assert np.allclose(H.toarray(), [[2.0, 0.0], [1.0, 3.0]])


def test_optional_native_backends_are_import_safe_when_dependencies_absent():
    # This environment currently lacks both dependencies. The contract is that importing
    # the project must still work and availability is explicit rather than a hard import.
    assert isinstance(OSQPNativeBackend().is_available(), bool)


def test_scip_status_normalization():
    from solverpilot.backends.pyscipopt_native import _scip_status

    assert _scip_status("optimal", True) == "optimal"
    assert _scip_status("infeasible", False) == "infeasible"
    assert _scip_status("unbounded", False) == "unbounded"
    assert _scip_status("inforunbd", False) == "infeasible_or_unbounded"
    assert _scip_status("timelimit", True) == "limit_feasible"
    assert _scip_status("timelimit", False) == "limit_no_solution"


def test_highspy_model_builder_is_accepted_by_scipy_vendored_highs_lp():
    """Compatibility check only; not a substitute for public highspy integration."""
    from scipy.optimize._highspy import _core
    from solverpilot.backends.highspy_native import HighspyNativeBackend

    p = LinearProblem.from_data(
        A=[[1.0, 1.0]],
        c=[1.0, 2.0],
        variable_lower=[0.0, 0.0],
        variable_upper=[1.0, 1.0],
        constraint_lower=[1.0],
        constraint_upper=[np.inf],
    )
    lp = HighspyNativeBackend._make_lp_model(_core, p)
    h = _core._Highs()
    h.setOptionValue("output_flag", False)
    assert h.passModel(lp) != _core.HighsStatus.kError
    assert h.run() != _core.HighsStatus.kError
    assert h.getModelStatus() == _core.HighsModelStatus.kOptimal


def test_highspy_model_builder_is_accepted_by_scipy_vendored_highs_qp():
    """Checks the adapter's Hessian orientation against a real HiGHS binary."""
    from scipy.optimize._highspy import _core
    from solverpilot.backends.highspy_native import HighspyNativeBackend

    p = QuadraticProblem.from_data(
        P=[[2.0, 0.0], [0.0, 4.0]],
        A=np.zeros((0, 2)),
        q=[-2.0, -4.0],
        variable_lower=[-5.0, -5.0],
        variable_upper=[5.0, 5.0],
        constraint_lower=[],
        constraint_upper=[],
    )
    model = HighspyNativeBackend._make_qp_model(_core, p)
    h = _core._Highs()
    h.setOptionValue("output_flag", False)
    assert h.passModel(model) != _core.HighsStatus.kError
    assert h.run() != _core.HighsStatus.kError
    assert h.getModelStatus() == _core.HighsModelStatus.kOptimal
    assert np.allclose(h.getSolution().col_value, [1.0, 1.0], atol=1e-5)
