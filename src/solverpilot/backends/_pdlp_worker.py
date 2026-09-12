"""Executed by file path, intentionally without importing SolverPilot/HiGHS."""

import json
import sys
import numpy as np
from scipy import sparse
from ortools.pdlp.python import pdlp
from ortools.pdlp import solvers_pb2, solve_log_pb2


def main():
    config = json.loads(sys.stdin.read())
    if config.get("protocol") != "solverpilot-pdlp-v1":
        raise ValueError("invalid protocol")
    with np.load(sys.argv[1], allow_pickle=False) as data:
        m, n = (int(v) for v in data["A_shape"])
        qp = pdlp.QuadraticProgram()
        qp.resize_and_initialize(n, m)
        qp.constraint_matrix = sparse.csc_matrix(
            sparse.csr_matrix((data["A_data"], data["A_indices"], data["A_indptr"]), shape=(m, n))
        )
        qp.variable_lower_bounds = data["lower"]
        qp.variable_upper_bounds = data["upper"]
        qp.constraint_lower_bounds = data["row_lower"]
        qp.constraint_upper_bounds = data["row_upper"]
        qp.objective_vector = data["q"]
        if np.any(data["diagonal"]):
            qp.set_objective_matrix_diagonal(data["diagonal"])
    params = solvers_pb2.PrimalDualHybridGradientParams()
    params.num_threads = config["threads"]
    params.termination_criteria.eps_optimal_absolute = config["absolute_tolerance"]
    params.termination_criteria.eps_optimal_relative = config["relative_tolerance"]
    params.termination_criteria.iteration_limit = config["iteration_limit"]
    if config["time_limit_s"] is not None:
        params.termination_criteria.time_sec_limit = config["time_limit_s"]
    result = pdlp.primal_dual_hybrid_gradient(qp, params)

    def finite_vector(value):
        a = np.asarray(value)
        return a.tolist() if np.isfinite(a).all() else None

    bound = None
    for info in result.solve_log.solution_stats.convergence_information:
        if info.candidate_type == result.solve_log.solution_type:
            value = info.corrected_dual_objective
            bound = float(value) if np.isfinite(value) else None
    payload = {
        "protocol": config["protocol"],
        "termination": solve_log_pb2.TerminationReason.Name(result.solve_log.termination_reason),
        "iterations": result.solve_log.iteration_count,
        "x": finite_vector(result.primal_solution),
        "dual": finite_vector(result.dual_solution),
        "reduced_costs": finite_vector(result.reduced_costs),
        "corrected_dual_objective": bound,
    }
    print("SOLVERPILOT_PDLP_RESULT=" + json.dumps(payload, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
