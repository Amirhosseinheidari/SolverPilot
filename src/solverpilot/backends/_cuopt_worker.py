"""Execute by file path; keep the CUDA runtime outside the caller's process."""

import json
import sys
from time import perf_counter


def main():
    start = perf_counter()
    config = json.loads(sys.stdin.read())
    if config.get("protocol") != "solverpilot-cuopt-v1":
        raise ValueError("invalid protocol")
    import numpy as np
    import cupy
    from cuopt import linear_programming as lp
    from cuopt.linear_programming.solver_settings import SolverMethod

    imported = perf_counter()
    count = cupy.cuda.runtime.getDeviceCount()
    if count < 1:
        raise RuntimeError("no CUDA GPU available")
    device = cupy.cuda.runtime.getDeviceProperties(0)
    model = lp.DataModel()
    with np.load(sys.argv[1], allow_pickle=False) as data:
        original_rows = len(data["row_lower"])
        if original_rows:
            model.set_csr_constraint_matrix(data["A_data"], data["A_indices"], data["A_indptr"])
            model.set_constraint_lower_bounds(data["row_lower"])
            model.set_constraint_upper_bounds(data["row_upper"])
        else:
            # cuOpt rejects a zero-row CSR. Repeat an existing variable bound
            # as one redundant row, preserving the original feasible set.
            model.set_csr_constraint_matrix(
                np.array([1.0]), np.array([0], dtype=np.int32), np.array([0, 1], dtype=np.int32)
            )
            model.set_constraint_lower_bounds(data["lower"][:1])
            model.set_constraint_upper_bounds(data["upper"][:1])
        model.set_objective_coefficients(data["q"])
        model.set_variable_lower_bounds(data["lower"])
        model.set_variable_upper_bounds(data["upper"])
    settings = lp.SolverSettings()
    # Force GPU PDLP; disable CPU presolve/crossover and concurrent simplex.
    settings.set_parameter("method", SolverMethod.PDLP)
    settings.set_parameter("pdlp_precision", 1)  # Explicit FP64, independent of native defaults.
    settings.set_parameter("presolve", 0)
    settings.set_parameter("crossover", False)
    settings.set_parameter("log_to_console", False)
    settings.set_parameter("num_cpu_threads", config["threads"])
    settings.set_parameter("iteration_limit", config["iteration_limit"])
    settings.set_parameter("infeasibility_detection", True)
    settings.set_optimality_tolerance(config["tolerance"])
    remaining = config["time_limit_s"] - (perf_counter() - start)
    if remaining <= 0:
        raise TimeoutError("cuOpt worker initialization exhausted time limit")
    settings.set_parameter("time_limit", remaining)
    built = perf_counter()
    result = lp.Solve(model, settings)
    finished = perf_counter()
    if int(result.get_error_status()) != 0:
        raise RuntimeError(result.get_error_message())

    def vector(value):
        if value is None:
            return None
        a = np.asarray(value, dtype=float)
        return a.tolist() if np.isfinite(a).all() else None

    def finite(value):
        return float(value) if value is not None and np.isfinite(value) else None

    payload = {
        "protocol": config["protocol"],
        "termination": result.get_termination_reason(),
        "solved_by": result.get_solved_by().name,
        "gpu_count": count,
        "gpu_device": device["name"].decode()
        if isinstance(device["name"], bytes)
        else device["name"],
        "x": vector(result.get_primal_solution()),
        "dual": (
            None
            if result.get_dual_solution() is None
            else vector(result.get_dual_solution()[:original_rows])
        ),
        "reduced_costs": vector(result.get_reduced_cost()),
        "objective": finite(result.get_primal_objective()),
        "native_solve_s": finite(result.get_solve_time()),
        "statistics": {k: finite(v) for k, v in result.get_lp_stats().items()},
        "timings": {
            "import_s": imported - start,
            "build_s": built - imported,
            "native_call_s": finished - built,
        },
    }
    print("SOLVERPILOT_CUOPT_RESULT=" + json.dumps(payload, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
