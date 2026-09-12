"""Optional first-order LP solver and bounded Farkas-witness recovery."""

import numpy as np
from solverpilot import LinearProblem, solve
from solverpilot.backends.pdlp import PDLPBackend
from solverpilot.validate import recover_lp_certificate, verify_lp_certificate


def main():
    infeasible = LinearProblem.from_data(
        A=[[1], [-1]],
        c=[0],
        variable_lower=[-np.inf],
        variable_upper=[np.inf],
        constraint_lower=[-np.inf, -np.inf],
        constraint_upper=[0, -1],
    )
    certificate = recover_lp_certificate(infeasible, termination="infeasible", time_limit_s=2)
    print("Independent Farkas witness:", verify_lp_certificate(infeasible, certificate))
    backend = PDLPBackend(time_limit_s=10)
    if not backend.is_available():
        print("Optional PDLP example: install solverpilot[pdlp]")
        return
    p = LinearProblem.from_data(
        A=[[1, 1]],
        c=[1, 2],
        variable_lower=[0, 0],
        variable_upper=[2, 2],
        constraint_lower=[1],
        constraint_upper=[np.inf],
    )
    r = solve(p, backend=backend)
    if not r.validation.valid:
        raise RuntimeError("invalid candidate")
    print(
        "PDLP objective:",
        r.objective,
        "independent:",
        r.optimality_evidence.independently_verified_optimal,
    )


if __name__ == "__main__":
    main()
