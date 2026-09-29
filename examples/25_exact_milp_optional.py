"""Run with explicit SOLVERPILOT_EXACT_SCIP and SOLVERPILOT_VIPR paths."""
import os

from solverpilot import LinearProblem
from solverpilot.exact import solve_exact


def main():
    scip = os.getenv("SOLVERPILOT_EXACT_SCIP")
    checker = os.getenv("SOLVERPILOT_VIPR")
    if not scip or not checker:
        print("Exact MILP example skipped: configure exact SCIP and VIPR executable paths.")
        return
    p = LinearProblem([[2, 2]], [1, 1], [0, 0], [2, 2], [3], [float("inf")],
                      ["integer", "integer"])
    result = solve_exact(p, scip_executable=scip, checker_executable=checker)
    assert result.status == "optimal" and result.independently_verified, result.reason
    assert result.objective == 2 and result.absolute_gap == 0
    print("Exact optimum:", result.objective, "certificate:", result.certificate_sha256)


if __name__ == "__main__":
    main()
