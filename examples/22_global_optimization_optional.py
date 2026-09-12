"""Explicit bounded global MINLP, equality and indicator composition."""

from solverpilot.model import Model
from solverpilot.globalopt import SCIPGlobalBackend, absolute, solve_global


def main():
    backend = SCIPGlobalBackend(time_limit_s=20)
    if not backend.is_available():
        print("Optional example: install solverpilot[global]")
        return
    m = Model("bounded global example")
    x = m.variable(lower=-2, upper=2)
    z = m.variable(lower=0, upper=4, domain="integer")
    enabled = m.variable(lower=0, upper=1, domain="binary")
    m.indicator(enabled, x * x <= 1)
    m.minimize((x * x - 1) ** 2 + 0.1 * x + (z - 2.3) ** 2 + 0.01 * absolute(x))
    result = solve_global(m, backend=backend)
    if not result.validation.valid:
        raise RuntimeError("invalid candidate")
    print(result.objective, result.raw_statistics["solver_dual_bound"])
    print("SCIP numerical global termination:", result.backend_status)
    print("Independent global proof:", result.optimality_evidence.independently_verified_optimal)


if __name__ == "__main__":
    main()
