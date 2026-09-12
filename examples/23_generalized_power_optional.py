"""Generalized power cone with a two-dimensional Euclidean tail."""

from solverpilot.model import Model, GeneralizedPowerCone
from solverpilot.conic import ClarabelBackend


def main():
    backend = ClarabelBackend()
    if not backend.is_available():
        print("Optional example: install solverpilot[clarabel]")
        return
    m = Model()
    x = m.variable(5, lower=[2, 3, 5, 0, 0], upper=[2, 3, 5, 10, 0])
    m.add_in_set(x, GeneralizedPowerCone((0.2, 0.3, 0.5), tail_dimension=2))
    m.minimize(-x[3])
    r = m.solve(backend=backend)
    if not r.validation.valid:
        raise RuntimeError("invalid candidate")
    print("Power radius:", -r.objective_reported)
    print("Independent optimality:", r.optimality_evidence.independently_verified_optimal)


if __name__ == "__main__":
    main()
