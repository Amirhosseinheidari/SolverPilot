from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from solverpilot import LinearProblem
from solverpilot.backends import ScipyVendoredHighsDevBackend


def main() -> None:
    backend = ScipyVendoredHighsDevBackend()
    if not backend.is_available():
        raise SystemExit("SciPy vendored HiGHS unavailable")
    passed = 0
    rows = []
    for seed in range(50):
        rng = np.random.default_rng(seed)
        n = 3
        j = seed % n
        A = np.zeros((3, n), dtype=float)
        A[0, j] = 1.0
        A[1, j] = 1.0
        A[2] = rng.normal(size=n)
        p = LinearProblem.from_data(
            A=A,
            c=np.zeros(n),
            variable_lower=np.full(n, -np.inf),
            variable_upper=np.full(n, np.inf),
            constraint_lower=[2.0, -np.inf, -100.0],
            constraint_upper=[np.inf, 1.0, 100.0],
        )
        iis = backend.compute_iis(p)
        ok = bool(iis.valid and 0 in iis.row_indices and 1 in iis.row_indices)
        passed += int(ok)
        rows.append({"seed": seed, "valid": iis.valid, "row_indices": list(iis.row_indices), "passed": ok})
    out = {"cases": 50, "passed": passed, "failed": 50-passed, "rows": rows}
    Path("benchmarks/results/m6-native-iis-stress.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k:v for k,v in out.items() if k!='rows'}, indent=2))
    if passed != 50:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
