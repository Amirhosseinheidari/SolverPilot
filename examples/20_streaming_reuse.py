"""Bounded input and process reuse across two batches."""

import numpy as np
from solverpilot import LinearProblem
from solverpilot.runtime.batch import BatchExecutor


def main():
    problem = LinearProblem.from_data(
        A=[[1.0]],
        c=[1.0],
        variable_lower=[0.0],
        variable_upper=[2.0],
        constraint_lower=[1.0],
        constraint_upper=[np.inf],
    )
    with BatchExecutor(
        backend="scipy-highs-ds", max_workers=1, max_pending=2, timeout_s=20
    ) as executor:
        for batch in range(2):
            results = list(executor.iter(problem for _ in range(3)))
            assert all(
                item.validation_valid and abs(item.objective - 1.0) < 1e-7 for item in results
            )
            print(batch, [item.objective for item in results])


if __name__ == "__main__":
    main()
