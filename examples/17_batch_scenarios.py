"""Independent production-demand scenarios with bounded process concurrency."""
import numpy as np
from solverpilot import LinearProblem
from solverpilot.runtime.batch import solve_batch


def main():
    problems = [LinearProblem.from_data(A=[[1.]], c=[2.], variable_lower=[0.],
                variable_upper=[10.], constraint_lower=[demand], constraint_upper=[np.inf])
                for demand in (1., 3., 5.)]
    results = solve_batch(problems, backend='scipy-highs-ds', max_workers=2, timeout_s=30)
    assert all(r.validation_valid for r in results)
    assert np.allclose([r.objective for r in results], [2., 6., 10.])
    print([(r.index, r.status, r.objective) for r in results])


if __name__ == '__main__':
    main()
