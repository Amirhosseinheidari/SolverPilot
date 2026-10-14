"""Internal cold HiGHS worker; fallback is disabled to prevent recursion."""
import json
import sys
from time import monotonic

from solverpilot.backends._highs_thread_transport import PROTOCOL, PREFIX, read_problem
from solverpilot.backends.highspy_native import HighspyNativeBackend
from solverpilot.backends.base import BackendSolveResult
from solverpilot.runtime.manifest import json_value


def main():
    config = json.loads(sys.stdin.read())
    if config['protocol'] != PROTOCOL:
        raise ValueError('unsupported HiGHS worker protocol')
    problem = read_problem(sys.argv[1])
    if problem.data_hash != config['problem_data_hash']:
        raise ValueError('isolated model identity mismatch')
    settings = dict(config['settings'])
    deadline = config['deadline_s']
    remaining = None if deadline is None else deadline-monotonic()
    if remaining is not None and remaining <= 0:
        result = BackendSolveResult('limit_no_solution', None)
    else:
        settings['time_limit_s'] = remaining
        with HighspyNativeBackend(**settings, reuse=False) as backend:
            backend._allow_thread_isolation = False
            result = backend.solve(problem)
    payload = dict(protocol=PROTOCOL, problem_data_hash=problem.data_hash, status=result.backend_status,
                   x=result.x, objective=result.objective_reported, statistics=result.raw_statistics or {})
    print(PREFIX + json.dumps(json_value(payload), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
