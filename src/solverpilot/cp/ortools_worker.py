from __future__ import annotations

import json
import sys

from solverpilot._version import __version__ as SOLVERPILOT_VERSION
from .ortools_backend import (
    CP_WORKER_PROTOCOL_VERSION,
    CP_WORKER_REQUEST_SCHEMA,
    CP_WORKER_RESULT_SCHEMA,
    ORToolsCPSATBackend,
    VERIFIED_ORTOOLS_VERSION,
)
from .serialization import problem_from_canonical_dict


def _require_request_text(request: dict, field: str) -> str:
    value = request.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f'invalid CP-SAT worker request field {field}')
    return value


def main() -> int:
    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict):
            raise ValueError('CP-SAT worker request must be a JSON object')
        if request.get('schema') != CP_WORKER_REQUEST_SCHEMA:
            raise ValueError('unknown CP-SAT worker request schema')
        if request.get('protocol_version') != CP_WORKER_PROTOCOL_VERSION:
            raise ValueError('CP-SAT worker protocol version mismatch')

        request_id = _require_request_text(request, 'request_id')
        requested_solverpilot = _require_request_text(request, 'solverpilot_version')
        requested_ortools = _require_request_text(request, 'ortools_version')
        expected_structural_hash = _require_request_text(request, 'problem_structural_hash')
        expected_data_hash = _require_request_text(request, 'problem_data_hash')
        if requested_solverpilot != SOLVERPILOT_VERSION:
            raise RuntimeError(f'SolverPilot worker version mismatch: request={requested_solverpilot} local={SOLVERPILOT_VERSION}')

        backend = ORToolsCPSATBackend(max_time_s=request.get('max_time_s'), num_workers=request.get('num_workers', 1))
        problem = problem_from_canonical_dict(request['problem'])
        if problem.structural_hash != expected_structural_hash:
            raise RuntimeError('CP-SAT worker structural hash mismatch')
        if problem.data_hash != expected_data_hash:
            raise RuntimeError('CP-SAT worker data hash mismatch')
        backend._require()
        if backend.binding_version != VERIFIED_ORTOOLS_VERSION or requested_ortools != backend.binding_version:
            raise RuntimeError(f'OR-Tools worker version mismatch: request={requested_ortools} local={backend.binding_version}')

        result = backend._solve_in_process(problem)
        payload = {
            'schema': CP_WORKER_RESULT_SCHEMA,
            'protocol_version': CP_WORKER_PROTOCOL_VERSION,
            'request_id': request_id,
            'solverpilot_version': SOLVERPILOT_VERSION,
            'ortools_version': backend.binding_version,
            'problem_structural_hash': problem.structural_hash,
            'problem_data_hash': problem.data_hash,
            'status': result.status,
            'assignment': None if result.assignment is None else {str(k): int(v) for k, v in result.assignment.items()},
            'objective': result.objective,
            'optimality_proven': bool(result.optimality_proven),
            'raw_statistics': dict(result.raw_statistics),
        }
        sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(',', ':')))
        return 0
    except Exception as exc:
        sys.stderr.write(f'{type(exc).__name__}: {exc}\n')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
