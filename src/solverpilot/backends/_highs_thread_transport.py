"""One-shot native execution when a process-global HiGHS scheduler conflicts."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic, perf_counter

import numpy as np
from scipy import sparse

from solverpilot.problem import LinearProblem, QuadraticProblem
from .base import BackendSolveResult

PROTOCOL = 'solverpilot-highs-thread-isolation.v1'
PREFIX = 'SOLVERPILOT_HIGHS_THREAD_RESULT='


def write_problem(path, problem):
    linear = problem.linear if isinstance(problem, QuadraticProblem) else problem
    values = dict(A_data=linear.A.data, A_indices=linear.A.indices, A_indptr=linear.A.indptr,
        A_shape=linear.A.shape, c=linear.c, variable_lower=linear.variable_lower,
        variable_upper=linear.variable_upper, constraint_lower=linear.constraint_lower,
        constraint_upper=linear.constraint_upper, domains=linear.domains,
        objective_sense=linear.objective_sense.value, objective_offset=linear.objective_offset,
        kind='qp' if isinstance(problem, QuadraticProblem) else 'lp')
    if isinstance(problem, QuadraticProblem):
        values.update(P_data=problem.P.data, P_indices=problem.P.indices, P_indptr=problem.P.indptr,
                      P_shape=problem.P.shape)
    np.savez(path, **values)


def read_problem(path):
    with np.load(path, allow_pickle=False) as data:
        def matrix(prefix):
            return sparse.csr_matrix((data[prefix+'_data'], data[prefix+'_indices'],
                                      data[prefix+'_indptr']), shape=tuple(data[prefix+'_shape']))
        values = {k: data[k] for k in ('variable_lower', 'variable_upper', 'constraint_lower', 'constraint_upper')}
        values.update(A=matrix('A'), objective_sense=data['objective_sense'].item(),
                      objective_offset=float(data['objective_offset']))
        if data['kind'].item() == 'qp':
            return QuadraticProblem.from_data(P=matrix('P'), q=data['c'], **values)
        if data['kind'].item() != 'lp':
            raise ValueError('unsupported isolated HiGHS model')
        return LinearProblem.from_data(c=data['c'], domains=data['domains'], **values)


def solve_isolated_threads(problem, backend, *, started):
    def limited(reason):
        return BackendSolveResult('limit_no_solution', None, raw_statistics={
            'reason': reason, 'reuse_applied': False, 'reuse_mode': 'isolated_cold',
            'reuse_reason': 'global_thread_configuration_conflict', 'isolated_thread_configuration': True})
    deadline = None if backend.time_limit_s is None else monotonic() + backend.time_limit_s - (perf_counter()-started)
    with TemporaryDirectory(prefix='solverpilot-highs-threads-') as directory:
        path = Path(directory)/'problem.npz'
        write_problem(path, problem)
        remaining = None if deadline is None else deadline-monotonic()
        if remaining is not None and remaining <= 0:
            return limited('thread isolation preparation exhausted the allowance')
        config = dict(protocol=PROTOCOL, problem_data_hash=problem.data_hash, deadline_s=deadline,
                      settings={k: v for k, v in asdict_public_backend(backend).items() if k != 'reuse'})
        # Resolve the same package checkout/installation even when the caller
        # inserted it through sys.path rather than an inherited PYTHONPATH.
        source_root = str(Path(__file__).resolve().parents[2])
        env = dict(os.environ, PYTHONPATH=source_root + os.pathsep + os.environ.get('PYTHONPATH', ''),
                   OMP_NUM_THREADS=str(backend.threads), OPENBLAS_NUM_THREADS=str(backend.threads),
                   MKL_NUM_THREADS=str(backend.threads))
        try:
            completed = subprocess.run([sys.executable, '-m', 'solverpilot.backends._highs_thread_worker', str(path)],
                input=json.dumps(config, allow_nan=False), capture_output=True, text=True, encoding='utf-8',
                env=env, timeout=remaining)
        except subprocess.TimeoutExpired:
            return limited('isolated thread configuration reached the allowance')
    if deadline is not None and monotonic() >= deadline:
        return limited('isolated thread result arrived after the allowance')
    if completed.returncode:
        raise RuntimeError('isolated HiGHS thread execution failed')
    records = [line[len(PREFIX):] for line in completed.stdout.splitlines() if line.startswith(PREFIX)]
    if len(records) != 1:
        raise RuntimeError('invalid isolated HiGHS response')
    try:
        payload = json.loads(records[0])
        if payload['protocol'] != PROTOCOL or payload['problem_data_hash'] != problem.data_hash:
            raise ValueError('isolated result identity mismatch')
        if payload['status'] not in {'optimal', 'limit_feasible', 'limit_no_solution', 'infeasible',
                'unbounded', 'infeasible_or_unbounded', 'solver_error', 'unknown'}:
            raise ValueError('unknown isolated status')
        x = None if payload['x'] is None else np.asarray(payload['x'], dtype=float)
        if x is not None and (x.shape != (problem.n_variables,) or not np.isfinite(x).all()):
            raise ValueError('invalid isolated candidate')
        objective = payload['objective']
        if objective is not None and (type(objective) not in (int, float) or not np.isfinite(objective)):
            raise ValueError('invalid isolated objective')
        raw = dict(payload['statistics'])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError('invalid isolated HiGHS response') from exc
    if deadline is not None and monotonic() >= deadline:
        return limited('isolated result validation exhausted the allowance')
    elapsed = perf_counter()-started
    raw.update(reuse_applied=False, reuse_mode='isolated_cold',
               reuse_reason='global_thread_configuration_conflict', isolated_thread_configuration=True,
               thread_isolation_total_s=elapsed)
    return BackendSolveResult(payload['status'], x, objective, raw)


def asdict_public_backend(backend):
    # Never use dataclasses.asdict on locks/native handles.
    from dataclasses import fields
    return {f.name: getattr(backend, f.name) for f in fields(backend) if f.init and not f.name.startswith('_')}
