"""Reproducible single-host latency and repeated-solve memory measurements.

Run with the desired installed wheel in an isolated environment. No coverage or
profiler should be active. These synthetic workloads are not industrial claims.
"""
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import gc
import json
import platform
import statistics
from time import perf_counter
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
import solverpilot as sp
from solverpilot.backends import ScipyHighsLPBackend, OSQPNativeBackend


def measure(methods, rounds=9):
    samples = {k: [] for k in methods}
    for iteration in range(rounds+2):
        names = list(methods); np.random.default_rng(20260912+iteration).shuffle(names)
        for name in names:
            start = perf_counter(); methods[name](); elapsed = perf_counter()-start
            if iteration >= 2:
                samples[name].append(elapsed*1000)
    return {k: {'median_ms': statistics.median(v), 'p95_ms': float(np.percentile(v, 95)), 'samples_ms': v} for k, v in samples.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--memory-rounds', type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(20260912)
    out = {'version': sp.__version__, 'import_path': sp.__file__, 'platform': platform.platform(),
           'thread_environment': {k: os.environ[k] for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
           'seed': 20260912, 'benchmarks': {}, 'method': 'two warmups, nine alternating randomized timed rounds; milliseconds; construction excluded unless named'}
    for n, m, density in [(32, 16, .7), (256, 128, .1), (1500, 750, .02)]:
        A = sparse.random(m, n, density=density, random_state=rng, format='csr')
        rhs = .25*np.asarray(A.sum(axis=1)).reshape(-1); c = rng.uniform(.5, 2., n)
        p = sp.LinearProblem.from_data(A=A, c=c, variable_lower=np.zeros(n), variable_upper=np.ones(n),
                                      constraint_lower=rhs, constraint_upper=np.full(m, np.inf))
        registry = sp.default_registry(); b = ScipyHighsLPBackend()
        expected = linprog(c, A_ub=-A, b_ub=-rhs, bounds=(0., 1.), method='highs-ds').fun
        def wrapped():
            r = sp.execute(p, b)
            assert r.validation.valid and abs(r.objective-expected) < 1e-6
        methods = {'direct': lambda: linprog(c, A_ub=-A, b_ub=-rhs, bounds=(0., 1.), method='highs-ds'), 'execute': wrapped}
        if n == 256:
            methods.update({'auto': lambda: sp.solve(p), 'auto_shared_registry': lambda: sp.solve(p, registry=registry)})
        out['benchmarks'][f'LP-{n}'] = measure(methods)
    if OSQPNativeBackend().is_available():
        n = 256; q = rng.normal(size=n)
        def construct():
            return sp.QuadraticProblem.from_data(P=sparse.eye(n), A=sparse.csr_matrix((0, n)), q=q,
                                                variable_lower=np.zeros(n), variable_upper=np.ones(n),
                                                constraint_lower=[], constraint_upper=[])
        p = construct(); b = OSQPNativeBackend(eps_abs=1e-8, eps_rel=1e-8)
        out['benchmarks']['QP-256'] = measure({'construct': construct, 'fresh': lambda: sp.execute(p, OSQPNativeBackend()), 'reuse': lambda: sp.execute(p, b)})
        try:
            import psutil
            process = psutil.Process(); checkpoints = []
            for i in range(args.memory_rounds+100):
                result = sp.execute(p, b)
                assert result.validation.valid
                if i >= 99 and (i-99) % 100 == 0:
                    gc.collect(); checkpoints.append({'round': i+1, 'rss_mib': process.memory_info().rss/1024**2})
            out['memory'] = {'scope': 'same QP, one reused backend; latest result retained; excludes session history',
                             'rounds': args.memory_rounds+100, 'checkpoints': checkpoints}
        except ImportError:
            out['memory'] = {'unavailable': 'install psutil for RSS measurements'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({'version': out['version'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
