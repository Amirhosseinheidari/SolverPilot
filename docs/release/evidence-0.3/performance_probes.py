"""Audit-only experiments against the installed release; no package edits."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import gc
import json
import statistics
import time
import tracemalloc
from pathlib import Path
import numpy as np
from scipy import sparse
import solverpilot as sp
from solverpilot.backends import ScipyHighsLPBackend
from solverpilot.runtime.batch import CancellationToken, solve_batch, BatchExecutor


def main():
    out = {'version': sp.__version__, 'package': sp.__file__, 'thread_environment': 'OMP/MKL/OpenBLAS=1', 'seed': 8319}
    p = sp.LinearProblem.from_data(A=[[1., 1.]], c=[1., 2.], variable_lower=[0., 0.],
        variable_upper=[2., 2.], constraint_lower=[1.], constraint_upper=[np.inf])
    backend = ScipyHighsLPBackend()
    def serial():
        for _ in range(12):
            r = sp.execute(p, backend)
            assert r.validation.valid and abs(r.objective-1.) < 1e-7
    def batch(workers):
        results = solve_batch([p]*12, backend='scipy-highs-ds', max_workers=workers, timeout_s=30)
        assert all(r.validation_valid and abs(r.objective-1.) < 1e-7 for r in results)
    methods = {'serial_12': serial, **{f'batch_12_workers_{n}': lambda n=n: batch(n) for n in [1, 2, 4]}}
    persistent = BatchExecutor(backend='scipy-highs-ds', max_workers=4, max_pending=8, timeout_s=30)
    def reused():
        results = list(persistent.iter([p]*12))
        assert len(results) == 12 and all(r.validation_valid for r in results)
    def sequential():
        results = solve_batch([p]*12, backend='scipy-highs-ds', mode='sequential')
        assert len(results) == 12 and all(r.validation_valid for r in results)
    methods.update(persistent_batch_12_workers_4=reused, sequential_batch_12=sequential)
    samples = {k: [] for k in methods}
    for iteration in range(4):
        names = list(methods)
        np.random.default_rng(8319+iteration).shuffle(names)
        for name in names:
            t = time.perf_counter(); methods[name](); elapsed = time.perf_counter()-t
            if iteration: samples[name].append(elapsed*1000)
    out['tiny_lp_batch'] = {k: {'median_ms': statistics.median(v), 'samples_ms': v} for k, v in samples.items()}
    persistent.close()
    print('Batch latency measurements complete', flush=True)

    token = CancellationToken(); token.cancel(); count = 0
    def generated():
        nonlocal count
        for _ in range(20000):
            count += 1
            yield p
    gc.collect(); tracemalloc.start(); t = time.perf_counter()
    results = solve_batch(generated(), max_workers=2, cancellation=token)
    elapsed = time.perf_counter()-t; _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    assert all(r.status == 'cancelled' for r in results)
    out['already_cancelled_batch'] = {'generated_inputs': count, 'result_count': len(results),
        'peak_python_mib': peak/1024**2, 'elapsed_s': elapsed, 'note': 'Python allocations only; includes queued futures and retained outputs; not native RSS'}

    rng = np.random.default_rng(8319); compilation = []
    for n, m in [(64, 32), (256, 128), (512, 256)]:
        A = sparse.random(m, n, density=.02, random_state=rng, format='csr'); c = np.ones(n)
        b = .25*np.asarray(A.sum(axis=1)).reshape(-1)
        times = {'canonical_sparse': [], 'semantic_dense': [], 'semantic_sparse': []}
        for repeat in range(3):
            for method in (list(times) if repeat%2 else list(reversed(times))):
                start = time.perf_counter()
                if method == 'canonical_sparse':
                    compiled = sp.LinearProblem.from_data(A=A, c=c, variable_lower=np.zeros(n), variable_upper=np.ones(n), constraint_lower=b, constraint_upper=np.full(m, np.inf))
                else:
                    model = sp.Model(); x = model.variable(n, lower=0., upper=1.)
                    model.add((model.constant(A) if method == 'semantic_sparse' else model.constant(A.toarray())) @ x >= b); model.minimize(c@x)
                    compiled = model.compile().execution_ir
                times[method].append((time.perf_counter()-start)*1000)
                assert compiled.A.shape == (m, n) and (compiled.A-A).nnz == 0
        compilation.append({'n': n, 'm': m, 'nonzeros': A.nnz,
            'median_ms': {k: statistics.median(v) for k,v in times.items()}, 'samples_ms': times,
            'scope': 'model construction plus first compile versus canonical IR construction; no solves'})
    out['compilation'] = compilation
    Path(__file__).with_name('performance-probes.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
