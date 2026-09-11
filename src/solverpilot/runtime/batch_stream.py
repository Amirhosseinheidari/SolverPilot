"""Streaming batches with bounded submission and restartable persistent workers."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import os
import threading
from time import monotonic
import numpy as np

from solverpilot.exceptions import BudgetNotSupportedError
from solverpilot.problem import LinearProblem, QuadraticProblem
from solverpilot.validate import CandidateSolution, ValidationTolerances, validate_solution
from .batch import BatchItem, CancellationToken, _PROCESS_LIFECYCLE_LOCK


def _loop(connection, backend, tolerances, memory_mb, budget):
    try:
        if memory_mb is not None:
            import resource

            limit = int(memory_mb * 1024 * 1024)
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        from .auto import default_registry, solve

        registry = default_registry()
        connection.send({"ready": True})
        while True:
            problem = connection.recv()
            if problem is None:
                break
            try:
                result = solve(
                    problem,
                    backend=backend,
                    registry=registry,
                    tolerances=tolerances,
                    budget=budget,
                )
                connection.send(
                    {
                        "status": result.status.value,
                        "x": None if result.x is None else result.x.tolist(),
                        "objective": result.objective,
                        "backend": result.trace.backend,
                    }
                )
            except Exception as exc:
                connection.send({"error": f"{type(exc).__name__}: {exc}"})
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        connection.close()


class _Worker:
    def __init__(self, owner):
        self.owner, self.process, self.connection = owner, None, None

    def close(self):
        process, connection = self.process, self.connection
        self.process = self.connection = None
        if connection is not None:
            connection.close()
        if process is not None and process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(2)
            if process.is_alive():
                process.kill()
                process.join()
            with _PROCESS_LIFECYCLE_LOCK:
                process.close()

    def run(self, index, problem, token, local):
        owner = self.owner
        start = monotonic()

        def stopped(status, error=None):
            return BatchItem(index, status, None, None, False, None, monotonic() - start, error)

        def cancelled():
            return (
                local.cancelled or owner._stop.cancelled or (token is not None and token.cancelled)
            )

        if cancelled():
            return stopped("cancelled")
        try:
            if self.process is None:
                parent, child = mp.get_context("spawn").Pipe()
                self.connection = parent
                self.process = mp.get_context("spawn").Process(
                    target=_loop,
                    args=(
                        child,
                        owner.backend,
                        owner.tolerances,
                        owner.memory_mb,
                        owner.solver_budget,
                    ),
                )
                try:
                    with _PROCESS_LIFECYCLE_LOCK:
                        self.process.start()
                finally:
                    child.close()
                # Wait for imports/registry setup before sending a potentially
                # large payload. Otherwise send() can block on a full pipe
                # while the child is still starting, bypassing the deadline.
                while not self.connection.poll(0.01):
                    if cancelled():
                        self.close()
                        return stopped("cancelled")
                    if owner.timeout_s is not None and monotonic() - start >= owner.timeout_s:
                        self.close()
                        return stopped("timeout")
                    if not self.process.is_alive():
                        self.close()
                        return stopped("error", "worker exited during startup")
                if self.connection.recv() != {"ready": True}:
                    self.close()
                    return stopped("error", "invalid worker startup response")
            self.connection.send(problem)
            while True:
                if cancelled():
                    self.close()
                    return stopped("cancelled")
                if owner.timeout_s is not None and monotonic() - start >= owner.timeout_s:
                    self.close()
                    return stopped("timeout")
                if self.connection.poll(0.01):
                    payload = self.connection.recv()
                    if "error" in payload:
                        self.close()
                        return stopped("error", payload["error"])
                    x = payload["x"]
                    validation = (
                        None
                        if x is None
                        else validate_solution(
                            problem,
                            CandidateSolution(np.asarray(x), payload["objective"]),
                            tolerances=owner.tolerances,
                        )
                    )
                    valid = validation is not None and validation.valid
                    return BatchItem(
                        index,
                        payload["status"] if x is None or valid else "invalid_solution",
                        x,
                        None if validation is None else validation.objective_recomputed,
                        valid,
                        payload["backend"],
                        monotonic() - start,
                    )
                if not self.process.is_alive():
                    error = f"worker exited with code {self.process.exitcode}"
                    self.close()
                    return stopped("error", error)
        except (EOFError, OSError, ValueError) as exc:
            self.close()
            return stopped("error", f"{type(exc).__name__}: {exc}")


class BatchExecutor:
    """Keep solver processes alive across ordered, bounded batches.

    Use as a context manager. Timeouts include initial process startup; each
    timed-out worker is killed and replaced for its next job. Solver budgets
    additionally limit native solve time. Only trusted local problems use IPC.
    """

    def __init__(
        self,
        *,
        backend=None,
        max_workers=1,
        max_pending=None,
        timeout_s=None,
        memory_mb=None,
        tolerances=None,
        solver_budget=None,
    ):
        if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        max_pending = 2 * max_workers if max_pending is None else max_pending
        if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending < 1:
            raise ValueError("max_pending must be a positive integer")
        for key, value in [("timeout_s", timeout_s), ("memory_mb", memory_mb)]:
            if value is not None and (
                isinstance(value, bool) or not np.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{key} must be finite and positive")
        if memory_mb is not None and os.name != "posix":
            raise BudgetNotSupportedError("batch address-space limits require POSIX RLIMIT_AS")
        if backend is not None and not isinstance(backend, str):
            raise TypeError("batch backend must be a registered name")
        self.backend, self.timeout_s, self.memory_mb = backend, timeout_s, memory_mb
        self.solver_budget, self.tolerances = solver_budget, tolerances or ValidationTolerances()
        self.max_pending = max_pending
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._local = threading.local()
        self._workers, self._guard, self._stop = [], threading.Lock(), CancellationToken()

    def _run(self, index, problem, token, local):
        if not hasattr(self._local, "worker"):
            self._local.worker = _Worker(self)
            with self._guard:
                self._workers.append(self._local.worker)
        return self._local.worker.run(index, problem, token, local)

    def iter(self, problems, *, cancellation=None):
        if self._stop.cancelled:
            raise RuntimeError("batch executor is closed")
        source, pending, local = enumerate(problems), deque(), CancellationToken()
        exhausted = False
        try:
            while True:
                while (
                    not exhausted
                    and len(pending) < self.max_pending
                    and not self._stop.cancelled
                    and not (cancellation and cancellation.cancelled)
                ):
                    try:
                        index, problem = next(source)
                    except StopIteration:
                        exhausted = True
                        break
                    if not isinstance(problem, (LinearProblem, QuadraticProblem)):
                        raise TypeError("batch requires canonical LP/QP problems")
                    pending.append(
                        self._pool.submit(self._run, index, problem, cancellation, local)
                    )
                if not pending:
                    return
                yield pending.popleft().result()
        finally:
            local.cancel()
            for future in pending:
                if not future.cancel():
                    future.result()

    def close(self):
        self._stop.cancel()
        self._pool.shutdown(wait=True, cancel_futures=True)
        for worker in self._workers:
            worker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def iter_solve_batch(problems, *, cancellation=None, mode="process", **options):
    """Consume at most max_pending inputs ahead of the last delivered result.

    Sequential mode has no process isolation and rejects hard resource limits.
    Close the iterator (or its context-managed executor) when ending early.
    """
    if mode not in ("process", "sequential"):
        raise ValueError("mode must be process or sequential")
    with BatchExecutor(**options) as executor:
        if mode == "process":
            yield from executor.iter(problems, cancellation=cancellation)
            return
        if executor.timeout_s is not None or executor.memory_mb is not None:
            raise BudgetNotSupportedError("hard resource limits require process mode")
        from .auto import default_registry, solve

        registry = default_registry()
        source = iter(problems)
        index = 0
        while cancellation is None or not cancellation.cancelled:
            try:
                problem = next(source)
            except StopIteration:
                return
            if not isinstance(problem, (LinearProblem, QuadraticProblem)):
                raise TypeError("batch requires canonical LP/QP problems")
            start = monotonic()
            result = solve(
                problem,
                backend=executor.backend,
                registry=registry,
                tolerances=executor.tolerances,
                budget=executor.solver_budget,
            )
            yield BatchItem(
                index,
                result.status.value,
                result.x,
                result.objective,
                bool(result.validation and result.validation.valid),
                result.trace.backend,
                monotonic() - start,
            )
            index += 1
