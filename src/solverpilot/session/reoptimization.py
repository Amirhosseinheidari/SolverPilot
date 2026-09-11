"""An owned semantic-model session for concurrent application callers."""
from threading import RLock
from solverpilot._synchronization import serialized
from solverpilot.model import Model
from solverpilot.backends import OSQPNativeBackend
from solverpilot.problem import LinearProblem, QuadraticProblem


class ReoptimizationSession:
    """Own a clone; update and solve form one serialized transaction.

    The caller's original model is never mutated. Parameters must have unique,
    nonempty names. A supplied backend is transferred to this session and should
    not be mutated externally while the session is in use.
    """
    def __init__(self, model: Model, *, backend=None):
        self._lock = RLock()
        names = [p.name for p in model.parameters]
        if any(not n for n in names) or len(set(names)) != len(names):
            raise ValueError('session parameters require unique nonempty names')
        self._model = model.clone()
        self._parameters = {p.name: p for p in self._model.parameters}
        self._compiled = self._model.compile()
        if backend is None and isinstance(self._compiled.execution_ir, QuadraticProblem) and OSQPNativeBackend().is_available():
            backend = OSQPNativeBackend()
        from solverpilot.conic import ConicProblem, ClarabelBackend
        if backend is None and isinstance(self._compiled.execution_ir, ConicProblem) and ClarabelBackend().is_available():
            backend = ClarabelBackend(reuse=True)
        self._backend = backend
        from solverpilot.runtime import default_registry
        self._registry = default_registry() if backend is None else None
        self._closed = False
        self._solving = False
        self.revision = 0
        self.last_result = None
        self.last_compilation_report = self._compiled.compilation_report

    @serialized
    def solve(self, *, updates=None, **options):
        if self._closed:
            raise RuntimeError('session is closed')
        if self._solving:
            raise RuntimeError('a solve callback cannot reenter its session')
        changes = dict(updates or {})
        if set(changes)-self._parameters.keys():
            raise KeyError(f'unknown parameters: {sorted(set(changes)-self._parameters.keys())}')
        old = {name: self._parameters[name].value for name in changes}
        before = self._model.data_hash
        self._solving = True
        try:
            for name, value in changes.items():
                self._parameters[name].value = value
            compiled = self._model.compile()
            if self._backend is not None:
                options['backend'] = self._backend
            elif isinstance(compiled.execution_ir, (LinearProblem, QuadraticProblem)):
                options['registry'] = self._registry
            result = compiled.solve(**options)
        except BaseException:
            for name, value in old.items():
                self._parameters[name].value = value
            raise
        finally:
            self._solving = False
        self.revision += before != self._model.data_hash
        self._compiled = compiled
        self.last_compilation_report = compiled.compilation_report
        self.last_result = result
        return result

    @serialized
    def values(self):
        from solverpilot.model import named_values
        return () if self.last_result is None else named_values(self._model, self._compiled, self.last_result)

    @serialized
    def close(self):
        if self._solving:
            raise RuntimeError('cannot close a session from its active solve callback')
        if not self._closed:
            close = getattr(self._backend, 'close', None)
            if callable(close):
                close()
            self._backend = None
            self._registry = None
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
