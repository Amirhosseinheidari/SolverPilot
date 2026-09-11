from functools import wraps


def serialized(method):
    """Serialize operations on an instance-owned reentrant lock."""
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call
