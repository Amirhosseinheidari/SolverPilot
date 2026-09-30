"""Explicit measured-environment binding for the experimental CPU LP selector."""
import hashlib
import json
from pathlib import Path

from solverpilot.benchmark.environment import benchmark_environment_fingerprint


def _fingerprint(environment):
    return benchmark_environment_fingerprint(environment, thread_env_limit=1,
                                              solver_threads=1, worker_python_mode="normal")


def bind_lp_environment(model, training_protocol: str | Path, current: dict) -> dict:
    """Check the frozen training protocol, allowing <=1 MiB guest-RAM reporting drift.

    This does not mutate either environment or claim identical raw fingerprints.
    All other fingerprint fields must match exactly. The returned decision ID is
    an explicit compatibility binding, not a new observation or promotion grant.
    The current v1 protocol describes CPU single-thread normal-mode execution only.
    """
    raw = Path(training_protocol).read_bytes()
    if hashlib.sha256(raw).hexdigest() != model.protocol_sha256:
        raise ValueError("training protocol digest mismatch")
    protocol = json.loads(raw)
    if (protocol.get("schema") != "solverpilot.local-learned-lp.v1"
            or protocol.get("cpu_only") is not True):
        raise ValueError("unsupported training environment protocol")
    training = protocol["environment"]
    if _fingerprint(training) != model.environment_id:
        raise ValueError("training environment is not bound to this model")
    actual = _fingerprint(current)
    old_memory, new_memory = training.get("memory_bytes"), current.get("memory_bytes")
    memory_drift = (new_memory-old_memory if type(old_memory) is int and type(new_memory) is int
                    and old_memory > 0 and new_memory > 0 else None)
    adjusted = dict(current)
    if memory_drift is not None and abs(memory_drift) <= 1024*1024:
        adjusted["memory_bytes"] = old_memory
    # Effective environment variables must also agree, not just the declared caps.
    threads_match = current.get("env_threads") == training.get("env_threads")
    compatible = bool(threads_match and _fingerprint(adjusted) == model.environment_id)
    return {"compatible":compatible, "actual_environment_id":actual,
            # The legacy fingerprint omits observed thread variables. Prefix a
            # rejected binding so an equal raw hash cannot accidentally opt in.
            "decision_environment_id":model.environment_id if compatible else "incompatible:"+actual,
            "training_environment_id":model.environment_id,
            "memory_reporting_delta_bytes":memory_drift,
            "memory_reporting_tolerance_bytes":1024*1024,
            "reason":("exact_match" if actual == model.environment_id and threads_match
                      else "bounded_guest_memory_reporting_drift" if compatible else "environment_mismatch")}
