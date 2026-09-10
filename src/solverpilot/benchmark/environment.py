from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys
from typing import Iterable


_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def _memory_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(errors="ignore").splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    return None


def thread_environment(limit: int | None) -> dict[str, str]:
    """Return environment variables used to cap common C/BLAS thread pools.

    This does not prove that a solver's own internal thread pool is controlled.
    Backend-level thread control is represented separately by ``solver_threads`` in
    the benchmark protocol.
    """

    if limit is None:
        return {}
    if limit < 1:
        raise ValueError("thread environment limit must be >= 1")
    value = str(int(limit))
    return {key: value for key in _THREAD_ENV_KEYS}


def capture_environment(*, packages: Iterable[str] = ()) -> dict:
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "memory_bytes": _memory_bytes(),
        "packages": versions,
        "env_threads": {key: os.environ.get(key) for key in _THREAD_ENV_KEYS},
    }


def benchmark_environment_fingerprint(
    environment: dict,
    *,
    thread_env_limit: int | None,
    solver_threads: int | None,
    worker_python_mode: str = "no_site",
) -> str:
    """Hash timing-relevant execution environment fields, excluding hostname.

    Hostname is deliberately excluded so homogeneous cluster nodes can merge. CPU,
    platform, Python/package versions, and the declared thread policy are retained.
    """

    payload = {
        "schema_version": "1.0",
        "python": environment.get("python"),
        "platform": environment.get("platform"),
        "machine": environment.get("machine"),
        "cpu_model": environment.get("cpu_model"),
        "logical_cpus": environment.get("logical_cpus"),
        "memory_bytes": environment.get("memory_bytes"),
        "packages": environment.get("packages", {}),
        "thread_env_limit": thread_env_limit,
        "solver_threads": solver_threads,
        "worker_python_mode": worker_python_mode,
        "effective_thread_env": thread_environment(thread_env_limit),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def write_environment(path: str | Path, *, packages: Iterable[str] = ()) -> dict:
    payload = capture_environment(packages=packages)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
