from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from enum import Enum
from importlib.metadata import PackageNotFoundError, version as package_version

import numpy as np
import scipy

from solverpilot import __version__, builtin_backend_candidates
from solverpilot.backends import probe_backends


def _jsonable(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def build_report() -> dict[str, object]:
    reports = probe_backends(builtin_backend_candidates())
    try:
        dist_version = package_version("solverpilot")
    except PackageNotFoundError:
        dist_version = None

    return {
        "schema_version": "1.0",
        "solverpilot_version": __version__,
        "distribution_version": dist_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "backends": [_jsonable(asdict(report)) for report in reports],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="solverpilot-backend-health",
        description="Actively smoke-test every built-in SolverPilot backend candidate.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="emit compact JSON instead of indented JSON",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="return non-zero unless every built-in candidate is healthy",
    )
    args = parser.parse_args(argv)

    report = build_report()
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))

    statuses = [entry["status"] for entry in report["backends"]]  # type: ignore[index]
    unhealthy = any(status == "unhealthy" for status in statuses)
    if unhealthy:
        return 2
    if args.require_all and any(status != "healthy" for status in statuses):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
