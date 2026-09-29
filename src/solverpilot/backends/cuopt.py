"""Explicit, isolated cuOpt GPU PDLP adapter for continuous linear programs.

Importability is a prerequisite, not execution qualification. No automatic
routing, CPU fallback, QP/MIP support, or workspace reuse is implied.
"""

from dataclasses import dataclass
from importlib.util import find_spec
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import json
import os
import platform
import subprocess
import sys

import numpy as np

from solverpilot.capabilities import BackendManifest, Capability, SupportLevel
from solverpilot.problem import LinearProblem, ObjectiveSense
from .base import BackendSolveResult, BackendUnavailableError
from .metadata import version

PROTOCOL = "solverpilot-cuopt-v1"
PREFIX = "SOLVERPILOT_CUOPT_RESULT="


@dataclass(slots=True)
class CuOptBackend:
    time_limit_s: float = 60.0
    threads: int = 1
    tolerance: float = 1e-9
    iteration_limit: int = 100000

    @property
    def name(self):
        return "cuopt-gpu"

    def is_available(self):
        return platform.system() == "Linux" and find_spec("cuopt") is not None

    @property
    def manifest(self):
        return BackendManifest(
            name=self.name,
            version=_binding_version(),
            capabilities={Capability.LP: SupportLevel.NATIVE},
            metadata={
                "native_adapter": True,
                "isolated_worker": True,
                "automatic_selection": False,
                "requires_gpu": True,
                "algorithm": "PDLP",
                "scope": "continuous LP only",
                "supports_wall_time_budget": True,
            },
        )

    def solve(self, problem):
        if not isinstance(problem, LinearProblem):
            raise TypeError("cuOpt GPU adapter requires a continuous LinearProblem")
        if problem.has_integer_variables:
            raise ValueError("cuOpt GPU adapter does not support integer variables")
        for name in ("time_limit_s", "tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("threads", "iteration_limit"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if max(problem.n_variables, problem.n_constraints, problem.A.nnz) > np.iinfo(np.int32).max:
            raise ValueError("cuOpt CSR dimensions and nonzero count must fit int32")
        if not self.is_available():
            raise BackendUnavailableError(
                "cuOpt requires Linux/WSL2, cuopt-cu12 and a compatible GPU"
            )
        start = perf_counter()
        sign = 1.0 if problem.objective_sense is ObjectiveSense.MINIMIZE else -1.0
        config = {
            "protocol": PROTOCOL,
            "threads": self.threads,
            "time_limit_s": self.time_limit_s,
            "tolerance": self.tolerance,
            "iteration_limit": self.iteration_limit,
        }
        with TemporaryDirectory(prefix="solverpilot-cuopt-") as directory:
            request = Path(directory) / "problem.npz"
            np.savez(
                request,
                A_data=problem.A.data,
                A_indices=problem.A.indices.astype(np.int32),
                A_indptr=problem.A.indptr.astype(np.int32),
                q=sign * problem.c,
                lower=problem.variable_lower,
                upper=problem.variable_upper,
                row_lower=problem.constraint_lower,
                row_upper=problem.constraint_upper,
            )
            prepared = perf_counter()
            remaining = self.time_limit_s - (prepared - start)
            if remaining <= 0:
                return _timeout("cuOpt preparation exhausted wall time")
            config["time_limit_s"] = remaining
            env = dict(
                os.environ,
                OMP_NUM_THREADS=str(self.threads),
                OPENBLAS_NUM_THREADS=str(self.threads),
                MKL_NUM_THREADS=str(self.threads),
            )
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("_cuopt_worker.py")),
                        str(request),
                    ],
                    input=json.dumps(config),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    env=env,
                    timeout=remaining,
                )
            except subprocess.TimeoutExpired:
                return _timeout("isolated cuOpt worker reached wall-time limit")
        elapsed = perf_counter() - start
        if completed.returncode:
            raise BackendUnavailableError("cuOpt worker failed: " + completed.stderr[-2000:])
        lines = [
            line[len(PREFIX) :] for line in completed.stdout.splitlines() if line.startswith(PREFIX)
        ]
        if len(lines) != 1:
            raise RuntimeError("invalid cuOpt worker response")
        payload = json.loads(lines[0])
        if not isinstance(payload, dict) or payload.get("protocol") != PROTOCOL:
            raise RuntimeError("cuOpt worker protocol mismatch")
        if (
            type(payload.get("gpu_count")) is not int
            or payload["gpu_count"] < 1
            or payload.get("solved_by") != "PDLP"
        ):
            raise RuntimeError("cuOpt did not report the requested GPU PDLP execution")
        reason = payload["termination"]
        status = {
            "Optimal": "optimal",
            "PrimalInfeasible": "infeasible",
            # Dual infeasibility alone does not establish primal feasibility.
            "DualInfeasible": "infeasible_or_unbounded",
            "PrimalOrDualInfeasible": "infeasible_or_unbounded",
            "UnboundedOrInfeasible": "infeasible_or_unbounded",
        }.get(reason)
        candidate = payload.get("x")
        if status is None:
            status = (
                ("limit_feasible" if candidate is not None else "limit_no_solution")
                if reason in {"TimeLimit", "IterationLimit"}
                else "solver_error"
            )
        x = None
        if candidate is not None and status in {"optimal", "limit_feasible"}:
            x = _vector(candidate, problem.n_variables, "primal")
        dual = None
        if (
            x is not None
            and payload.get("dual") is not None
            and payload.get("reduced_costs") is not None
        ):
            y = _vector(payload["dual"], problem.n_constraints, "dual")
            r = _vector(payload["reduced_costs"], problem.n_variables, "reduced costs")
            dual = np.r_[-y, -r].tolist()
        # Preserve the native objective, so the public validator can detect a
        # transport/objective mismatch instead of comparing a recomputed value to itself.
        objective = payload.get("objective") if x is not None else None
        if status == "optimal" and (x is None or objective is None):
            raise RuntimeError("cuOpt optimal termination omitted the primal solution or objective")
        if objective is not None:
            if isinstance(objective, bool) or not np.isfinite(objective):
                raise RuntimeError("invalid cuOpt objective")
            objective = sign * float(objective) + problem.objective_offset
        raw = {
            "native_termination": reason,
            "canonical_dual": dual,
            "gpu_execution_reported": True,
            "gpu_device": payload.get("gpu_device"),
            "native_solved_by": payload["solved_by"],
            "native_statistics": payload.get("statistics"),
            "native_solve_s": payload.get("native_solve_s"),
            "worker_timings": payload.get("timings"),
            "isolated_worker": True,
            "solver_parameters": config,
            "reuse_applied": False,
            "reuse_mode": "isolated_cold_solve",
            "reuse_report": {
                "workspace": "not_used",
                "primal_dual_start": "not_used",
                "numeric_factorization": "not_applicable",
            },
            "phase_timings": {
                "backend_build_s": prepared - start,
                "solve_s": elapsed - (prepared - start),
            },
        }
        return BackendSolveResult(status, x, objective, raw)


def _vector(value, size, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise RuntimeError(f"invalid cuOpt {name} dimensions or finite values")
    return a


def _timeout(reason):
    return BackendSolveResult("limit_no_solution", None, raw_statistics={"reason": reason})


def _binding_version():
    for package in ("cuopt-cu12", "cuopt-cu13"):
        try:
            return version(package)
        except PackageNotFoundError:
            pass
    return None
