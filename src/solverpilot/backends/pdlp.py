"""Optional CPU PDLP transport in an isolated process, for LP and diagonal QP."""

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import json
import os
import subprocess
import sys
import numpy as np
from scipy import sparse
from solverpilot.capabilities import BackendManifest, Capability, SupportLevel
from solverpilot.problem import LinearProblem, QuadraticProblem, ObjectiveSense
from solverpilot.problem.quadratic import require_confirmed_convexity
from .base import BackendSolveResult, BackendUnavailableError
from .metadata import version

PROTOCOL = "solverpilot-pdlp-v1"


@dataclass(slots=True)
class PDLPBackend:
    time_limit_s: float | None = 60.0
    threads: int = 1
    absolute_tolerance: float = 1e-9
    relative_tolerance: float = 1e-9
    iteration_limit: int = 100000

    @property
    def name(self):
        return "ortools-pdlp"

    def is_available(self):
        return find_spec("ortools") is not None

    @property
    def manifest(self):
        return BackendManifest(
            name=self.name,
            version=version("ortools") if self.is_available() else None,
            capabilities={
                Capability.LP: SupportLevel.NATIVE,
                Capability.CONVEX_QP: SupportLevel.NATIVE,
            },
            metadata={
                "native_adapter": True,
                "isolated_worker": True,
                "cpu_only": True,
                "quadratic_scope": "nonnegative diagonal only",
                "automatic_selection": False,
            },
        )

    def solve(self, problem):
        if not self.is_available():
            raise BackendUnavailableError("install solverpilot[pdlp]")
        if not isinstance(problem, (LinearProblem, QuadraticProblem)):
            raise TypeError("PDLP requires LinearProblem or diagonal QuadraticProblem")
        linear = problem.linear if isinstance(problem, QuadraticProblem) else problem
        if linear.has_integer_variables:
            raise ValueError("PDLP does not support integer variables")
        if isinstance(problem, QuadraticProblem):
            require_confirmed_convexity(problem)
            diagonal = problem.P.diagonal()
            if (problem.P - sparse.diags(diagonal, format="csr")).nnz or np.any(diagonal < 0):
                raise ValueError("PDLP supports only nonnegative diagonal quadratic objectives")
        else:
            diagonal = np.zeros(linear.n_variables)
        if type(self.threads) is not int or self.threads < 1:
            raise ValueError("threads must be a positive integer")
        if type(self.iteration_limit) is not int or self.iteration_limit < 1:
            raise ValueError("iteration_limit must be positive")
        for name, v in (
            ("absolute_tolerance", self.absolute_tolerance),
            ("relative_tolerance", self.relative_tolerance),
        ):
            if isinstance(v, bool) or not np.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.absolute_tolerance == self.relative_tolerance == 0:
            raise ValueError("at least one PDLP tolerance must be positive")
        if self.time_limit_s is not None and (
            isinstance(self.time_limit_s, bool)
            or not np.isfinite(self.time_limit_s)
            or self.time_limit_s <= 0
        ):
            raise ValueError("time_limit_s must be finite and positive")
        start = perf_counter()
        sign = 1.0 if linear.objective_sense is ObjectiveSense.MINIMIZE else -1.0
        config = {
            "protocol": PROTOCOL,
            "threads": self.threads,
            "time_limit_s": self.time_limit_s,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "iteration_limit": self.iteration_limit,
        }
        with TemporaryDirectory(prefix="solverpilot-pdlp-") as directory:
            request = Path(directory) / "problem.npz"
            np.savez(
                request,
                A_data=linear.A.data,
                A_indices=linear.A.indices,
                A_indptr=linear.A.indptr,
                A_shape=linear.A.shape,
                q=sign * linear.c,
                diagonal=diagonal,
                lower=linear.variable_lower,
                upper=linear.variable_upper,
                row_lower=linear.constraint_lower,
                row_upper=linear.constraint_upper,
            )
            prepared = perf_counter()
            remaining = (
                None if self.time_limit_s is None else self.time_limit_s - (prepared - start)
            )
            if remaining is not None and remaining <= 0:
                return BackendSolveResult(
                    "limit_no_solution",
                    None,
                    raw_statistics={"reason": "PDLP preparation exhausted wall time"},
                )
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
                        str(Path(__file__).with_name("_pdlp_worker.py")),
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
                return BackendSolveResult(
                    "limit_no_solution",
                    None,
                    raw_statistics={"reason": "isolated PDLP worker reached wall-time limit"},
                )
        elapsed = perf_counter() - start
        if completed.returncode:
            raise BackendUnavailableError("PDLP worker failed: " + completed.stderr[-2000:])
        prefix = "SOLVERPILOT_PDLP_RESULT="
        lines = [
            line[len(prefix) :] for line in completed.stdout.splitlines() if line.startswith(prefix)
        ]
        if len(lines) != 1:
            raise RuntimeError("invalid PDLP worker response")
        payload = json.loads(lines[0])
        if payload.get("protocol") != PROTOCOL:
            raise RuntimeError("PDLP worker protocol mismatch")
        reason = payload["termination"]
        status = {
            "TERMINATION_REASON_OPTIMAL": "optimal",
            "TERMINATION_REASON_PRIMAL_INFEASIBLE": "infeasible",
            # PDLP's dual-infeasible termination does not establish primal
            # feasibility: the primal may itself be infeasible (solve_log.proto).
            "TERMINATION_REASON_DUAL_INFEASIBLE": "infeasible_or_unbounded",
            "TERMINATION_REASON_PRIMAL_OR_DUAL_INFEASIBLE": "infeasible_or_unbounded",
        }.get(reason)
        candidate = payload.get("x")
        if status is None:
            status = (
                ("limit_feasible" if candidate is not None else "limit_no_solution")
                if reason
                in {
                    "TERMINATION_REASON_TIME_LIMIT",
                    "TERMINATION_REASON_ITERATION_LIMIT",
                    "TERMINATION_REASON_KKT_MATRIX_PASS_LIMIT",
                }
                else "solver_error"
            )
        x = (
            None
            if status in {"infeasible", "unbounded", "infeasible_or_unbounded", "solver_error"}
            or candidate is None
            else np.asarray(candidate, dtype=float)
        )
        if x is not None and (x.shape != (linear.n_variables,) or not np.isfinite(x).all()):
            raise RuntimeError("invalid PDLP primal shape or finite values")
        objective = (
            None
            if x is None
            else float(0.5 * np.dot(diagonal * x, x) + linear.c @ x + linear.objective_offset)
        )
        canonical_dual = None
        if (
            x is not None
            and payload.get("dual") is not None
            and payload.get("reduced_costs") is not None
        ):
            y = np.asarray(payload["dual"])
            r = np.asarray(payload["reduced_costs"])
            if y.shape != (linear.n_constraints,) or r.shape != (linear.n_variables,):
                raise RuntimeError("invalid PDLP dual dimensions")
            canonical_dual = np.r_[-y, -r].tolist()
        bound = payload.get("corrected_dual_objective")
        raw = {
            "native_termination": reason,
            "native_iterations": payload["iterations"],
            "canonical_dual": canonical_dual,
            "cpu_only": True,
            "isolated_worker": True,
            "solver_corrected_dual_bound": None
            if bound is None
            else sign * bound + linear.objective_offset,
            "bound_orientation": "lower" if sign > 0 else "upper",
            "bound_origin": "pdlp_reported",
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
