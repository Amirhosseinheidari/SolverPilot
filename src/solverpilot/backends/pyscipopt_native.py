from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from importlib.metadata import PackageNotFoundError
from .metadata import version

import numpy as np

from solverpilot.capabilities import BackendManifest, Capability, SupportLevel
from solverpilot.problem import LinearProblem, ObjectiveSense, QuadraticProblem, VariableDomain
from solverpilot.problem.quadratic import require_confirmed_convexity

from .base import BackendSolveResult, BackendUnavailableError


def _scip_status(status: str, has_solution: bool) -> str:
    s = status.strip().lower().replace("_", "")
    if s == "optimal":
        return "optimal"
    if s == "infeasible":
        return "infeasible"
    if s == "unbounded":
        return "unbounded"
    if s in {"inforunbd", "infeasibleorunbounded"}:
        return "infeasible_or_unbounded"
    if any(token in s for token in ("timelimit", "nodelimit", "stallnodelimit", "gaplimit", "sollimit", "bestsollimit", "memlimit")):
        return "limit_feasible" if has_solution else "limit_no_solution"
    if s in {"userinterrupt", "restartlimit"}:
        return "limit_feasible" if has_solution else "limit_no_solution"
    if s in {"unknown", "terminate"}:
        return "unknown"
    return "solver_error"


@dataclass(frozen=True, slots=True)
class SCIPIISResult:
    constraint_names: tuple[str, ...]
    variable_names: tuple[str, ...]
    irreducible: bool
    time_s: float
    nodes: int


@dataclass(slots=True)
class PySCIPOptNativeBackend:
    """Optional native SCIP adapter for LP/MILP plus IIS extraction.

    The implementation intentionally uses only documented PySCIPOpt surface APIs.
    Native reoptimization and solution injection are left unclaimed until exercised in
    an environment containing PySCIPOpt/SCIP.
    """

    time_limit_s: float | None = None
    display: bool = False

    @property
    def manifest(self) -> BackendManifest:
        pkg_version = None
        if self.is_available():
            try:
                pkg_version = version("pyscipopt")
            except PackageNotFoundError:
                pass
        return BackendManifest(
            name="pyscipopt-native",
            version=pkg_version,
            capabilities={
                Capability.LP: SupportLevel.NATIVE,
                Capability.CONVEX_QP: SupportLevel.UNSUPPORTED,
                Capability.MILP: SupportLevel.NATIVE,
                Capability.MIP_START: SupportLevel.UNKNOWN,
                Capability.PRIMAL_START: SupportLevel.UNKNOWN,
                Capability.DUAL_START: SupportLevel.UNKNOWN,
                Capability.BASIS_START: SupportLevel.UNKNOWN,
                Capability.SAME_SPARSITY_DATA_UPDATE: SupportLevel.UNKNOWN,
                Capability.STRUCTURAL_INCREMENTAL_UPDATE: SupportLevel.UNKNOWN,
                Capability.NATIVE_REOPTIMIZATION: SupportLevel.UNKNOWN,
                Capability.CALLBACK_PROGRESS: SupportLevel.UNKNOWN,
                Capability.IIS: SupportLevel.NATIVE,
                Capability.INFEASIBILITY_CERTIFICATE: SupportLevel.UNKNOWN,
            },
            metadata={
                "native_adapter": True,
                "iis_method": "Model.generateIIS",
                "reuse_claim": "none in M2 adapter",
            },
        )

    def is_available(self) -> bool:
        return importlib.util.find_spec("pyscipopt") is not None

    def _build_model(self, problem: LinearProblem):
        if not self.is_available():
            raise BackendUnavailableError("PySCIPOpt is not installed")
        from pyscipopt import Model, quicksum

        model = Model("solverpilot")
        if not self.display:
            model.hideOutput()
        if self.time_limit_s is not None:
            if self.time_limit_s <= 0:
                raise ValueError("time_limit_s must be positive")
            model.setParam("limits/time", float(self.time_limit_s))

        variables = []
        for j, domain in enumerate(problem.domains):
            lbv = float(problem.variable_lower[j])
            ubv = float(problem.variable_upper[j])
            lb = None if np.isneginf(lbv) else lbv
            ub = None if np.isposinf(ubv) else ubv
            if domain == VariableDomain.BINARY.value:
                vtype = "B"
            elif domain == VariableDomain.INTEGER.value:
                vtype = "I"
            else:
                vtype = "C"
            variables.append(model.addVar(name=f"x_{j}", vtype=vtype, lb=lb, ub=ub))

        objective = quicksum(float(problem.c[j]) * variables[j] for j in range(problem.n_variables) if problem.c[j] != 0.0)
        sense = "minimize" if problem.objective_sense is ObjectiveSense.MINIMIZE else "maximize"
        model.setObjective(objective, sense=sense)
        if problem.objective_offset != 0.0:
            # PySCIPOpt exposes objective constants separately from the linear expression.
            # Keeping the offset in the native model makes solver-reported bounds/gaps and
            # objective values semantically consistent with the canonical problem.
            model.addObjoffset(float(problem.objective_offset))

        A = problem.A.tocsr()
        for i in range(problem.n_constraints):
            start, end = A.indptr[i], A.indptr[i + 1]
            expr = quicksum(float(A.data[k]) * variables[int(A.indices[k])] for k in range(start, end))
            lo = float(problem.constraint_lower[i])
            hi = float(problem.constraint_upper[i])
            if np.isfinite(lo) and np.isfinite(hi) and lo == hi:
                model.addCons(expr == lo, name=f"row_{i}_eq")
            else:
                if np.isfinite(lo):
                    model.addCons(expr >= lo, name=f"row_{i}_lower")
                if np.isfinite(hi):
                    model.addCons(expr <= hi, name=f"row_{i}_upper")
        return model, variables

    def solve(self, problem: LinearProblem | QuadraticProblem) -> BackendSolveResult:
        if isinstance(problem, QuadraticProblem):
            raise ValueError("M2 PySCIPOpt adapter supports LP/MILP only")
        if not isinstance(problem, LinearProblem):
            raise TypeError(f"unsupported problem type: {type(problem)!r}")
        model, variables = self._build_model(problem)
        model.optimize()
        status_text = str(model.getStatus())
        has_solution = int(model.getNSols()) > 0
        backend_status = _scip_status(status_text, has_solution)
        x = None
        if has_solution and backend_status in {"optimal", "limit_feasible"}:
            sol = model.getBestSol()
            x_candidate = np.asarray([model.getSolVal(sol, v) for v in variables], dtype=np.float64)
            if np.all(np.isfinite(x_candidate)):
                x = x_candidate
        objective = None if x is None else float(problem.c @ x + problem.objective_offset)
        raw: dict[str, object] = {
            "scip_status": status_text,
            "n_solutions": int(model.getNSols()),
            "solve_time": float(model.getSolvingTime()),
            "nodes": int(model.getNTotalNodes()),
            "reuse_applied": False,
            "reuse_mode": "cold_native_solve",
        }
        if has_solution:
            try:
                raw["gap"] = float(model.getGap())
                raw["dual_bound"] = float(model.getDualbound())
            except Exception:
                pass
        return BackendSolveResult(
            backend_status=backend_status,
            x=x,
            objective_reported=objective,
            raw_statistics=raw,
        )

    def generate_iis(self, problem: LinearProblem) -> SCIPIISResult:
        """Generate a SCIP IIS for an infeasible LP/MILP.

        Raises ValueError when SCIP does not classify the model as infeasible.
        """
        if not isinstance(problem, LinearProblem):
            raise TypeError("SCIP IIS adapter currently accepts LinearProblem only")
        model, _ = self._build_model(problem)
        model.optimize()
        if str(model.getStatus()).strip().lower() != "infeasible":
            raise ValueError("IIS requested for a model not classified infeasible by SCIP")
        iis = model.generateIIS()
        subscip = iis.getSubscip()
        return SCIPIISResult(
            constraint_names=tuple(sorted(str(c.name) for c in subscip.getConss())),
            variable_names=tuple(sorted(str(v.name) for v in subscip.getVars())),
            irreducible=bool(iis.isSubscipIrreducible()),
            time_s=float(iis.getTime()),
            nodes=int(iis.getNNodes()),
        )
