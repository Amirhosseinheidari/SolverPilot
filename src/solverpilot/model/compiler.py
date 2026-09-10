from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import hashlib
import json
from time import perf_counter
from typing import Any, TYPE_CHECKING

import numpy as np
from scipy import sparse

from solverpilot.problem import LinearProblem, ObjectiveSense, QuadraticProblem, VariableDomain
from solverpilot.session.mutations import classify_mutations
from solverpilot.bridges import (
    AffineScalar, BridgePolicy, BridgeUnavailable, MappingAvailability, TransformationTape,
    choose_indicator_path, lower_indicator_big_m, validate_semantic_primal,
)
from solverpilot.capabilities.v2 import BackendCapabilityManifestV2

from .errors import CompileError, ShapeError
from .expression import ExprNode
from .model import Constraint, IndicatorConstraint, Model
from .sets import EqualTo, GreaterThan, Interval, LessThan

if TYPE_CHECKING:
    from solverpilot.conic.ir import ConicProblem
    from solverpilot.nlp.ir import NLPProblem
    from solverpilot.minlp.ir import MINLPProblem


_CAPABILITY_SIGNATURE = "legacy-ir:p2:v1"
_NUMERICAL_POLICY = "p0-rfc-012:p2-default-v1"
_CACHE_KEY = f"{_CAPABILITY_SIGNATURE}|{_NUMERICAL_POLICY}"
_P4_NUMERICAL_POLICY = "p0-rfc-012:p4-safe-bridge-v1"
_P4_CAPABILITY_SIGNATURE = "legacy-ir:p4:v1"
_MAX_SNAPSHOTS = 8


@dataclass(slots=True)
class _Polynomial:
    constant: float
    linear: dict[int, float]
    quadratic: dict[tuple[int, int], float]

    @classmethod
    def scalar(cls, value: float) -> "_Polynomial":
        return cls(float(value), {}, {})

    @classmethod
    def variable(cls, index: int) -> "_Polynomial":
        return cls(0.0, {index: 1.0}, {})

    @property
    def degree(self) -> int:
        if self.quadratic:
            return 2
        if self.linear:
            return 1
        return 0

    def __add__(self, other: "_Polynomial") -> "_Polynomial":
        lin = dict(self.linear)
        for i, v in other.linear.items():
            lin[i] = lin.get(i, 0.0) + v
            if lin[i] == 0.0:
                del lin[i]
        quad = dict(self.quadratic)
        for k, v in other.quadratic.items():
            quad[k] = quad.get(k, 0.0) + v
            if quad[k] == 0.0:
                del quad[k]
        return _Polynomial(self.constant + other.constant, lin, quad)

    def __neg__(self) -> "_Polynomial":
        return _Polynomial(
            -self.constant,
            {i: -v for i, v in self.linear.items()},
            {k: -v for k, v in self.quadratic.items()},
        )

    def __sub__(self, other: "_Polynomial") -> "_Polynomial":
        return self + (-other)

    def __mul__(self, other: "_Polynomial") -> "_Polynomial":
        if self.degree + other.degree > 2:
            raise CompileError("expression degree exceeds quadratic P2 scope")
        constant = self.constant * other.constant
        lin: dict[int, float] = {}
        for i, v in self.linear.items():
            lin[i] = lin.get(i, 0.0) + v * other.constant
        for i, v in other.linear.items():
            lin[i] = lin.get(i, 0.0) + v * self.constant
        quad: dict[tuple[int, int], float] = {}
        for k, v in self.quadratic.items():
            quad[k] = quad.get(k, 0.0) + v * other.constant
        for k, v in other.quadratic.items():
            quad[k] = quad.get(k, 0.0) + v * self.constant
        for i, a in self.linear.items():
            for j, b in other.linear.items():
                key = (i, j) if i <= j else (j, i)
                quad[key] = quad.get(key, 0.0) + a * b
        lin = {k: v for k, v in lin.items() if v != 0.0}
        quad = {k: v for k, v in quad.items() if v != 0.0}
        return _Polynomial(constant, lin, quad)


@dataclass(frozen=True, slots=True)
class CompilationReport:
    target: str
    semantic_analysis_s: float
    lowering_s: float
    total_s: float
    n_semantic_variables: int
    n_execution_variables: int
    n_semantic_constraints: int
    n_execution_constraints: int
    warnings: tuple[str, ...] = ()
    cache_status: str = "miss"
    changed_parameters: tuple[str, ...] = ()
    reused_execution_rows: int = 0
    recompiled_execution_rows: int = 0
    objective_recompiled: bool = True
    execution_mutations: tuple[str, ...] = ()
    bridge_search_s: float = 0.0
    n_transformations: int = 0
    rejected_bridge_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CompilerCacheInfo:
    semantic_hash: str | None
    snapshots: int
    full_compiles: int
    partial_compiles: int
    snapshot_hits: int
    semantic_invalidations: int
    last_data_hash: str | None


@dataclass(frozen=True, slots=True)
class CompiledModel:
    execution_ir: LinearProblem | QuadraticProblem | "ConicProblem" | "NLPProblem" | "MINLPProblem"
    semantic_hash: str
    data_hash: str
    compilation_hash: str
    parameter_map: dict[str, Any]
    source_map: dict[str, Any]
    transformation_tape: tuple[dict[str, Any], ...]
    capability_signature: str
    mutation_plan: dict[str, Any]
    plan_sensitive_dependencies: tuple[str, ...]
    precondition_certificate_hashes: tuple[str, ...]
    compilation_report: CompilationReport
    reconstruction_contract: dict[str, Any]
    schema_version: str = "solverpilot.compiled-model.p2.v1"

    def solve(self, **kwargs):
        from solverpilot.conic import ConicProblem, solve_conic
        if isinstance(self.execution_ir, ConicProblem):
            return solve_conic(self.execution_ir, **kwargs)
        try:
            from solverpilot.minlp import MINLPProblem, solve_outer_approximation
            if isinstance(self.execution_ir, MINLPProblem):
                return solve_outer_approximation(self.execution_ir, **kwargs)
        except ImportError:
            pass
        try:
            from solverpilot.nlp import NLPProblem, CasadiIpoptBackend
            if isinstance(self.execution_ir, NLPProblem):
                backend = kwargs.pop("backend", None) or CasadiIpoptBackend()
                return backend.solve(self.execution_ir, **kwargs)
        except ImportError:
            pass
        from solverpilot.runtime import solve
        return solve(self.execution_ir, **kwargs)

    def refresh(self, model: Model, *, use_cache: bool = True, bridge_policy=None, capabilities=None) -> "CompiledModel":
        return compile_model(model, use_cache=use_cache, bridge_policy=bridge_policy, capabilities=capabilities)

    def reconstruct_primal(self, x):
        import numpy as _np
        # P4 bridges currently add no execution variables, so primal reconstruction
        # is an identity map. The tape still records this explicitly for auditability.
        for step in self.transformation_tape:
            if step.get("primal_mapping") not in {"identity", "available"}:
                raise CompileError("primal reconstruction is unavailable for a transformation step")
        return _np.asarray(x, dtype=float).copy()

    def validate_original(self, model: Model, x, *, atol: float = 1e-8):
        try:
            from solverpilot.minlp import MINLPProblem, validate_minlp_solution
            if isinstance(self.execution_ir, MINLPProblem):
                return validate_minlp_solution(self.execution_ir, self.reconstruct_primal(x), atol=atol, rtol=atol)
        except ImportError:
            pass
        try:
            from solverpilot.nlp import NLPProblem, validate_nlp_solution
            if isinstance(self.execution_ir, NLPProblem):
                return validate_nlp_solution(self.execution_ir, self.reconstruct_primal(x), atol=atol, rtol=atol)
        except ImportError:
            pass
        return validate_semantic_primal(model, self.reconstruct_primal(x), atol=atol)


@dataclass(frozen=True, slots=True)
class _Template:
    semantic_hash: str
    target: str
    offsets: dict[str, tuple[int, int]]
    n_variables: int
    variable_lower: np.ndarray
    variable_upper: np.ndarray
    domains: tuple[str, ...]
    source_map: dict[str, Any]
    constraint_by_id: dict[str, Constraint]
    constraint_dependencies: dict[str, frozenset[str]]
    objective_dependencies: frozenset[str]
    parameter_to_constraints: dict[str, tuple[str, ...]]
    parameter_affects_objective: dict[str, bool]
    compilation_hash: str


@dataclass(slots=True)
class _CacheEntry:
    semantic_hash: str
    template: _Template
    current: CompiledModel
    parameter_versions: dict[str, int]
    snapshots: OrderedDict[str, CompiledModel]
    full_compiles: int = 1
    partial_compiles: int = 0
    snapshot_hits: int = 0
    semantic_invalidations: int = 0


class _Evaluator:
    def __init__(self, model: Model, offsets: dict[str, tuple[int, int]]) -> None:
        self.model = model
        self.offsets = offsets
        self._cache: dict[int, np.ndarray] = {}

    def evaluate(self, node: ExprNode) -> np.ndarray:
        cache_key = id(node)
        if cache_key in self._cache:
            return self._cache[cache_key]
        if node.kind == "constant":
            raw = np.asarray(node.payload, dtype=np.float64)
            out = np.empty(raw.shape if raw.shape else (), dtype=object)
            if raw.shape:
                for idx in np.ndindex(raw.shape):
                    out[idx] = _Polynomial.scalar(float(raw[idx]))
            else:
                out[()] = _Polynomial.scalar(float(raw))
        elif node.kind == "parameter":
            raw = np.asarray(self.model._parameters[node.payload].value, dtype=np.float64)
            out = np.empty(raw.shape if raw.shape else (), dtype=object)
            if raw.shape:
                for idx in np.ndindex(raw.shape):
                    out[idx] = _Polynomial.scalar(float(raw[idx]))
            else:
                out[()] = _Polynomial.scalar(float(raw))
        elif node.kind == "variable":
            data = self.model._variables[node.payload]
            start, _ = self.offsets[node.payload]
            out = np.empty(data.shape if data.shape else (), dtype=object)
            if data.shape:
                for local, idx in enumerate(np.ndindex(data.shape)):
                    out[idx] = _Polynomial.variable(start + local)
            else:
                out[()] = _Polynomial.variable(start)
        elif node.kind in {"add", "mul"}:
            a = self.evaluate(node.args[0])
            b = self.evaluate(node.args[1])
            aa, bb = np.broadcast_arrays(a, b)
            out = np.empty(aa.shape, dtype=object)
            for idx in np.ndindex(aa.shape):
                out[idx] = aa[idx] + bb[idx] if node.kind == "add" else aa[idx] * bb[idx]
        elif node.kind == "neg":
            a = self.evaluate(node.args[0])
            out = np.empty(a.shape, dtype=object)
            if a.shape:
                for idx in np.ndindex(a.shape):
                    out[idx] = -a[idx]
            else:
                out[()] = -a[()]
        elif node.kind == "matmul":
            a = self.evaluate(node.args[0])
            b = self.evaluate(node.args[1])
            try:
                out = np.matmul(a, b)
            except ValueError as exc:
                raise ShapeError("matmul failed during scalarization") from exc
            out = np.asarray(out, dtype=object)
        elif node.kind == "index":
            out = np.asarray(self.evaluate(node.args[0])[node.payload], dtype=object)
        elif node.kind == "transpose":
            out = np.asarray(self.evaluate(node.args[0]).T, dtype=object)
        elif node.kind == "sum":
            a = self.evaluate(node.args[0])
            out = np.asarray(np.sum(a, axis=node.payload), dtype=object)
        elif node.kind == "concat":
            pieces = [np.asarray(self.evaluate(arg), dtype=object).reshape(-1) for arg in node.args]
            out = np.concatenate(pieces) if pieces else np.asarray([], dtype=object)
        else:
            raise CompileError(f"unsupported symbolic expression node {node.kind!r}")
        self._cache[cache_key] = out
        return out


def _scalar_polynomial(arr: np.ndarray, *, context: str) -> _Polynomial:
    arr = np.asarray(arr, dtype=object)
    if arr.shape != ():
        raise CompileError(f"{context} must be scalar, got shape {arr.shape}")
    value = arr[()]
    if not isinstance(value, _Polynomial):
        raise CompileError(f"{context} did not scalarize to a polynomial")
    return value


def _structural_target(model: Model) -> str:
    if model.objective is None:
        raise CompileError("model has no objective")
    has_indicator = False
    for constraint in model.constraints:
        degree = constraint.function.polynomial_degree
        if degree is None or degree > 1:
            raise CompileError(f"quadratic constraint {constraint.entity_id} is outside P4 scope")
        has_indicator = has_indicator or isinstance(constraint, IndicatorConstraint)
    degree = model.objective.expression.polynomial_degree
    if degree is None or degree > 2:
        raise CompileError("objective is outside quadratic P4 scope")
    if has_indicator:
        if degree > 1:
            raise CompileError("P4 indicator lowering currently targets legacy MILP and does not support quadratic objectives")
        return "legacy-linear-v1"
    if degree <= 1:
        return "legacy-linear-v1"
    if model.objective.sense is ObjectiveSense.MAXIMIZE:
        raise CompileError("P4 quadratic compilation supports convex minimization only")
    if any(data.domain is not VariableDomain.CONTINUOUS for data in model._variables.values()):
        raise CompileError("P4 quadratic compilation supports continuous variables only")
    return "legacy-qp-v1"


def _compilation_hash(model: Model, target: str) -> str:
    # P0 RFC-004: CompilationHash is plan identity, not execution-data identity.
    payload = {
        "semantic_hash": model.semantic_hash,
        "target": target,
        "capability_signature": _CAPABILITY_SIGNATURE,
        "numerical_policy": _NUMERICAL_POLICY,
        "transformation_tape": [],
        "precondition_certificate_hashes": [],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _variable_layout(model: Model) -> tuple[dict[str, tuple[int, int]], int, np.ndarray, np.ndarray, tuple[str, ...], dict[str, Any]]:
    offsets: dict[str, tuple[int, int]] = {}
    n = 0
    var_lower: list[float] = []
    var_upper: list[float] = []
    domains: list[str] = []
    source_vars: dict[str, Any] = {}
    for key, data in model._variables.items():
        size = int(np.prod(data.shape, dtype=np.int64)) if data.shape else 1
        offsets[key] = (n, n + size)
        source_vars[key] = {"execution_slice": [n, n + size], "shape": list(data.shape)}
        var_lower.extend(np.asarray(data.lower).reshape(-1).tolist())
        var_upper.extend(np.asarray(data.upper).reshape(-1).tolist())
        domains.extend([data.domain.value] * size)
        n += size
    return (
        offsets,
        n,
        np.asarray(var_lower, dtype=np.float64),
        np.asarray(var_upper, dtype=np.float64),
        tuple(domains),
        source_vars,
    )


def _constraint_block(evaluator: _Evaluator, constraint: Constraint) -> tuple[list[dict[int, float]], list[float], list[float]]:
    arr = np.asarray(evaluator.evaluate(constraint.function._node), dtype=object)
    rows: list[dict[int, float]] = []
    cl: list[float] = []
    cu: list[float] = []
    indices = list(np.ndindex(arr.shape)) if arr.shape else [()]
    for idx in indices:
        poly = arr[idx]
        if not isinstance(poly, _Polynomial) or poly.degree > 1:
            raise CompileError(f"quadratic constraint {constraint.entity_id} is outside P2 scope")
        rows.append(dict(poly.linear))
        if isinstance(constraint.set, LessThan):
            cl.append(-np.inf)
            cu.append(float(constraint.set.upper) - poly.constant)
        elif isinstance(constraint.set, GreaterThan):
            cl.append(float(constraint.set.lower) - poly.constant)
            cu.append(np.inf)
        elif isinstance(constraint.set, EqualTo):
            rhs = float(constraint.set.value) - poly.constant
            cl.append(rhs)
            cu.append(rhs)
        elif isinstance(constraint.set, Interval):
            cl.append(float(constraint.set.lower) - poly.constant)
            cu.append(float(constraint.set.upper) - poly.constant)
        else:
            raise CompileError(f"unsupported constraint set {type(constraint.set).__name__}")
    return rows, cl, cu


def _rows_to_csr(rows: list[dict[int, float]], n: int) -> sparse.csr_matrix:
    if not rows:
        return sparse.csr_matrix((0, n), dtype=np.float64)
    rr: list[int] = []
    cc: list[int] = []
    vv: list[float] = []
    for r, coeffs in enumerate(rows):
        for c, v in coeffs.items():
            if v != 0.0:
                rr.append(r)
                cc.append(c)
                vv.append(float(v))
    return sparse.csr_matrix((vv, (rr, cc)), shape=(len(rows), n), dtype=np.float64)


def _objective_data(evaluator: _Evaluator, model: Model, n: int, target: str) -> tuple[np.ndarray, float, sparse.csr_matrix | None]:
    if model.objective is None:
        raise CompileError("model objective is required before objective compilation")
    obj = _scalar_polynomial(evaluator.evaluate(model.objective.expression._node), context="objective")
    c = np.zeros(n, dtype=np.float64)
    for i, v in obj.linear.items():
        c[i] = v
    if target == "legacy-linear-v1":
        # Structural target is authoritative. A structural affine objective cannot
        # become quadratic solely because parameter values change.
        if obj.quadratic:
            raise CompileError("affine semantic objective evaluated to quadratic data")
        return c, obj.constant, None

    pr: list[int] = []
    pc: list[int] = []
    pv: list[float] = []
    for (i, j), q in obj.quadratic.items():
        if i == j:
            val = 2.0 * q
            if val != 0.0:
                pr.append(i)
                pc.append(i)
                pv.append(val)
        elif q != 0.0:
            pr.extend([i, j])
            pc.extend([j, i])
            pv.extend([q, q])
    P = sparse.csr_matrix((pv, (pr, pc)), shape=(n, n), dtype=np.float64)
    return c, obj.constant, P


def _make_metadata(model: Model, *, compiled_by: str = "solverpilot-p2", compiler_policy: str = _NUMERICAL_POLICY) -> dict[str, Any]:
    return {
        "semantic_hash": model.semantic_hash,
        "semantic_model_schema": "solverpilot.semantic-model.p1.v1",
        "compiled_by": compiled_by,
        "compiler_policy": compiler_policy,
    }


def _build_execution_ir(
    *,
    model: Model,
    target: str,
    A: sparse.csr_matrix,
    c: np.ndarray,
    P: sparse.csr_matrix | None,
    variable_lower: np.ndarray,
    variable_upper: np.ndarray,
    constraint_lower: np.ndarray,
    constraint_upper: np.ndarray,
    domains: tuple[str, ...],
    objective_offset: float,
    previous: LinearProblem | QuadraticProblem | None = None,
    objective_recompiled: bool = True,
    metadata: dict[str, Any] | None = None,
) -> LinearProblem | QuadraticProblem:
    if model.objective is None:
        raise CompileError("model objective is required before execution IR construction")
    metadata = _make_metadata(model) if metadata is None else metadata
    if target == "legacy-linear-v1":
        return LinearProblem.from_data(
            A=A,
            c=c,
            variable_lower=variable_lower,
            variable_upper=variable_upper,
            constraint_lower=constraint_lower,
            constraint_upper=constraint_upper,
            domains=domains,
            objective_sense=model.objective.sense,
            objective_offset=objective_offset,
            name=model.name,
            metadata=metadata,
        )

    if P is None:
        raise CompileError("P2 QP target requires a quadratic matrix")
    # If the quadratic objective was not touched, preserve the already verified P
    # and convexity status to avoid a redundant eigenvalue check.
    if isinstance(previous, QuadraticProblem) and not objective_recompiled and P is previous.P:
        linear = LinearProblem.from_data(
            A=A,
            c=c,
            variable_lower=variable_lower,
            variable_upper=variable_upper,
            constraint_lower=constraint_lower,
            constraint_upper=constraint_upper,
            objective_sense=model.objective.sense,
            objective_offset=objective_offset,
            name=model.name,
            metadata=metadata,
        )
        return QuadraticProblem(
            linear=linear,
            P=previous.P,
            convexity_status=previous.convexity_status,
            metadata=metadata,
        )
    return QuadraticProblem.from_data(
        P=P,
        A=A,
        q=c,
        variable_lower=variable_lower,
        variable_upper=variable_upper,
        constraint_lower=constraint_lower,
        constraint_upper=constraint_upper,
        objective_sense=model.objective.sense,
        objective_offset=objective_offset,
        verify_convexity=True,
        metadata=metadata,
    )


def _build_template(model: Model, target: str, offsets: dict[str, tuple[int, int]], n: int, vl: np.ndarray, vu: np.ndarray, domains: tuple[str, ...], source_vars: dict[str, Any], source_constraints: dict[str, Any]) -> _Template:
    constraint_by_id = {c.entity_id.value: c for c in model.constraints}
    constraint_dependencies = {
        cid: frozenset(c.function.parameter_dependencies)
        for cid, c in constraint_by_id.items()
    }
    objective_dependencies = frozenset(model.objective.expression.parameter_dependencies if model.objective is not None else ())
    parameter_to_constraints: dict[str, tuple[str, ...]] = {}
    parameter_affects_objective: dict[str, bool] = {}
    for pid in model._parameters:
        parameter_to_constraints[pid] = tuple(
            cid for cid, deps in constraint_dependencies.items() if pid in deps
        )
        parameter_affects_objective[pid] = pid in objective_dependencies
    source_map = {
        "variables": source_vars,
        "constraints": source_constraints,
        "objective": {} if model.objective is None else {model.objective.entity_id.value: {"execution_objective": True}},
    }
    return _Template(
        semantic_hash=model.semantic_hash,
        target=target,
        offsets=offsets,
        n_variables=n,
        variable_lower=vl,
        variable_upper=vu,
        domains=domains,
        source_map=source_map,
        constraint_by_id=constraint_by_id,
        constraint_dependencies=constraint_dependencies,
        objective_dependencies=objective_dependencies,
        parameter_to_constraints=parameter_to_constraints,
        parameter_affects_objective=parameter_affects_objective,
        compilation_hash=_compilation_hash(model, target),
    )


def _parameter_map(model: Model, template: _Template) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pid, data in model._parameters.items():
        constraints = list(template.parameter_to_constraints.get(pid, ()))
        objective = bool(template.parameter_affects_objective.get(pid, False))
        regions = [f"constraint:{cid}" for cid in constraints]
        if objective:
            regions.append("objective")
        out[pid] = {
            "shape": list(data.shape),
            "patchable": bool(regions),
            "parameter_compile_class": "partial_recompile_p2" if regions else "unused_parameter_p2",
            "plan_sensitive": False,
            "constraint_ids": constraints,
            "objective": objective,
            "execution_regions": regions,
        }
    return out


def _mutation_plan(changed_parameters: tuple[str, ...], parameter_map: dict[str, Any], mutations: tuple[str, ...]) -> dict[str, Any]:
    affected_regions: list[str] = []
    for pid in changed_parameters:
        for region in parameter_map.get(pid, {}).get("execution_regions", []):
            if region not in affected_regions:
                affected_regions.append(region)
    return {
        "mode": "partial_recompile_p2",
        "incremental_parameter_patching": True,
        "backend_session_patch": False,
        "changed_parameters": list(changed_parameters),
        "affected_regions": affected_regions,
        "execution_mutations": list(mutations),
    }


def _full_compile(model: Model, *, cache_status: str) -> tuple[CompiledModel, _Template]:
    total_t0 = perf_counter()
    target = _structural_target(model)
    analysis_t0 = perf_counter()
    offsets, n, vl, vu, domains, source_vars = _variable_layout(model)
    semantic_analysis_s = perf_counter() - analysis_t0

    lower_t0 = perf_counter()
    evaluator = _Evaluator(model, offsets)
    rows: list[dict[int, float]] = []
    cl: list[float] = []
    cu: list[float] = []
    source_constraints: dict[str, Any] = {}
    for constraint in model.constraints:
        block_rows, block_cl, block_cu = _constraint_block(evaluator, constraint)
        start = len(rows)
        rows.extend(block_rows)
        cl.extend(block_cl)
        cu.extend(block_cu)
        source_constraints[constraint.entity_id.value] = {
            "execution_rows": list(range(start, len(rows))),
            "shape": list(constraint.function.shape),
        }
    A = _rows_to_csr(rows, n)
    c, offset, P = _objective_data(evaluator, model, n, target)
    ir = _build_execution_ir(
        model=model,
        target=target,
        A=A,
        c=c,
        P=P,
        variable_lower=vl,
        variable_upper=vu,
        constraint_lower=np.asarray(cl, dtype=np.float64),
        constraint_upper=np.asarray(cu, dtype=np.float64),
        domains=domains,
        objective_offset=offset,
        previous=None,
        objective_recompiled=True,
    )
    lowering_s = perf_counter() - lower_t0
    template = _build_template(model, target, offsets, n, vl, vu, domains, source_vars, source_constraints)
    pmap = _parameter_map(model, template)
    report = CompilationReport(
        target=target,
        semantic_analysis_s=semantic_analysis_s,
        lowering_s=lowering_s,
        total_s=perf_counter() - total_t0,
        n_semantic_variables=len(model._variables),
        n_execution_variables=n,
        n_semantic_constraints=len(model._constraints),
        n_execution_constraints=len(rows),
        cache_status=cache_status,
        reused_execution_rows=0,
        recompiled_execution_rows=len(rows),
        objective_recompiled=True,
    )
    compiled = CompiledModel(
        execution_ir=ir,
        semantic_hash=model.semantic_hash,
        data_hash=model.data_hash,
        compilation_hash=template.compilation_hash,
        parameter_map=pmap,
        source_map=template.source_map,
        transformation_tape=(),
        capability_signature=_CAPABILITY_SIGNATURE,
        mutation_plan=_mutation_plan((), pmap, ()),
        plan_sensitive_dependencies=(),
        precondition_certificate_hashes=(),
        compilation_report=report,
        reconstruction_contract={"primal": "identity_by_source_map", "dual": "legacy_row_mapping", "certificate": "none_p2"},
    )
    return compiled, template


def _changed_parameters(model: Model, entry: _CacheEntry) -> tuple[str, ...]:
    return tuple(
        pid for pid in model._parameters
        if model._parameter_versions.get(pid, 0) != entry.parameter_versions.get(pid, 0)
    )


def _snapshot_hit(snapshot: CompiledModel, changed: tuple[str, ...]) -> CompiledModel:
    report = replace(
        snapshot.compilation_report,
        semantic_analysis_s=0.0,
        lowering_s=0.0,
        total_s=0.0,
        cache_status="snapshot_hit",
        changed_parameters=changed,
        reused_execution_rows=snapshot.compilation_report.n_execution_constraints,
        recompiled_execution_rows=0,
        objective_recompiled=False,
        execution_mutations=(),
    )
    return replace(snapshot, compilation_report=report)


def _incremental_compile(model: Model, entry: _CacheEntry, changed: tuple[str, ...]) -> CompiledModel:
    total_t0 = perf_counter()
    template = entry.template
    old = entry.current.execution_ir
    old_linear = old.linear if isinstance(old, QuadraticProblem) else old
    affected_constraints: set[str] = set()
    objective_recompiled = False
    for pid in changed:
        affected_constraints.update(template.parameter_to_constraints.get(pid, ()))
        objective_recompiled = objective_recompiled or template.parameter_affects_objective.get(pid, False)

    lower_t0 = perf_counter()
    A_lil = old_linear.A.tolil(copy=True)
    cl = np.array(old_linear.constraint_lower, copy=True)
    cu = np.array(old_linear.constraint_upper, copy=True)
    evaluator = _Evaluator(model, template.offsets)
    recompiled_rows = 0
    for cid in sorted(affected_constraints):
        constraint = template.constraint_by_id[cid]
        block_rows, block_cl, block_cu = _constraint_block(evaluator, constraint)
        execution_rows = template.source_map["constraints"][cid]["execution_rows"]
        if len(block_rows) != len(execution_rows):
            raise CompileError("parameter update changed execution row cardinality; full semantic recompile required")
        for local, row_index in enumerate(execution_rows):
            coeffs = block_rows[local]
            cols = sorted(k for k, v in coeffs.items() if v != 0.0)
            A_lil.rows[row_index] = cols
            A_lil.data[row_index] = [float(coeffs[k]) for k in cols]
            cl[row_index] = block_cl[local]
            cu[row_index] = block_cu[local]
            recompiled_rows += 1
    A = A_lil.tocsr()

    if isinstance(old, QuadraticProblem):
        old_c = old.linear.c
        old_offset = old.linear.objective_offset
        old_P: sparse.csr_matrix | None = old.P
    else:
        old_c = old.c
        old_offset = old.objective_offset
        old_P = None

    if objective_recompiled:
        c, offset, P = _objective_data(evaluator, model, template.n_variables, template.target)
    else:
        c = old_c
        offset = old_offset
        P = old_P

    new_ir = _build_execution_ir(
        model=model,
        target=template.target,
        A=A,
        c=c,
        P=P,
        variable_lower=template.variable_lower,
        variable_upper=template.variable_upper,
        constraint_lower=cl,
        constraint_upper=cu,
        domains=template.domains,
        objective_offset=offset,
        previous=old,
        objective_recompiled=objective_recompiled,
    )
    lowering_s = perf_counter() - lower_t0
    record = classify_mutations(old, new_ir, revision_before=0, revision_after=1)
    mutations = tuple(kind.value for kind in record.kinds)
    pmap = _parameter_map(model, template)
    report = CompilationReport(
        target=template.target,
        semantic_analysis_s=0.0,
        lowering_s=lowering_s,
        total_s=perf_counter() - total_t0,
        n_semantic_variables=len(model._variables),
        n_execution_variables=template.n_variables,
        n_semantic_constraints=len(model._constraints),
        n_execution_constraints=old_linear.n_constraints,
        cache_status="partial_recompile",
        changed_parameters=changed,
        reused_execution_rows=old_linear.n_constraints - recompiled_rows,
        recompiled_execution_rows=recompiled_rows,
        objective_recompiled=objective_recompiled,
        execution_mutations=mutations,
    )
    return CompiledModel(
        execution_ir=new_ir,
        semantic_hash=model.semantic_hash,
        data_hash=model.data_hash,
        compilation_hash=template.compilation_hash,
        parameter_map=pmap,
        source_map=template.source_map,
        transformation_tape=(),
        capability_signature=_CAPABILITY_SIGNATURE,
        mutation_plan=_mutation_plan(changed, pmap, mutations),
        plan_sensitive_dependencies=(),
        precondition_certificate_hashes=(),
        compilation_report=report,
        reconstruction_contract={"primal": "identity_by_source_map", "dual": "legacy_row_mapping", "certificate": "none_p2"},
    )


def _remember_snapshot(entry: _CacheEntry, compiled: CompiledModel) -> None:
    entry.snapshots[compiled.data_hash] = compiled
    entry.snapshots.move_to_end(compiled.data_hash)
    while len(entry.snapshots) > _MAX_SNAPSHOTS:
        entry.snapshots.popitem(last=False)



@dataclass(slots=True)
class _BridgeCacheEntry:
    semantic_hash: str
    current: CompiledModel
    parameter_versions: dict[str, int]
    snapshots: OrderedDict[str, CompiledModel]
    full_compiles: int = 1
    snapshot_hits: int = 0


def _p4_capability_signature(capabilities: BackendCapabilityManifestV2 | None) -> str:
    if capabilities is None:
        return _P4_CAPABILITY_SIGNATURE
    return f"backend-capability-v2:{capabilities.signature}:legacy-indicator-transport=0"


def _p4_compilation_hash(
    model: Model,
    target: str,
    *,
    capability_signature: str,
    bridge_policy: BridgePolicy,
    tape: TransformationTape,
) -> str:
    payload = {
        "semantic_hash": model.semantic_hash,
        "target": target,
        "capability_signature": capability_signature,
        "numerical_policy": _P4_NUMERICAL_POLICY,
        "bridge_policy": bridge_policy.policy_version,
        "transformation_tape": tape.to_dicts(),
        "precondition_certificate_hashes": list(tape.certificate_hashes),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()


def _p4_parameter_map(model: Model, tape: TransformationTape) -> dict[str, Any]:
    plan_sensitive = set(tape.plan_sensitive_dependencies)
    out: dict[str, Any] = {}
    for pid, data in model._parameters.items():
        constraint_ids = [
            c.entity_id.value for c in model.constraints
            if pid in c.function.parameter_dependencies
        ]
        objective = bool(model.objective is not None and pid in model.objective.expression.parameter_dependencies)
        regions = [f"constraint:{cid}" for cid in constraint_ids]
        if objective:
            regions.append("objective")
        sensitive = pid in plan_sensitive
        out[pid] = {
            "shape": list(data.shape),
            "patchable": bool(regions) and not sensitive,
            "parameter_compile_class": "plan_sensitive_full_recompile_p4" if sensitive else "partial_recompile_p2",
            "plan_sensitive": sensitive,
            "constraint_ids": constraint_ids,
            "objective": objective,
            "execution_regions": regions,
        }
    return out


def _indicator_block_p4(
    evaluator: _Evaluator,
    constraint: IndicatorConstraint,
    *,
    offsets: dict[str, tuple[int, int]],
    variable_lower: np.ndarray,
    variable_upper: np.ndarray,
    domains: tuple[str, ...],
    capabilities: BackendCapabilityManifestV2 | None,
    bridge_policy: BridgePolicy,
):
    arr = np.asarray(evaluator.evaluate(constraint.function._node), dtype=object)
    poly = _scalar_polynomial(arr, context=f"indicator body {constraint.entity_id.value}")
    if poly.degree > 1:
        raise CompileError("P4 indicator body must be affine")
    start, end = offsets[constraint.indicator.id.value]
    if end - start != 1:
        raise CompileError("P4 indicator variable must lower to exactly one execution variable")
    z_index = start
    decision = choose_indicator_path(
        capabilities,
        native_transport_available=False,
        policy=bridge_policy,
    )
    try:
        lowered = lower_indicator_big_m(
            source_id=constraint.entity_id.value,
            indicator_index=z_index,
            active_value=constraint.active_value,
            indicator_domain=VariableDomain(domains[z_index]),
            indicator_lower=float(variable_lower[z_index]),
            indicator_upper=float(variable_upper[z_index]),
            body=AffineScalar(poly.constant, dict(poly.linear)),
            body_set=constraint.set,
            variable_lower=variable_lower,
            variable_upper=variable_upper,
            parameter_dependencies=tuple(sorted(constraint.function.parameter_dependencies)),
            policy=bridge_policy,
        )
    except BridgeUnavailable as exc:
        raise CompileError(f"indicator bridge unavailable for {constraint.entity_id.value}: {exc}") from exc
    return lowered, decision


def _full_compile_p4(
    model: Model,
    *,
    cache_status: str,
    changed_parameters: tuple[str, ...],
    bridge_policy: BridgePolicy,
    capabilities: BackendCapabilityManifestV2 | None,
) -> CompiledModel:
    total_t0 = perf_counter()
    target = _structural_target(model)
    if target != "legacy-linear-v1":
        raise CompileError("P4 bridge compiler currently targets legacy linear/MILP execution IR")
    analysis_t0 = perf_counter()
    offsets, n, vl, vu, domains, source_vars = _variable_layout(model)
    semantic_analysis_s = perf_counter() - analysis_t0

    lower_t0 = perf_counter()
    evaluator = _Evaluator(model, offsets)
    rows: list[dict[int, float]] = []
    cl: list[float] = []
    cu: list[float] = []
    source_constraints: dict[str, Any] = {}
    tape_steps = []
    rejected_paths: list[str] = []
    bridge_search_s = 0.0
    for constraint in model.constraints:
        start_row = len(rows)
        if isinstance(constraint, IndicatorConstraint):
            bt0 = perf_counter()
            lowered, decision = _indicator_block_p4(
                evaluator,
                constraint,
                offsets=offsets,
                variable_lower=vl,
                variable_upper=vu,
                domains=domains,
                capabilities=capabilities,
                bridge_policy=bridge_policy,
            )
            bridge_search_s += perf_counter() - bt0
            rows.extend(dict(r) for r in lowered.rows)
            cl.extend(lowered.lower)
            cu.extend(lowered.upper)
            tape_steps.append(lowered.step)
            rejected_paths.extend(decision.rejected_paths)
            source_constraints[constraint.entity_id.value] = {
                "execution_rows": list(range(start_row, len(rows))),
                "shape": [],
                "semantic_kind": "indicator",
                "bridge_step_id": lowered.step.step_id,
                "generated_target_ids": list(lowered.step.generated_target_ids),
            }
        else:
            block_rows, block_cl, block_cu = _constraint_block(evaluator, constraint)
            rows.extend(block_rows)
            cl.extend(block_cl)
            cu.extend(block_cu)
            source_constraints[constraint.entity_id.value] = {
                "execution_rows": list(range(start_row, len(rows))),
                "shape": list(constraint.function.shape),
                "semantic_kind": "function_in_set",
            }
    A = _rows_to_csr(rows, n)
    c, offset, P = _objective_data(evaluator, model, n, target)
    if P is not None:
        raise CompileError("P4 indicator bridge unexpectedly produced a quadratic target")
    tape = TransformationTape(tuple(tape_steps))
    capability_signature = _p4_capability_signature(capabilities)
    compilation_hash = _p4_compilation_hash(
        model,
        target,
        capability_signature=capability_signature,
        bridge_policy=bridge_policy,
        tape=tape,
    )
    metadata = _make_metadata(model, compiled_by="solverpilot-p4", compiler_policy=_P4_NUMERICAL_POLICY)
    metadata.update({
        "compilation_hash": compilation_hash,
        "transformation_count": len(tape.steps),
        "bridge_policy": bridge_policy.policy_version,
    })
    ir = _build_execution_ir(
        model=model,
        target=target,
        A=A,
        c=c,
        P=None,
        variable_lower=vl,
        variable_upper=vu,
        constraint_lower=np.asarray(cl, dtype=np.float64),
        constraint_upper=np.asarray(cu, dtype=np.float64),
        domains=domains,
        objective_offset=offset,
        previous=None,
        objective_recompiled=True,
        metadata=metadata,
    )
    lowering_s = perf_counter() - lower_t0
    pmap = _p4_parameter_map(model, tape)
    source_map = {"variables": source_vars, "constraints": source_constraints}
    report = CompilationReport(
        target=target,
        semantic_analysis_s=semantic_analysis_s,
        lowering_s=lowering_s,
        total_s=perf_counter() - total_t0,
        n_semantic_variables=len(model._variables),
        n_execution_variables=n,
        n_semantic_constraints=len(model._constraints),
        n_execution_constraints=len(rows),
        cache_status=cache_status,
        changed_parameters=changed_parameters,
        reused_execution_rows=0,
        recompiled_execution_rows=len(rows),
        objective_recompiled=True,
        execution_mutations=(),
        bridge_search_s=bridge_search_s,
        n_transformations=len(tape.steps),
        rejected_bridge_paths=tuple(rejected_paths),
    )
    return CompiledModel(
        execution_ir=ir,
        semantic_hash=model.semantic_hash,
        data_hash=model.data_hash,
        compilation_hash=compilation_hash,
        parameter_map=pmap,
        source_map=source_map,
        transformation_tape=tape.to_dicts(),
        capability_signature=capability_signature,
        mutation_plan={
            "mode": "bridge_full_recompile_p4",
            "incremental_parameter_patching": False,
            "backend_session_patch": False,
            "changed_parameters": list(changed_parameters),
            "affected_regions": [
                region for pid in changed_parameters for region in pmap.get(pid, {}).get("execution_regions", [])
            ],
            "execution_mutations": [],
            "reason": "P4 plan-sensitive certificates are revalidated by full semantic lowering",
        },
        plan_sensitive_dependencies=tape.plan_sensitive_dependencies,
        precondition_certificate_hashes=tape.certificate_hashes,
        compilation_report=report,
        reconstruction_contract={
            "primal": "identity_by_source_map",
            "dual": "unavailable_for_indicator_bridges",
            "certificate": "unavailable_for_indicator_bridges",
            "original_space_validation": "solverpilot.bridges.validate_semantic_primal",
        },
        schema_version="solverpilot.compiled-model.p4.v1",
    )


def _p4_cache_key(capabilities: BackendCapabilityManifestV2 | None, bridge_policy: BridgePolicy) -> str:
    return f"p4|{_p4_capability_signature(capabilities)}|{bridge_policy.policy_version}"


def _compile_model_p4(
    model: Model,
    *,
    use_cache: bool,
    bridge_policy: BridgePolicy,
    capabilities: BackendCapabilityManifestV2 | None,
) -> CompiledModel:
    key = _p4_cache_key(capabilities, bridge_policy)
    entry = model._compiler_cache.get(key)
    changed: tuple[str, ...] = ()
    if isinstance(entry, _BridgeCacheEntry):
        changed = tuple(
            pid for pid in model._parameters
            if model._parameter_versions.get(pid, 0) != entry.parameter_versions.get(pid, 0)
        )
    if not use_cache:
        return _full_compile_p4(
            model,
            cache_status="disabled",
            changed_parameters=changed,
            bridge_policy=bridge_policy,
            capabilities=capabilities,
        )
    semantic_hash = model.semantic_hash
    if entry is None or not isinstance(entry, _BridgeCacheEntry) or entry.semantic_hash != semantic_hash:
        compiled = _full_compile_p4(
            model,
            cache_status="miss",
            changed_parameters=(),
            bridge_policy=bridge_policy,
            capabilities=capabilities,
        )
        snapshots: OrderedDict[str, CompiledModel] = OrderedDict([(compiled.data_hash, compiled)])
        model._compiler_cache[key] = _BridgeCacheEntry(
            semantic_hash=semantic_hash,
            current=compiled,
            parameter_versions=dict(model._parameter_versions),
            snapshots=snapshots,
        )
        return compiled
    data_hash = model.data_hash
    if data_hash in entry.snapshots:
        entry.snapshot_hits += 1
        snap = entry.snapshots[data_hash]
        hit = replace(
            snap,
            compilation_report=replace(
                snap.compilation_report,
                semantic_analysis_s=0.0,
                lowering_s=0.0,
                total_s=0.0,
                bridge_search_s=0.0,
                cache_status="snapshot_hit",
                changed_parameters=changed,
                reused_execution_rows=snap.compilation_report.n_execution_constraints,
                recompiled_execution_rows=0,
                objective_recompiled=False,
            ),
        )
        entry.current = hit
        entry.parameter_versions = dict(model._parameter_versions)
        entry.snapshots[data_hash] = hit
        entry.snapshots.move_to_end(data_hash)
        return hit
    compiled = _full_compile_p4(
        model,
        cache_status="plan_sensitive_recompile",
        changed_parameters=changed,
        bridge_policy=bridge_policy,
        capabilities=capabilities,
    )
    entry.full_compiles += 1
    entry.current = compiled
    entry.parameter_versions = dict(model._parameter_versions)
    entry.snapshots[data_hash] = compiled
    entry.snapshots.move_to_end(data_hash)
    while len(entry.snapshots) > _MAX_SNAPSHOTS:
        entry.snapshots.popitem(last=False)
    return compiled

def compile_model(
    model: Model,
    *,
    use_cache: bool = True,
    bridge_policy: BridgePolicy | None = None,
    capabilities: BackendCapabilityManifestV2 | None = None,
) -> CompiledModel:
    from .sets import PositiveSemidefiniteCone, RotatedSecondOrderCone, SecondOrderCone
    if any(isinstance(c.set, (SecondOrderCone, RotatedSecondOrderCone, PositiveSemidefiniteCone)) for c in model.constraints):
        if any(isinstance(c, IndicatorConstraint) for c in model.constraints):
            raise CompileError("P6 does not combine indicator/MILP bridges with continuous conic targets")
        from solverpilot.conic.compiler import compile_conic_model
        return compile_conic_model(model, use_cache=use_cache)
    def _node_has_p7_nonlinearity(node):
        if node.kind in {"sin", "cos", "exp", "log", "sqrt", "tanh", "pow", "div"}:
            return True
        return any(_node_has_p7_nonlinearity(a) for a in node.args)

    has_nlp = False
    has_discrete = any(data.domain is not VariableDomain.CONTINUOUS for data in model._variables.values())
    if model.objective is not None:
        deg = model.objective.expression.polynomial_degree
        has_nlp = _node_has_p7_nonlinearity(model.objective.expression._node) or deg is None or (deg is not None and deg > 2) or (has_discrete and deg is not None and deg > 1)
    if not has_nlp:
        for c in model.constraints:
            if isinstance(c, IndicatorConstraint):
                continue
            deg = c.function.polynomial_degree
            if _node_has_p7_nonlinearity(c.function._node) or deg is None or (deg is not None and deg > 1):
                has_nlp = True; break
    if has_nlp:
        if any(isinstance(c, IndicatorConstraint) for c in model.constraints):
            raise CompileError("P8 MINLP v1 does not combine indicator constraints with nonlinear orchestration")
        if any(data.domain is not VariableDomain.CONTINUOUS for data in model._variables.values()):
            from solverpilot.minlp.compiler import compile_minlp_model
            return compile_minlp_model(model, use_cache=use_cache)
        from solverpilot.nlp.compiler import compile_nlp_model
        return compile_nlp_model(model, use_cache=use_cache)

    if any(isinstance(c, IndicatorConstraint) for c in model.constraints):
        return _compile_model_p4(
            model,
            use_cache=use_cache,
            bridge_policy=BridgePolicy() if bridge_policy is None else bridge_policy,
            capabilities=capabilities,
        )
    if not use_cache:
        compiled, _ = _full_compile(model, cache_status="disabled")
        return compiled

    semantic_hash = model.semantic_hash
    entry = model._compiler_cache.get(_CACHE_KEY)
    if entry is None or not isinstance(entry, _CacheEntry) or entry.semantic_hash != semantic_hash:
        invalidations = 0 if entry is None or not isinstance(entry, _CacheEntry) else entry.semantic_invalidations + 1
        compiled, template = _full_compile(model, cache_status="miss")
        snapshots: OrderedDict[str, CompiledModel] = OrderedDict()
        snapshots[compiled.data_hash] = compiled
        model._compiler_cache[_CACHE_KEY] = _CacheEntry(
            semantic_hash=semantic_hash,
            template=template,
            current=compiled,
            parameter_versions=dict(model._parameter_versions),
            snapshots=snapshots,
            semantic_invalidations=invalidations,
        )
        return compiled

    changed = _changed_parameters(model, entry)
    data_hash = model.data_hash
    if data_hash in entry.snapshots:
        entry.snapshot_hits += 1
        hit = _snapshot_hit(entry.snapshots[data_hash], changed)
        entry.current = hit
        entry.parameter_versions = dict(model._parameter_versions)
        entry.snapshots[data_hash] = hit
        entry.snapshots.move_to_end(data_hash)
        return hit

    compiled = _incremental_compile(model, entry, changed)
    entry.partial_compiles += 1
    entry.current = compiled
    entry.parameter_versions = dict(model._parameter_versions)
    _remember_snapshot(entry, compiled)
    return compiled


def compiler_cache_info(model: Model) -> CompilerCacheInfo:
    entry = model._compiler_cache.get(_CACHE_KEY)
    if not isinstance(entry, _CacheEntry):
        return CompilerCacheInfo(None, 0, 0, 0, 0, 0, None)
    return CompilerCacheInfo(
        semantic_hash=entry.semantic_hash,
        snapshots=len(entry.snapshots),
        full_compiles=entry.full_compiles,
        partial_compiles=entry.partial_compiles,
        snapshot_hits=entry.snapshot_hits,
        semantic_invalidations=entry.semantic_invalidations,
        last_data_hash=entry.current.data_hash,
    )
