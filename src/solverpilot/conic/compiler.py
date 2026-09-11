from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import hashlib
import json
from time import perf_counter
from typing import Any

import numpy as np
from scipy import sparse

from solverpilot.problem import ObjectiveSense, VariableDomain
from solverpilot.model.errors import CompileError
from solverpilot.model.sets import (
    EqualTo, GreaterThan, Interval, LessThan,
    ExponentialCone, PowerCone, PositiveSemidefiniteCone, RotatedSecondOrderCone, SecondOrderCone,
)

from .ir import ConeAffineBlock, ConeKind, ConicProblem


_P6_CACHE_KEY = "conic-ir:p6:v1"
_P6_CAPABILITY_SIGNATURE = "conic-ir:p6:v1:casadi-superscs-bridge-verified-soc-rsoc-psd-unverified"
_P6_POLICY = "p0-rfc-012:p6-convex-conic-v1"
_MAX_SNAPSHOTS = 8


@dataclass(slots=True)
class _ConicCacheEntry:
    semantic_hash: str
    current: Any
    parameter_versions: dict[str, int]
    snapshots: OrderedDict[str, Any]
    full_compiles: int = 1
    snapshot_hits: int = 0


def _compilation_hash(model) -> str:
    payload = {
        "semantic_hash": model.semantic_hash,
        "target": "conic-ir-v1",
        "capability_signature": _P6_CAPABILITY_SIGNATURE,
        "numerical_policy": _P6_POLICY,
        "cone_semantics": ["soc", "rotated_soc", "psd"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _poly_to_affine(poly, n: int, *, context: str) -> tuple[dict[int, float], float]:
    if poly.degree > 1:
        raise CompileError(f"{context} must be affine for P6 conic lowering")
    return dict(poly.linear), float(poly.constant)


def _affine_map(evaluator, node, n: int, *, context: str) -> tuple[sparse.csr_matrix, np.ndarray, tuple[int, ...]]:
    arr = np.asarray(evaluator.evaluate(node), dtype=object)
    shape = tuple(arr.shape)
    flat = arr.reshape(-1)
    rr: list[int] = []
    cc: list[int] = []
    vv: list[float] = []
    g = np.zeros(flat.size, dtype=float)
    for r, poly in enumerate(flat):
        coeffs, const = _poly_to_affine(poly, n, context=context)
        g[r] = const
        for c, value in coeffs.items():
            if value != 0.0:
                rr.append(r); cc.append(int(c)); vv.append(float(value))
    F = sparse.csr_matrix((vv, (rr, cc)), shape=(flat.size, n), dtype=float)
    return F, g, shape


def _assert_psd_affine_symmetry(evaluator, node, *, context: str) -> None:
    arr = np.asarray(evaluator.evaluate(node), dtype=object)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise CompileError(f"{context} must be a square matrix")
    for i in range(arr.shape[0]):
        for j in range(i + 1, arr.shape[1]):
            if arr[i, j] != arr[j, i]:
                raise CompileError(
                    f"{context} is not structurally symmetric; P6 refuses to silently symmetrize PSD semantics"
                )


def _build_conic(model, *, cache_status: str, changed_parameters: tuple[str, ...]):
    # Import P2/P4 compiler internals only at execution time to avoid module init cycles.
    from solverpilot.model.compiler import (
        CompiledModel, CompilationReport, _Evaluator, _constraint_block,
        _objective_data, _rows_to_csr, _variable_layout,
    )

    total_t0 = perf_counter()
    if model.objective is None:
        raise CompileError("model has no objective")
    if model.objective.sense is not ObjectiveSense.MINIMIZE:
        raise CompileError("P6 conic compilation supports convex minimization only")
    if any(data.domain is not VariableDomain.CONTINUOUS for data in model._variables.values()):
        raise CompileError("P6 conic execution currently supports continuous variables only")
    if model.objective.expression.polynomial_degree is None or model.objective.expression.polynomial_degree > 2:
        raise CompileError("P6 objective must be affine or convex quadratic")

    analysis_t0 = perf_counter()
    offsets, n, vl, vu, domains, source_vars = _variable_layout(model)
    semantic_analysis_s = perf_counter() - analysis_t0
    evaluator = _Evaluator(model, offsets)

    lower_t0 = perf_counter()
    rows: list[dict[int, float]] = []
    cl: list[float] = []
    cu: list[float] = []
    cones: list[ConeAffineBlock] = []
    source_constraints: dict[str, Any] = {}
    cone_dependencies: set[str] = set()

    for constraint in model.constraints:
        cid = constraint.entity_id.value
        set_ = constraint.set
        if isinstance(set_, (LessThan, GreaterThan, EqualTo, Interval)):
            block_rows, block_cl, block_cu = _constraint_block(evaluator, constraint)
            start = len(rows)
            rows.extend(block_rows); cl.extend(block_cl); cu.extend(block_cu)
            source_constraints[cid] = {
                "semantic_kind": "function_in_scalar_set",
                "execution_rows": list(range(start, len(rows))),
                "shape": list(constraint.function.shape),
            }
            continue

        F, g, shape = _affine_map(evaluator, constraint.function._node, n, context=f"cone constraint {cid}")
        deps = tuple(sorted(constraint.function.parameter_dependencies))
        cone_dependencies.update(deps)
        if isinstance(set_, ExponentialCone):
            kind, expected = ConeKind.EXPONENTIAL, (3,)
        elif isinstance(set_, PowerCone):
            kind, expected = ConeKind.POWER, (3,)
        elif isinstance(set_, SecondOrderCone):
            kind = ConeKind.SECOND_ORDER
            expected = (set_.dimension,)
        elif isinstance(set_, RotatedSecondOrderCone):
            kind = ConeKind.ROTATED_SECOND_ORDER
            expected = (set_.dimension,)
        elif isinstance(set_, PositiveSemidefiniteCone):
            _assert_psd_affine_symmetry(evaluator, constraint.function._node, context=f"PSD constraint {cid}")
            kind = ConeKind.POSITIVE_SEMIDEFINITE
            expected = (set_.dimension, set_.dimension)
        else:
            raise CompileError(f"unsupported P6 constraint set {type(set_).__name__}")
        if shape != expected:
            raise CompileError(f"cone shape mismatch for {cid}: expected {expected}, got {shape}")
        index = len(cones)
        cones.append(ConeAffineBlock(kind, F, g, expected, source_id=cid, metadata={"parameter_dependencies": deps, **({"alpha": set_.alpha} if isinstance(set_, PowerCone) else {})}))
        source_constraints[cid] = {
            "semantic_kind": kind.value,
            "cone_block": index,
            "shape": list(shape),
            "parameter_dependencies": list(deps),
        }

    if not cones:
        raise CompileError("P6 conic compiler received a model with no cone constraints")
    A = _rows_to_csr(rows, n)
    # Reuse the already verified P2 quadratic convention 0.5*x'P*x + q'x + offset.
    target = "legacy-qp-v1" if model.objective.expression.polynomial_degree == 2 else "legacy-linear-v1"
    q, offset, P = _objective_data(evaluator, model, n, target)
    if P is None:
        P = sparse.csr_matrix((n, n), dtype=float)
    try:
        problem = ConicProblem.from_data(
            P=P,
            q=q,
            A=A,
            variable_lower=vl,
            variable_upper=vu,
            constraint_lower=np.asarray(cl, dtype=float),
            constraint_upper=np.asarray(cu, dtype=float),
            cones=cones,
            objective_offset=offset,
            name=model.name,
            metadata={
                "semantic_hash": model.semantic_hash,
                "compiled_by": "solverpilot-p6",
                "compiler_policy": _P6_POLICY,
                "cone_families": sorted({c.kind.value for c in cones}),
            },
            verify_convexity=True,
        )
    except ValueError as exc:
        raise CompileError(str(exc)) from exc

    pmap: dict[str, Any] = {}
    for pid, pdata in model._parameters.items():
        regions: list[str] = []
        for cid, info in source_constraints.items():
            if pid in info.get("parameter_dependencies", []):
                regions.append(f"cone:{cid}")
        if model.objective is not None and pid in model.objective.expression.parameter_dependencies:
            regions.append("objective")
        for c in model.constraints:
            if pid in c.function.parameter_dependencies and f"cone:{c.entity_id.value}" not in regions:
                regions.append(f"constraint:{c.entity_id.value}")
        pmap[pid] = {
            "shape": list(pdata.shape),
            "patchable": False,
            "parameter_compile_class": "p6_full_conic_relower",
            "plan_sensitive": True if regions else False,
            "execution_regions": regions,
        }
    comp_hash = _compilation_hash(model)
    source_map = {
        "variables": source_vars,
        "constraints": source_constraints,
        "objective": {model.objective.entity_id.value: {"execution_objective": True}},
    }
    lowering_s = perf_counter() - lower_t0
    report = CompilationReport(
        target="conic-ir-v1",
        semantic_analysis_s=semantic_analysis_s,
        lowering_s=lowering_s,
        total_s=perf_counter() - total_t0,
        n_semantic_variables=len(model._variables),
        n_execution_variables=n,
        n_semantic_constraints=len(model._constraints),
        n_execution_constraints=len(rows) + len(cones),
        cache_status=cache_status,
        changed_parameters=changed_parameters,
        reused_execution_rows=0,
        recompiled_execution_rows=len(rows) + len(cones),
        objective_recompiled=True,
        execution_mutations=(),
        n_transformations=0,
    )
    return CompiledModel(
        execution_ir=problem,
        semantic_hash=model.semantic_hash,
        data_hash=model.data_hash,
        compilation_hash=comp_hash,
        parameter_map=pmap,
        source_map=source_map,
        transformation_tape=(),
        capability_signature=_P6_CAPABILITY_SIGNATURE,
        mutation_plan={
            "mode": "p6_full_conic_relower",
            "incremental_parameter_patching": False,
            "backend_session_patch": False,
            "changed_parameters": list(changed_parameters),
            "affected_regions": [r for pid in changed_parameters for r in pmap.get(pid, {}).get("execution_regions", [])],
            "execution_mutations": [],
            "reason": "P6 cone maps are rebuilt from semantic affine expressions; persistent conic mutation is outside P6 scope",
        },
        plan_sensitive_dependencies=tuple(sorted(cone_dependencies)),
        precondition_certificate_hashes=(),
        compilation_report=report,
        reconstruction_contract={
            "primal": "identity_by_source_map",
            "dual": "backend_specific_not_standardized_p6",
            "certificate": "none_p6",
            "original_space_validation": "solverpilot.conic.validate_conic_solution + semantic validator",
        },
        schema_version="solverpilot.compiled-model.p6.v1",
    )


def compile_conic_model(model, *, use_cache: bool = True):
    entry = model._compiler_cache.get(_P6_CACHE_KEY)
    changed: tuple[str, ...] = ()
    if isinstance(entry, _ConicCacheEntry):
        changed = tuple(
            pid for pid in model._parameters
            if model._parameter_versions.get(pid, 0) != entry.parameter_versions.get(pid, 0)
        )
    if not use_cache:
        return _build_conic(model, cache_status="disabled", changed_parameters=changed)
    if not isinstance(entry, _ConicCacheEntry) or entry.semantic_hash != model.semantic_hash:
        compiled = _build_conic(model, cache_status="miss", changed_parameters=())
        snapshots = OrderedDict([(compiled.data_hash, compiled)])
        model._compiler_cache[_P6_CACHE_KEY] = _ConicCacheEntry(
            semantic_hash=model.semantic_hash,
            current=compiled,
            parameter_versions=dict(model._parameter_versions),
            snapshots=snapshots,
        )
        return compiled
    if model.data_hash in entry.snapshots:
        entry.snapshot_hits += 1
        snap = entry.snapshots[model.data_hash]
        hit = replace(
            snap,
            compilation_report=replace(
                snap.compilation_report,
                semantic_analysis_s=0.0,
                lowering_s=0.0,
                total_s=0.0,
                cache_status="snapshot_hit",
                changed_parameters=changed,
                reused_execution_rows=snap.compilation_report.n_execution_constraints,
                recompiled_execution_rows=0,
                objective_recompiled=False,
            ),
        )
        entry.current = hit
        entry.parameter_versions = dict(model._parameter_versions)
        entry.snapshots[model.data_hash] = hit
        entry.snapshots.move_to_end(model.data_hash)
        return hit
    compiled = _build_conic(model, cache_status="full_conic_relower", changed_parameters=changed)
    entry.full_compiles += 1
    entry.current = compiled
    entry.parameter_versions = dict(model._parameter_versions)
    entry.snapshots[compiled.data_hash] = compiled
    entry.snapshots.move_to_end(compiled.data_hash)
    while len(entry.snapshots) > _MAX_SNAPSHOTS:
        entry.snapshots.popitem(last=False)
    return compiled
