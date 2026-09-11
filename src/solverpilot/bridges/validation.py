from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from solverpilot.problem import VariableDomain

from .indicator import AffineScalar


@dataclass(frozen=True, slots=True)
class SemanticPrimalValidation:
    valid: bool
    max_violation: float
    violations: tuple[str, ...]


def _eval_expr(model, node, xflat: np.ndarray):
    from solverpilot.model.atoms import has_atoms, evaluate
    if has_atoms(node):
        return evaluate(model, node, xflat)
    from solverpilot.model.compiler import _Evaluator, _variable_layout
    offsets, *_ = _variable_layout(model)
    ev = _Evaluator(model, offsets)
    arr = ev.evaluate(node)
    out = np.empty(arr.shape, dtype=float)
    if arr.shape:
        for idx in np.ndindex(arr.shape):
            p = arr[idx]
            val = p.constant + sum(float(a) * float(xflat[i]) for i, a in p.linear.items())
            if p.quadratic:
                val += sum(float(q) * float(xflat[i]) * float(xflat[j]) for (i, j), q in p.quadratic.items())
            out[idx] = val
    else:
        p = arr[()]
        val = p.constant + sum(float(a) * float(xflat[i]) for i, a in p.linear.items())
        if p.quadratic:
            val += sum(float(q) * float(xflat[i]) * float(xflat[j]) for (i, j), q in p.quadratic.items())
        out = np.asarray(val)
    return out


def validate_semantic_primal(model, x: Any, *, atol: float = 1e-8) -> SemanticPrimalValidation:
    from solverpilot.model.model import Constraint, IndicatorConstraint
    from solverpilot.model.sets import EqualTo, GreaterThan, Interval, LessThan, PositiveSemidefiniteCone, RotatedSecondOrderCone, SecondOrderCone
    from solverpilot.model.compiler import _variable_layout

    if isinstance(atol, bool) or type(atol) not in (int, float) or not np.isfinite(float(atol)) or float(atol) < 0:
        raise ValueError("atol must be finite and non-negative")
    atol = float(atol)
    offsets, n, vl, vu, domains, _ = _variable_layout(model)
    xx = np.asarray(x, dtype=float).reshape(-1)
    if xx.shape != (n,):
        return SemanticPrimalValidation(False, float("inf"), (f"expected primal shape ({n},), got {xx.shape}",))
    if not np.all(np.isfinite(xx)):
        return SemanticPrimalValidation(False, float("inf"), ("primal solution contains NaN or infinity",))
    violations: list[str] = []
    max_v = 0.0
    for i in range(n):
        v = max(float(vl[i] - xx[i]), float(xx[i] - vu[i]), 0.0)
        if domains[i] in {VariableDomain.INTEGER.value, VariableDomain.BINARY.value}:
            v = max(v, abs(float(xx[i] - round(xx[i]))))
        if v > atol:
            violations.append(f"variable[{i}] violation={v}")
        max_v = max(max_v, v)

    def check_set(value: float, set_, label: str) -> None:
        nonlocal max_v
        if not np.isfinite(value): v = float('inf')
        elif isinstance(set_, LessThan): v = max(0.0, value - set_.upper)
        elif isinstance(set_, GreaterThan): v = max(0.0, set_.lower - value)
        elif isinstance(set_, EqualTo): v = abs(value - set_.value)
        elif isinstance(set_, Interval): v = max(0.0, set_.lower - value, value - set_.upper)
        else: v = float("inf")
        max_v = max(max_v, v)
        if v > atol: violations.append(f"{label} violation={v}")

    for constraint in model.constraints:
        if isinstance(constraint, Constraint):
            arr = np.asarray(_eval_expr(model, constraint.function._node, xx), dtype=float)
            if isinstance(constraint.set, SecondOrderCone):
                y = arr.reshape(-1); t = float(y[0]); norm = float(np.linalg.norm(y[1:])); v = max(0.0, norm - t)
                max_v = max(max_v, v)
                if v > atol: violations.append(f"constraint {constraint.entity_id.value} SOC violation={v}")
                continue
            if isinstance(constraint.set, RotatedSecondOrderCone):
                y = arr.reshape(-1); u = float(y[0]); vv = float(y[1]); norm2 = float(np.dot(y[2:], y[2:])); v = max(0.0, -u, -vv, norm2 - 2.0*u*vv)
                max_v = max(max_v, v)
                if v > atol: violations.append(f"constraint {constraint.entity_id.value} rotated-SOC violation={v}")
                continue
            if isinstance(constraint.set, PositiveSemidefiniteCone):
                M = arr.reshape(constraint.set.dimension, constraint.set.dimension)
                asym = float(np.max(np.abs(M-M.T))) if M.size else 0.0
                eigmin = float(np.linalg.eigvalsh(0.5*(M+M.T))[0]) if M.size else 0.0
                v = max(asym, max(0.0, -eigmin))
                max_v = max(max_v, v)
                if v > atol: violations.append(f"constraint {constraint.entity_id.value} PSD violation={v}")
                continue
            vals = arr.reshape(-1) if arr.shape else np.asarray([float(arr)])
            for k, value in enumerate(vals):
                check_set(float(value), constraint.set, f"constraint {constraint.entity_id.value}[{k}]")
        elif isinstance(constraint, IndicatorConstraint):
            start, end = offsets[constraint.indicator.id.value]
            if end - start != 1:
                violations.append(f"indicator {constraint.entity_id.value} is not scalar")
                max_v = float("inf")
                continue
            z = int(round(float(xx[start])))
            if z == constraint.active_value:
                value = float(np.asarray(_eval_expr(model, constraint.function._node, xx)))
                check_set(value, constraint.set, f"indicator {constraint.entity_id.value}")
    return SemanticPrimalValidation(not violations, max_v, tuple(violations))
