from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from solverpilot.problem import VariableDomain

from .types import (
    BridgePolicy,
    ExactnessClass,
    MappingAvailability,
    PreconditionCertificate,
    TransformationStep,
)


class BridgeUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AffineScalar:
    constant: float
    linear: Mapping[int, float]


@dataclass(frozen=True, slots=True)
class IndicatorLowering:
    rows: tuple[dict[int, float], ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    step: TransformationStep


def _affine_range(form: AffineScalar, lower, upper) -> tuple[float, float, list[dict[str, float | int]]]:
    lo = float(form.constant)
    hi = float(form.constant)
    terms: list[dict[str, float | int]] = []
    for index, coefficient in sorted(form.linear.items()):
        a = float(coefficient)
        if a == 0.0:
            continue
        v_lo = float(lower[index])
        v_hi = float(upper[index])
        need_lo = v_lo if a >= 0 else v_hi
        need_hi = v_hi if a >= 0 else v_lo
        if not math.isfinite(need_lo) or not math.isfinite(need_hi):
            raise BridgeUnavailable(
                f"cannot certify finite Big-M: variable index {index} has an infinite bound needed by affine range"
            )
        lo += a * need_lo
        hi += a * need_hi
        terms.append({"index": int(index), "coefficient": a, "lower": v_lo, "upper": v_hi})
    return lo, hi, terms


def _upper_row(form: AffineScalar, *, rhs: float, z_index: int, active_value: int, M: float):
    coeffs = dict(form.linear)
    if active_value == 1:
        coeffs[z_index] = coeffs.get(z_index, 0.0) + M
        upper = rhs + M - form.constant
    else:
        coeffs[z_index] = coeffs.get(z_index, 0.0) - M
        upper = rhs - form.constant
    return coeffs, -math.inf, float(upper)


def _lower_row(form: AffineScalar, *, rhs: float, z_index: int, active_value: int, M: float):
    coeffs = dict(form.linear)
    if active_value == 1:
        coeffs[z_index] = coeffs.get(z_index, 0.0) - M
        lower = rhs - M - form.constant
    else:
        coeffs[z_index] = coeffs.get(z_index, 0.0) + M
        lower = rhs - form.constant
    return coeffs, float(lower), math.inf


def lower_indicator_big_m(
    *,
    source_id: str,
    indicator_index: int,
    active_value: int,
    indicator_domain: VariableDomain,
    indicator_lower: float,
    indicator_upper: float,
    body: AffineScalar,
    body_set,
    variable_lower,
    variable_upper,
    parameter_dependencies: tuple[str, ...],
    policy: BridgePolicy,
) -> IndicatorLowering:
    if active_value not in {0, 1}:
        raise BridgeUnavailable("indicator active value must be 0 or 1")
    if indicator_domain is not VariableDomain.BINARY:
        raise BridgeUnavailable("indicator variable must be binary")
    if indicator_lower < 0.0 or indicator_upper > 1.0:
        raise BridgeUnavailable("indicator variable bounds must lie within [0, 1]")

    # Exact simplification when the premise is fixed by variable bounds.
    fixed = indicator_lower == indicator_upper and indicator_lower in {0.0, 1.0}
    if fixed:
        from solverpilot.model.sets import EqualTo, GreaterThan, Interval, LessThan
        active = int(indicator_lower) == active_value
        if not active:
            cert = PreconditionCertificate(
                "fixed-indicator-inactive",
                {"indicator_index": indicator_index, "fixed_value": int(indicator_lower), "active_value": active_value},
            )
            step = TransformationStep(
                step_id=f"{source_id}:indicator-fixed-inactive",
                bridge_id="solverpilot.bridge.indicator.fixed",
                bridge_version="1.0",
                source_entity_ids=(source_id,),
                generated_target_ids=(),
                exactness=ExactnessClass.EXACT_EQUIVALENT,
                preconditions=(cert,),
                primal_mapping=MappingAvailability.IDENTITY,
                dual_mapping=MappingAvailability.UNAVAILABLE,
                certificate_mapping=MappingAvailability.UNAVAILABLE,
                mutation_compatibility=("parameter_values",),
                size_delta={"variables": 0, "constraints": -1},
                numerical_risk={"class": "none", "reason": "no Big-M introduced"},
                notes=("premise fixed false; semantic indicator is redundant",),
            )
            return IndicatorLowering((), (), (), step)

        rows: list[dict[int, float]] = []
        lo: list[float] = []
        hi: list[float] = []
        if isinstance(body_set, LessThan):
            rows.append(dict(body.linear)); lo.append(-math.inf); hi.append(body_set.upper - body.constant)
        elif isinstance(body_set, GreaterThan):
            rows.append(dict(body.linear)); lo.append(body_set.lower - body.constant); hi.append(math.inf)
        elif isinstance(body_set, EqualTo):
            rows.append(dict(body.linear)); rhs = body_set.value - body.constant; lo.append(rhs); hi.append(rhs)
        elif isinstance(body_set, Interval):
            rows.append(dict(body.linear)); lo.append(body_set.lower - body.constant); hi.append(body_set.upper - body.constant)
        else:
            raise BridgeUnavailable(f"unsupported indicator body set {type(body_set).__name__}")
        ids = tuple(f"{source_id}:row:{i}" for i in range(len(rows)))
        cert = PreconditionCertificate(
            "fixed-indicator-active",
            {"indicator_index": indicator_index, "fixed_value": int(indicator_lower), "active_value": active_value},
        )
        step = TransformationStep(
            step_id=f"{source_id}:indicator-fixed-active",
            bridge_id="solverpilot.bridge.indicator.fixed",
            bridge_version="1.0",
            source_entity_ids=(source_id,),
            generated_target_ids=ids,
            exactness=ExactnessClass.EXACT_EQUIVALENT,
            preconditions=(cert,),
            primal_mapping=MappingAvailability.IDENTITY,
            dual_mapping=MappingAvailability.UNAVAILABLE,
            certificate_mapping=MappingAvailability.UNAVAILABLE,
            mutation_compatibility=("parameter_values",),
            size_delta={"variables": 0, "constraints": len(rows) - 1},
            numerical_risk={"class": "none", "reason": "fixed premise materializes original affine body"},
        )
        return IndicatorLowering(tuple(rows), tuple(lo), tuple(hi), step)

    if not policy.permits(ExactnessClass.EXACT_EQUIVALENT):
        raise BridgeUnavailable("bridge policy rejects exact indicator lowering")
    if policy.require_finite_certified_big_m is not True:
        raise BridgeUnavailable("P4 safe mode requires finite certified Big-M")

    from solverpilot.model.sets import EqualTo, GreaterThan, Interval, LessThan
    body_min, body_max, bound_terms = _affine_range(body, variable_lower, variable_upper)
    rows: list[dict[int, float]] = []
    lo: list[float] = []
    hi: list[float] = []
    m_values: dict[str, float] = {}

    def add_upper(rhs: float, label: str) -> None:
        M = max(0.0, body_max - float(rhs))
        row, l, u = _upper_row(body, rhs=float(rhs), z_index=indicator_index, active_value=active_value, M=M)
        rows.append(row); lo.append(l); hi.append(u); m_values[label] = M

    def add_lower(rhs: float, label: str) -> None:
        M = max(0.0, float(rhs) - body_min)
        row, l, u = _lower_row(body, rhs=float(rhs), z_index=indicator_index, active_value=active_value, M=M)
        rows.append(row); lo.append(l); hi.append(u); m_values[label] = M

    if isinstance(body_set, LessThan):
        add_upper(body_set.upper, "upper")
    elif isinstance(body_set, GreaterThan):
        add_lower(body_set.lower, "lower")
    elif isinstance(body_set, EqualTo):
        add_lower(body_set.value, "lower"); add_upper(body_set.value, "upper")
    elif isinstance(body_set, Interval):
        add_lower(body_set.lower, "lower"); add_upper(body_set.upper, "upper")
    else:
        raise BridgeUnavailable(f"unsupported indicator body set {type(body_set).__name__}")

    cert = PreconditionCertificate(
        "certified-affine-big-m",
        {
            "source_id": source_id,
            "indicator_index": indicator_index,
            "active_value": active_value,
            "body_min": body_min,
            "body_max": body_max,
            "m_values": m_values,
            "bound_terms": bound_terms,
            "parameter_dependencies": list(parameter_dependencies),
        },
    )
    generated = tuple(f"{source_id}:bigm:{i}" for i in range(len(rows)))
    step = TransformationStep(
        step_id=f"{source_id}:indicator-bigm",
        bridge_id="solverpilot.bridge.indicator.certified-big-m",
        bridge_version="1.0",
        source_entity_ids=(source_id,),
        generated_target_ids=generated,
        exactness=ExactnessClass.EXACT_EQUIVALENT,
        preconditions=(cert,),
        primal_mapping=MappingAvailability.IDENTITY,
        dual_mapping=MappingAvailability.UNAVAILABLE,
        certificate_mapping=MappingAvailability.UNAVAILABLE,
        mutation_compatibility=("parameter_values_revalidate_certificate",),
        size_delta={"variables": 0, "constraints": len(rows) - 1},
        numerical_risk={
            "class": "bound_derived_big_m",
            "max_abs_M": max((abs(v) for v in m_values.values()), default=0.0),
            "all_M_finite": all(math.isfinite(v) for v in m_values.values()),
        },
        notes=("Big-M values are derived from current finite variable bounds; no guessed constant is allowed",),
    )
    return IndicatorLowering(tuple(rows), tuple(lo), tuple(hi), step)
