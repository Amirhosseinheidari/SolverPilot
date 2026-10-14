"""Explicit bounded PWL epigraph/graph formulations without slope rounding."""
from dataclasses import dataclass
from fractions import Fraction
from bisect import bisect_right

import numpy as np

from solverpilot._immutability import readonly_array
from .expression import Expression
from .model import Model


def _display(value):
    try:
        return float(value)
    except OverflowError:
        return np.inf if value > 0 else -np.inf


@dataclass(frozen=True, slots=True)
class PiecewiseValidation:
    valid: bool
    input_value: float | None
    output_value: float | None
    expected_value: float | None
    domain_violation: float
    relation_violation: float
    reason: str


@dataclass(frozen=True, slots=True)
class PiecewiseLinear:
    model: Model
    input: Expression
    output: Expression
    relation: str
    breakpoints: np.ndarray
    values: np.ndarray
    auxiliary_variables: tuple
    constraints: tuple

    def __post_init__(self):
        object.__setattr__(self, "breakpoints", readonly_array(self.breakpoints, dtype=float))
        object.__setattr__(self, "values", readonly_array(self.values, dtype=float))

    def evaluate(self, value):
        """Original adjacent-knot interpolation; no extrapolation or clipping."""
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
            raise ValueError("PWL input value must be a finite real scalar")
        value = float(value)
        if not np.isfinite(value) or not self.breakpoints[0] <= value <= self.breakpoints[-1]:
            raise ValueError("PWL input is outside its bounded knot domain")
        return float(self._evaluate_exact(Fraction.from_float(value)))

    def _evaluate_exact(self, value):
        points = [Fraction.from_float(float(x)) for x in self.breakpoints]
        if not points[0] <= value <= points[-1]:
            raise ValueError("PWL input is outside its bounded knot domain")
        index = min(bisect_right(points, value) - 1, len(points) - 2)
        ya, yb = [Fraction.from_float(float(x)) for x in self.values[index:index + 2]]
        ratio = (value - points[index]) / (points[index + 1] - points[index])
        return ya + ratio * (yb - ya)

    def validate_original(self, compiled, candidate, *, atol=1e-8):
        """Check the frozen PWL source relation in addition to lifted feasibility.

        This is a primal relation check, not an optimality certificate. A stale
        compiled/model snapshot raises, consistently with CompiledModel.
        """
        if isinstance(atol, (bool, np.bool_)) or not np.isfinite(atol) or atol < 0:
            raise ValueError("atol must be finite and nonnegative")
        if compiled.semantic_hash != self.model.semantic_hash or compiled.data_hash != self.model.data_hash:
            raise ValueError("model differs from the compiled snapshot; refresh before validation")
        try:
            raw = np.asarray(candidate, dtype=float).reshape(-1)
        except (TypeError, ValueError, OverflowError):
            raw = np.array([])
        expected = compiled.execution_ir.n_variables
        if raw.size != expected or not np.isfinite(raw).all():
            return PiecewiseValidation(False, None, None, None, np.inf, np.inf, "invalid_candidate")
        primal = compiled.reconstruct_primal(raw)
        from .qcqp import _Evaluator, _Limits, _Unsupported
        offsets = {key: tuple(value['execution_slice']) for key, value in compiled.source_map['variables'].items()}
        try:
            poly = _Evaluator(self.model, offsets, _Limits(), False).evaluate(self.input._node).item()
        except _Unsupported:
            return PiecewiseValidation(False, None, None, None, np.inf, np.inf, "source_check_limit")
        exact_x = poly.coefficients.get((), Fraction(0))
        for key, coefficient in poly.coefficients.items():
            if key:
                exact_x += coefficient * Fraction.from_float(float(primal[key[0]]))
        y_index = offsets[self.output.id.value][0]
        exact_y = Fraction.from_float(float(primal[y_index]))
        x, y = _display(exact_x), float(exact_y)
        lower, upper = [Fraction.from_float(float(v)) for v in self.breakpoints[[0, -1]]]
        domain = max(lower - exact_x, exact_x - upper, Fraction(0))
        tolerance = Fraction.from_float(float(atol))
        if domain > tolerance:
            return PiecewiseValidation(False, x, y, None, _display(domain), np.inf, "outside_domain")
        # Standard feasibility tolerances permit a small endpoint violation; it
        # remains recorded above rather than changing the source function API.
        source = self._evaluate_exact(min(max(exact_x, lower), upper))
        violation = abs(exact_y - source) if self.relation == "graph" else max(source - exact_y, Fraction(0))
        lifted = compiled.validate_original(self.model, raw, atol=atol)
        valid = bool(lifted.valid and violation <= tolerance)
        return PiecewiseValidation(valid, x, y, float(source), _display(domain), _display(violation),
                                   "validated" if valid else "relation_or_lifted_violation")


def _validate(model, x, breakpoints, values, name):
    if not isinstance(model, Model) or not isinstance(x, Expression) or x._model is not model:
        raise ValueError("PWL input must belong to the supplied Model")
    if x.shape or x.polynomial_degree not in (0, 1):
        raise ValueError("PWL input must be scalar and affine")
    if not isinstance(name, str) or not name:
        raise ValueError("PWL name must be a nonempty string")
    arrays = []
    for value in [breakpoints, values]:
        raw = np.asarray(value)
        if raw.ndim != 1 or raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
            raise ValueError("PWL knots and values must be finite numeric constant vectors")
        converted = np.array(raw, dtype=float, copy=True)
        if raw.dtype.kind in "iu" and any(Fraction(int(a)) != Fraction.from_float(float(b)) for a, b in zip(raw, converted)):
            raise ValueError("PWL integer knots/values must be losslessly representable as binary64")
        if raw.dtype.kind == "f" and not np.array_equal(raw, converted):
            raise ValueError("PWL knots/values must be losslessly representable as binary64")
        arrays.append(converted)
    points, ordinates = arrays
    if not 2 <= len(points) <= 256 or len(ordinates) != len(points):
        raise ValueError("PWL needs matching vectors with 2 to 256 knots")
    if not np.isfinite(points).all() or not np.isfinite(ordinates).all() or np.any(points[1:] <= points[:-1]):
        raise ValueError("PWL knots must be finite and strictly increasing")
    return points, ordinates


def piecewise_linear_epigraph(model, x, breakpoints, values, *, name="pwl"):
    """Add y >= f(x) for convex PWL data on its closed knot interval.

    Convexity uses exact rational slopes, but the formulation uses barycentric
    weights directly. No slopes/intercepts are rounded into model coefficients.
    The caller explicitly chooses how output enters its objective/constraints.
    """
    points, ordinates = _validate(model, x, breakpoints, values, name)
    px = [Fraction.from_float(float(v)) for v in points]
    py = [Fraction.from_float(float(v)) for v in ordinates]
    slopes = [(py[i + 1] - py[i]) / (px[i + 1] - px[i]) for i in range(len(px) - 1)]
    if any(right < left for left, right in zip(slopes, slopes[1:])):
        raise ValueError("PWL epigraph requires convex knot data")
    output = model.variable(lower=float(ordinates.min()), name=f"{name}.output")
    weights = model.variable(len(points), lower=0., upper=1., name=f"{name}.weights")
    constraints = (
        model.add(x >= float(points[0]), name=f"{name}.domain.lower"),
        model.add(x <= float(points[-1]), name=f"{name}.domain.upper"),
        model.add(weights.sum() == 1., name=f"{name}.weights.sum"),
        model.add(x == model.constant(points) @ weights, name=f"{name}.input"),
        model.add(output >= model.constant(ordinates) @ weights, name=f"{name}.epigraph"),
    )
    return PiecewiseLinear(model, x, output, "epigraph", points, ordinates, (weights,), constraints)


def piecewise_linear_graph(model, x, breakpoints, values, *, name="pwl"):
    """Add y = f(x), including nonconvex PWL, using bounded segment binaries.

    Each selected segment has two endpoint weights summing to its binary.
    Original knot coefficients are used directly; arbitrary big-M, native SOS2
    transport and rounded breakpoint differences are not required.
    """
    points, ordinates = _validate(model, x, breakpoints, values, name)
    count = len(points) - 1
    output = model.variable(lower=float(ordinates.min()), upper=float(ordinates.max()), name=f"{name}.output")
    if count == 1:
        weights = model.variable(2, lower=0., upper=1., name=f"{name}.weights")
        constraints = (
            model.add(x >= float(points[0]), name=f"{name}.domain.lower"),
            model.add(x <= float(points[-1]), name=f"{name}.domain.upper"),
            model.add(weights.sum() == 1., name=f"{name}.weights.sum"),
            model.add(x == model.constant(points) @ weights, name=f"{name}.input"),
            model.add(output == model.constant(ordinates) @ weights, name=f"{name}.graph"),
        )
        return PiecewiseLinear(model, x, output, "graph", points, ordinates, (weights,), constraints)
    segments = model.binary(count, name=f"{name}.segments")
    weights = model.variable((count, 2), lower=0., upper=1., name=f"{name}.weights")
    left, right = weights[:, 0], weights[:, 1]
    constraints = (
        model.add(x >= float(points[0]), name=f"{name}.domain.lower"),
        model.add(x <= float(points[-1]), name=f"{name}.domain.upper"),
        model.add(segments.sum() == 1., name=f"{name}.segments.sum"),
        model.add(weights.sum(axis=1) == segments, name=f"{name}.weights.sum"),
        model.add(x == model.constant(points[:-1]) @ left + model.constant(points[1:]) @ right, name=f"{name}.input"),
        model.add(output == model.constant(ordinates[:-1]) @ left + model.constant(ordinates[1:]) @ right, name=f"{name}.graph"),
    )
    return PiecewiseLinear(model, x, output, "graph", points, ordinates, (segments, weights), constraints)
