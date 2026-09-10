from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

from .errors import DomainError, OwnershipError, ShapeError, SymbolicTruthValueError
from .types import Curvature, SignDomain

if TYPE_CHECKING:
    from .model import Model
    from .sets import ScalarSet


def _shape_tuple(shape: int | Iterable[int] | tuple[int, ...] | None) -> tuple[int, ...]:
    if shape is None:
        return ()
    if isinstance(shape, int):
        if shape < 0:
            raise ShapeError("shape dimensions must be nonnegative")
        return (shape,)
    out = tuple(int(x) for x in shape)
    if any(x < 0 for x in out):
        raise ShapeError("shape dimensions must be nonnegative")
    return out


def _sign_of_constant(value: np.ndarray) -> SignDomain:
    if value.size == 0 or np.all(value == 0):
        return SignDomain.ZERO
    if np.all(value >= 0):
        return SignDomain.NONNEGATIVE
    if np.all(value <= 0):
        return SignDomain.NONPOSITIVE
    return SignDomain.UNKNOWN


def _neg_sign(sign: SignDomain) -> SignDomain:
    if sign is SignDomain.NONNEGATIVE:
        return SignDomain.NONPOSITIVE
    if sign is SignDomain.NONPOSITIVE:
        return SignDomain.NONNEGATIVE
    return sign


def _add_sign(a: SignDomain, b: SignDomain) -> SignDomain:
    if a is SignDomain.ZERO:
        return b
    if b is SignDomain.ZERO:
        return a
    if a is b and a in {SignDomain.NONNEGATIVE, SignDomain.NONPOSITIVE}:
        return a
    return SignDomain.UNKNOWN


def _mul_sign(a: SignDomain, b: SignDomain) -> SignDomain:
    if SignDomain.ZERO in {a, b}:
        return SignDomain.ZERO
    if a is SignDomain.UNKNOWN or b is SignDomain.UNKNOWN:
        return SignDomain.UNKNOWN
    if a is b:
        return SignDomain.NONNEGATIVE
    return SignDomain.NONPOSITIVE


def _broadcast_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    try:
        return tuple(np.broadcast_shapes(a, b))
    except ValueError as exc:
        raise ShapeError(f"cannot broadcast shapes {a} and {b}") from exc


def _matmul_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    if len(a) == 0 or len(b) == 0:
        raise ShapeError("matrix multiplication requires non-scalar operands")
    if len(a) > 2 or len(b) > 2:
        raise ShapeError("P1 matrix multiplication supports at most 2-D operands")
    if len(a) == 1 and len(b) == 1:
        if a[0] != b[0]:
            raise ShapeError(f"matmul dimension mismatch: {a} @ {b}")
        return ()
    if len(a) == 2 and len(b) == 1:
        if a[1] != b[0]:
            raise ShapeError(f"matmul dimension mismatch: {a} @ {b}")
        return (a[0],)
    if len(a) == 1 and len(b) == 2:
        if a[0] != b[0]:
            raise ShapeError(f"matmul dimension mismatch: {a} @ {b}")
        return (b[1],)
    if a[1] != b[0]:
        raise ShapeError(f"matmul dimension mismatch: {a} @ {b}")
    return (a[0], b[1])


def _slice_length(sl: slice, n: int) -> int:
    start, stop, step = sl.indices(n)
    return len(range(start, stop, step))


def _index_shape(shape: tuple[int, ...], key: Any) -> tuple[int, ...]:
    if not isinstance(key, tuple):
        key = (key,)
    if len(key) > len(shape):
        raise ShapeError("too many indices for expression")
    out: list[int] = []
    for dim, item in zip(shape, key):
        if isinstance(item, (int, np.integer)):
            idx = int(item)
            if idx < -dim or idx >= dim:
                raise ShapeError("expression index out of range")
        elif isinstance(item, slice):
            out.append(_slice_length(item, dim))
        else:
            raise ShapeError("P1 supports integer and slice indexing only")
    out.extend(shape[len(key):])
    return tuple(out)


@dataclass(frozen=True, slots=True, eq=False)
class ExprNode:
    kind: str
    shape: tuple[int, ...]
    args: tuple["ExprNode", ...] = ()
    payload: Any = None
    variable_dependencies: frozenset[str] = frozenset()
    parameter_dependencies: frozenset[str] = frozenset()
    sign: SignDomain = SignDomain.UNKNOWN
    degree: int | None = None
    curvature: Curvature = Curvature.UNKNOWN


class Expression:
    __slots__ = ("_model", "_node")
    __array_priority__ = 10000

    def __init__(self, model: "Model", node: ExprNode) -> None:
        self._model = model
        self._node = node

    @property
    def shape(self) -> tuple[int, ...]:
        return self._node.shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) if self.shape else 1

    @property
    def variable_dependencies(self) -> frozenset[str]:
        return self._node.variable_dependencies

    @property
    def parameter_dependencies(self) -> frozenset[str]:
        return self._node.parameter_dependencies

    @property
    def sign(self) -> SignDomain:
        return self._node.sign

    @property
    def polynomial_degree(self) -> int | None:
        return self._node.degree

    @property
    def curvature(self) -> Curvature:
        return self._node.curvature

    def _coerce(self, other: Any) -> "Expression":
        if isinstance(other, Expression):
            if other._model is not self._model:
                raise OwnershipError("cross-model expression reference is not allowed")
            return other
        return self._model.constant(other)

    def _binary(self, other: Any, kind: str) -> "Expression":
        rhs = self._coerce(other)
        if kind in {"add", "mul"}:
            shape = _broadcast_shape(self.shape, rhs.shape)
        elif kind == "matmul":
            shape = _matmul_shape(self.shape, rhs.shape)
        else:
            raise AssertionError(kind)
        deps_v = self.variable_dependencies | rhs.variable_dependencies
        deps_p = self.parameter_dependencies | rhs.parameter_dependencies
        if kind == "add":
            degree = None if None in {self.polynomial_degree, rhs.polynomial_degree} else max(self.polynomial_degree, rhs.polynomial_degree)
            sign = _add_sign(self.sign, rhs.sign)
        elif kind == "mul":
            degree = None if None in {self.polynomial_degree, rhs.polynomial_degree} else self.polynomial_degree + rhs.polynomial_degree
            sign = _mul_sign(self.sign, rhs.sign)
        else:
            degree = None if None in {self.polynomial_degree, rhs.polynomial_degree} else self.polynomial_degree + rhs.polynomial_degree
            sign = SignDomain.UNKNOWN
        curvature = Curvature.CONSTANT if degree == 0 else Curvature.AFFINE if degree is not None and degree <= 1 else Curvature.UNKNOWN
        args = (self._node, rhs._node)
        if kind in {"add", "mul"}:
            args = tuple(sorted(args, key=self._model._node_sort_key))
        return Expression(self._model, ExprNode(kind, shape, args=args, variable_dependencies=deps_v, parameter_dependencies=deps_p, sign=sign, degree=degree, curvature=curvature))

    def __add__(self, other: Any) -> "Expression":
        return self._binary(other, "add")

    def __radd__(self, other: Any) -> "Expression":
        return self._coerce(other)._binary(self, "add")

    def __sub__(self, other: Any) -> "Expression":
        return self + (-self._coerce(other))

    def __rsub__(self, other: Any) -> "Expression":
        return self._coerce(other) + (-self)

    def __neg__(self) -> "Expression":
        degree = self.polynomial_degree
        curvature = Curvature.CONSTANT if degree == 0 else Curvature.AFFINE if degree is not None and degree <= 1 else Curvature.UNKNOWN
        return Expression(self._model, ExprNode("neg", self.shape, args=(self._node,), variable_dependencies=self.variable_dependencies, parameter_dependencies=self.parameter_dependencies, sign=_neg_sign(self.sign), degree=degree, curvature=curvature))

    def __mul__(self, other: Any) -> "Expression":
        return self._binary(other, "mul")

    def __rmul__(self, other: Any) -> "Expression":
        return self._coerce(other)._binary(self, "mul")

    def __truediv__(self, other: Any) -> "Expression":
        rhs = self._coerce(other)
        if not rhs.variable_dependencies and not rhs.parameter_dependencies and rhs.polynomial_degree == 0:
            value = self._model._constant_value(rhs._node)
            if np.any(value == 0):
                raise DomainError("division by zero")
            return self * self._model.constant(1.0 / value)
        shape = _broadcast_shape(self.shape, rhs.shape)
        return Expression(self._model, ExprNode("div", shape, args=(self._node, rhs._node), variable_dependencies=self.variable_dependencies | rhs.variable_dependencies, parameter_dependencies=self.parameter_dependencies | rhs.parameter_dependencies, sign=SignDomain.UNKNOWN, degree=None, curvature=Curvature.UNKNOWN))

    def __rtruediv__(self, other: Any) -> "Expression":
        return self._coerce(other).__truediv__(self)

    def __matmul__(self, other: Any) -> "Expression":
        return self._binary(other, "matmul")

    def __rmatmul__(self, other: Any) -> "Expression":
        return self._coerce(other)._binary(self, "matmul")

    def __pow__(self, exponent: int) -> "Expression":
        exponent = int(exponent)
        if exponent < 0:
            raise DomainError("negative symbolic powers are outside P7 smooth-core scope")
        if exponent == 0:
            return self._model.constant(np.ones(self.shape if self.shape else (), dtype=float))
        if exponent == 1:
            return self
        if exponent == 2:
            return self * self
        deps_v = self.variable_dependencies
        deps_p = self.parameter_dependencies
        degree = None if self.polynomial_degree is None else self.polynomial_degree * exponent
        sign = SignDomain.NONNEGATIVE if exponent % 2 == 0 else self.sign
        return Expression(self._model, ExprNode("pow", self.shape, args=(self._node,), payload=exponent, variable_dependencies=deps_v, parameter_dependencies=deps_p, sign=sign, degree=degree, curvature=Curvature.UNKNOWN))

    def _smooth_unary(self, kind: str) -> "Expression":
        sign = SignDomain.UNKNOWN
        if kind in {"exp", "sqrt"}:
            sign = SignDomain.NONNEGATIVE
        return Expression(self._model, ExprNode(kind, self.shape, args=(self._node,), variable_dependencies=self.variable_dependencies, parameter_dependencies=self.parameter_dependencies, sign=sign, degree=None, curvature=Curvature.UNKNOWN))

    def sin(self) -> "Expression": return self._smooth_unary("sin")
    def cos(self) -> "Expression": return self._smooth_unary("cos")
    def exp(self) -> "Expression": return self._smooth_unary("exp")
    def log(self) -> "Expression": return self._smooth_unary("log")
    def sqrt(self) -> "Expression": return self._smooth_unary("sqrt")
    def tanh(self) -> "Expression": return self._smooth_unary("tanh")

    def __getitem__(self, key: Any) -> "Expression":
        shape = _index_shape(self.shape, key)
        sign = self.sign
        if self._node.kind in {"variable", "parameter", "constant"}:
            sign = self._model._indexed_sign(self._node, key)
        return Expression(self._model, ExprNode("index", shape, args=(self._node,), payload=key, variable_dependencies=self.variable_dependencies, parameter_dependencies=self.parameter_dependencies, sign=sign, degree=self.polynomial_degree, curvature=self.curvature))

    @property
    def T(self) -> "Expression":
        if self.ndim < 2:
            return self
        if self.ndim != 2:
            raise ShapeError("P1 transpose supports at most 2-D expressions")
        return Expression(self._model, ExprNode("transpose", (self.shape[1], self.shape[0]), args=(self._node,), variable_dependencies=self.variable_dependencies, parameter_dependencies=self.parameter_dependencies, sign=self.sign, degree=self.polynomial_degree, curvature=self.curvature))

    def sum(self, axis: int | None = None) -> "Expression":
        if axis is not None:
            axis = int(axis)
            if axis < 0:
                axis += self.ndim
            if axis < 0 or axis >= self.ndim:
                raise ShapeError("sum axis out of range")
            shape = self.shape[:axis] + self.shape[axis + 1:]
        else:
            shape = ()
        return Expression(self._model, ExprNode("sum", shape, args=(self._node,), payload=axis, variable_dependencies=self.variable_dependencies, parameter_dependencies=self.parameter_dependencies, sign=self.sign, degree=self.polynomial_degree, curvature=self.curvature))

    def __le__(self, other: Any):
        from .model import PendingConstraint
        rhs = self._coerce(other)
        return PendingConstraint(self - rhs, "le")

    def __ge__(self, other: Any):
        from .model import PendingConstraint
        rhs = self._coerce(other)
        return PendingConstraint(self - rhs, "ge")

    def __eq__(self, other: Any):  # type: ignore[override]
        from .model import PendingConstraint
        rhs = self._coerce(other)
        return PendingConstraint(self - rhs, "eq")

    def __bool__(self) -> bool:
        raise SymbolicTruthValueError("symbolic expressions/relations cannot be converted to bool")

    def __float__(self) -> float:
        raise TypeError("symbolic expressions cannot be converted to float")

    def __array__(self, dtype=None):
        raise TypeError("symbolic expressions cannot be converted to NumPy arrays")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or kwargs.get("out") is not None:
            return NotImplemented
        if ufunc is np.add:
            return _as_expression(inputs[0], self._model) + inputs[1]
        if ufunc is np.subtract:
            return _as_expression(inputs[0], self._model) - inputs[1]
        if ufunc is np.multiply:
            return _as_expression(inputs[0], self._model) * inputs[1]
        if ufunc is np.divide:
            return _as_expression(inputs[0], self._model) / inputs[1]
        if ufunc is np.matmul:
            return _as_expression(inputs[0], self._model) @ inputs[1]
        if ufunc is np.negative:
            return -_as_expression(inputs[0], self._model)
        if ufunc is np.sin:
            return _as_expression(inputs[0], self._model).sin()
        if ufunc is np.cos:
            return _as_expression(inputs[0], self._model).cos()
        if ufunc is np.exp:
            return _as_expression(inputs[0], self._model).exp()
        if ufunc is np.log:
            return _as_expression(inputs[0], self._model).log()
        if ufunc is np.sqrt:
            return _as_expression(inputs[0], self._model).sqrt()
        if ufunc is np.tanh:
            return _as_expression(inputs[0], self._model).tanh()
        return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        if func is np.sum:
            expr = _as_expression(args[0], self._model)
            return expr.sum(axis=kwargs.get("axis"))
        if func is np.dot:
            return _as_expression(args[0], self._model) @ args[1]
        return NotImplemented

    def __repr__(self) -> str:
        return f"Expression(kind={self._node.kind!r}, shape={self.shape}, degree={self.polynomial_degree})"


def _as_expression(value: Any, model: "Model") -> Expression:
    if isinstance(value, Expression):
        if value._model is not model:
            raise OwnershipError("cross-model expression reference is not allowed")
        return value
    return model.constant(value)
