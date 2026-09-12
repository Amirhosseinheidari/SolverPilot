"""Exact rational enclosures for bounded polynomial indicator reformulations."""

import builtins
from dataclasses import dataclass
from fractions import Fraction as F
import numpy as np
from solverpilot.model.errors import DomainError


@dataclass(frozen=True, slots=True)
class Range:
    lo: F
    hi: F

    @staticmethod
    def coerce(value):
        return value if isinstance(value, Range) else Range(F(value), F(value))

    def __add__(self, other):
        other = self.coerce(other)
        return Range(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self):
        return Range(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -self.coerce(other)

    def __rsub__(self, other):
        return self.coerce(other) + -self

    def __mul__(self, other):
        other = self.coerce(other)
        v = [self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi]
        return Range(min(v), max(v))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.coerce(other)
        if other.lo <= 0 <= other.hi:
            raise DomainError("denominator interval contains zero")
        return self * Range(1 / other.hi, 1 / other.lo)

    def __pow__(self, n):
        if type(n) is not int or not 0 <= n <= 64:
            raise DomainError("unsupported interval exponent")
        if n == 0:
            return Range(F(1), F(1))
        if n % 2:
            return Range(self.lo**n, self.hi**n)
        return Range(
            F(0) if self.lo <= 0 <= self.hi else min(self.lo**n, self.hi**n),
            max(self.lo**n, self.hi**n),
        )

    def __abs__(self):
        return Range(
            F(0) if self.lo <= 0 <= self.hi else min(abs(self.lo), abs(self.hi)),
            max(abs(self.lo), abs(self.hi)),
        )


def bound_node(problem, node):
    p = problem.relaxation
    kind = node.kind

    def constants(value):
        a = np.asarray(value, dtype=float)
        return np.array(
            [Range(F(float(v)), F(float(v))) for v in a.reshape(-1)], dtype=object
        ).reshape(a.shape)

    if kind == "constant":
        return constants(node.payload)
    if kind == "parameter":
        return constants(p.parameter_values[node.payload])
    if kind == "variable":
        off, shape = p.variable_layout[node.payload]
        n = int(np.prod(shape)) if shape else 1
        return np.array(
            [
                Range(F(float(lo)), F(float(hi)))
                for lo, hi in zip(p.variable_lower[off : off + n], p.variable_upper[off : off + n])
            ],
            dtype=object,
        ).reshape(shape)
    args = [bound_node(problem, c) for c in node.args]
    if kind in ("sin", "cos", "tanh"):
        return np.full(node.shape, Range(F(-1), F(1)), dtype=object)
    if any(a is None for a in args):
        return None
    if kind == "index":
        return np.asarray(args[0][node.payload], dtype=object)
    if kind == "transpose":
        return args[0].T
    if kind == "sum":
        return np.asarray(np.sum(args[0], axis=node.payload), dtype=object)
    if kind == "neg":
        return -args[0]
    if kind == "add":
        return args[0] + args[1]
    if kind == "mul":
        return args[0] * args[1]
    if kind == "div":
        return args[0] / args[1]
    if kind == "matmul":
        return np.asarray(args[0] @ args[1], dtype=object)
    if kind == "pow":
        return args[0] ** node.payload
    if kind == "concat":
        return np.concatenate([a.reshape(-1) for a in args])
    if kind in ("atom_abs", "global_abs"):
        return np.vectorize(builtins.abs, otypes=[object])(args[0])
    if kind in ("global_max", "global_min"):
        op = max if kind == "global_max" else min
        combine = lambda a, b: Range(op(a.lo, b.lo), op(a.hi, b.hi))
        if len(args) == 2:
            return np.vectorize(combine, otypes=[object])(*args)
        seq = args[0].reshape(-1)
        value = seq[0]
        for term in seq[1:]:
            value = combine(value, term)
        return np.asarray(value, dtype=object)
    # No rounded elementary-function value is advertised as a rigorous bound.
    return None


def validate_domains(problem, node):
    for child in node.args:
        validate_domains(problem, child)
    if node.kind in ("log", "sqrt", "div"):
        child = node.args[1] if node.kind == "div" else node.args[0]
        enclosure = bound_node(problem, child)
        if enclosure is None:
            raise DomainError(f"{node.kind} domain cannot be certified from declared bounds")
        for value in np.asarray(enclosure).reshape(-1):
            safe = (
                value.lo > 0
                if node.kind == "log"
                else value.lo >= 0
                if node.kind == "sqrt"
                else not value.lo <= 0 <= value.hi
            )
            if not safe:
                raise DomainError(f"{node.kind} domain is not safe over declared bounds")


def upper_float(value):
    """Smallest available outward adjustment of a rational upper bound."""
    out = float(value)
    if not np.isfinite(out):
        raise DomainError("indicator bound cannot be represented finitely")
    if F(out) < value:
        out = float(np.nextafter(out, np.inf))
    if not np.isfinite(out):
        raise DomainError("indicator bound exceeds finite binary64")
    return out
