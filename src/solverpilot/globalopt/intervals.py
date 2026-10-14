"""Exact rational enclosures for bounded polynomial indicator reformulations."""

import builtins
from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
from math import isqrt
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


_PRECISION = 192
_EPSILON = F(1, 1 << _PRECISION)


def _dyadic_enclose(lo, hi):
    scale = 1 << _PRECISION
    lower = (lo.numerator * scale) // lo.denominator
    upper = -((-hi.numerator * scale) // hi.denominator)
    return Range(F(lower, scale), F(upper, scale))


def _exp_point(x):
    if abs(x) > 256: raise DomainError('exp interval exceeds bounded arithmetic policy')
    if x < 0:
        value = _exp_point(-x)
        return _dyadic_enclose(1/value.hi, 1/value.lo)
    reduced=x; squares=0
    while reduced > 1: reduced/=2; squares+=1
    term=F(1); total=F(1)
    for n in range(1, 512):
        term=term*reduced/n; total+=term
        following=term*reduced/(n+1)
        tail=following/(1-reduced/(n+2))
        if tail <= _EPSILON:
            value=_dyadic_enclose(total,total+tail)
            for _ in range(squares): value=_dyadic_enclose(value.lo**2,value.hi**2)
            return value
    raise DomainError('exp interval work budget exceeded')


def _log_unit(x):
    z=(x-1)/(x+1); term=z; total=F(0)
    for n in range(512):
        total+=term/(2*n+1); term*=z*z
        tail=2*term/((2*n+3)*(1-z*z))
        if tail <= _EPSILON: return _dyadic_enclose(2*total,2*total+tail)
    raise DomainError('log interval work budget exceeded')


@lru_cache(maxsize=1)
def _log_two():
    return _log_unit(F(2))


def _log_point(x):
    if x <= 0: raise DomainError('log interval requires positive input')
    if max(x.numerator.bit_length(), x.denominator.bit_length()) > 4096:
        raise DomainError('log interval exceeds bounded arithmetic policy')
    reduced=x; shift=0
    while reduced >= 2: reduced/=2; shift+=1
    while reduced < 1: reduced*=2; shift-=1
    base=_log_unit(reduced); two=_log_two()
    return _dyadic_enclose(base.lo+shift*(two.lo if shift>=0 else two.hi),
                          base.hi+shift*(two.hi if shift>=0 else two.lo))


def _sqrt_point(x):
    if x < 0: raise DomainError('sqrt interval requires nonnegative input')
    scale=1 << _PRECISION; square_scale=scale*scale
    m=isqrt((x.numerator*square_scale)//x.denominator)
    lo=F(m,scale)
    return Range(lo,lo if lo*lo==x else F(m+1,scale))


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
    if kind in ('exp','log','sqrt'):
        function={'exp':_exp_point,'log':_log_point,'sqrt':_sqrt_point}[kind]
        try:
            return np.vectorize(lambda r: Range(function(r.lo).lo,function(r.hi).hi),otypes=[object])(args[0])
        except DomainError:
            return None
    if kind == "index":
        return np.asarray(args[0][node.payload], dtype=object)
    if kind == "transpose":
        return args[0].T
    if kind == "sum":
        return np.asarray(np.sum(args[0], axis=node.payload), dtype=object).reshape(node.shape)
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
        if child.kind == 'exp':
            # Positive for every finite real argument, even below the absolute
            # dyadic enclosure grid. Child domains were checked recursively.
            return
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
