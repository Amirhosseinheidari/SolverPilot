"""Strict VIPR 1.0 envelope parsing and binding to original model assumptions.

This is not a replacement for VIPR's derivation checker. Both must pass.
"""
from dataclasses import dataclass
from fractions import Fraction
import re

from .problem import constraint_key

MAX_CERTIFICATE_BYTES = 32 * 1024 * 1024
MAX_COUNT = 200_000


@dataclass(frozen=True)
class BoundCertificate:
    relation: str
    lower: Fraction | None
    upper: Fraction | None
    solutions: tuple


class Tokens:
    def __init__(self, payload):
        if len(payload) > MAX_CERTIFICATE_BYTES:
            raise ValueError("certificate exceeds size limit")
        content = payload.decode("ascii")
        self.tokens = iter("\n".join(line for line in content.splitlines()
                                     if not line.lstrip().startswith("%")).split())

    def get(self):
        try:
            value = next(self.tokens)
        except StopIteration:
            raise ValueError("truncated certificate") from None
        if len(value) > 4096:
            raise ValueError("certificate token too long")
        return value

    def expect(self, value):
        if self.get() != value:
            raise ValueError(f"expected {value}")

    def integer(self, maximum=MAX_COUNT, minimum=0):
        token = self.get()
        if not re.fullmatch(r"-?[0-9]{1,9}", token):
            raise ValueError("invalid certificate index/count")
        n = int(token)
        if not minimum <= n <= maximum:
            raise ValueError("certificate index/count out of bounds")
        return n

    def number(self, token=None):
        value = self.get() if token is None else token
        if not re.fullmatch(r"-?[0-9]+(?:/[1-9][0-9]*|\.[0-9]+)?", value):
            raise ValueError("invalid rational literal")
        return Fraction(value)

    def vector(self, n, first=None):
        count = self.integer(n) if first is None else first
        values = {}
        for _ in range(count):
            j = self.integer(n - 1)
            if j in values:
                raise ValueError("duplicate sparse index")
            values[j] = self.number()
        return tuple(sorted((j, a) for j, a in values.items() if a))

    def constraint(self, n, objective):
        self.get()  # label is not interpreted
        sense = self.get()
        if sense not in ("G", "E", "L"):
            raise ValueError("invalid constraint sense")
        rhs = self.number()
        count = self.get()
        if count == "OBJ":
            row = objective
        else:
            if not re.fullmatch(r"[0-9]{1,9}", count) or int(count) > n:
                raise ValueError("invalid sparse length")
            row = self.vector(n, int(count))
        return row, sense, rhs


def bind_certificate(model, payload):
    t = Tokens(payload)
    t.expect("VER")
    t.expect("1.0")
    t.expect("VAR")
    n = t.integer(len(model.c))
    mapping = []
    for _ in range(n):
        name = t.get()
        match = re.fullmatch(r"(?:t_)?x([0-9]+)", name)
        if not match or int(match[1]) >= len(model.c):
            raise ValueError("certificate has an unknown variable")
        mapping.append(int(match[1]))
    if len(set(mapping)) != n:
        raise ValueError("duplicate certificate variable")
    t.expect("INT")
    integer_ids = [t.integer(n - 1) for _ in range(t.integer(n))]
    if len(set(integer_ids)) != len(integer_ids):
        raise ValueError("duplicate integer index")
    if any(mapping[j] not in model.integers for j in integer_ids):
        raise ValueError("certificate assumes integrality absent from original model")
    t.expect("OBJ")
    t.expect("min")
    objective = t.vector(n)
    original_objective = tuple((j, c * model.sign) for j, c in enumerate(model.c) if c)
    if tuple(sorted((mapping[j], a) for j, a in objective)) != original_objective:
        raise ValueError("certificate objective differs from original model")
    t.expect("CON")
    m = t.integer()
    t.integer(m)  # number of bound constraints
    assumptions = model.assumptions()
    for _ in range(m):
        row, sense, rhs = t.constraint(n, objective)
        if constraint_key(((mapping[j], a) for j, a in row), sense, rhs) not in assumptions:
            raise ValueError("certificate contains an unproved model transformation")
    t.expect("RTP")
    relation = t.get()
    lower = upper = None
    if relation == "range":
        lo, hi = t.get(), t.get()
        lower = None if lo == "-inf" else t.number(lo)
        upper = None if hi == "inf" else t.number(hi)
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("reversed certificate range")
    elif relation != "infeas":
        raise ValueError("unsupported certificate relation")
    t.expect("SOL")
    solutions = []
    for _ in range(t.integer(1000)):
        t.get()
        x = [Fraction()] * len(model.c)
        for j, v in t.vector(n):
            x[mapping[j]] = v
        if not model.feasible(x):
            raise ValueError("certificate solution violates the original model")
        solutions.append(tuple(x))
    if relation == "infeas" and solutions:
        raise ValueError("infeasibility certificate contains feasible solutions")
    if upper is not None:
        if not solutions or min(model.sign * (model.objective(x) - model.offset)
                                for x in solutions) > upper:
            raise ValueError("upper bound lacks an original-model feasible solution")
    t.expect("DER")
    d = t.integer()
    # Validate the entire grammar before passing untrusted bytes to the native checker.
    for idx in range(m, m + d):
        t.constraint(n, objective)
        t.expect("{")
        reason = t.get()
        if reason in ("lin", "rnd"):
            seen = set()
            for _ in range(t.integer(idx)):
                ref = t.integer(idx - 1)
                if ref in seen:
                    raise ValueError("duplicate derivation reference")
                seen.add(ref)
                t.number()
        elif reason == "uns":
            for _ in range(4):
                t.integer(idx - 1)
        elif reason not in ("asm", "sol"):
            raise ValueError("incomplete or unsupported derivation")
        t.expect("}")
        t.integer(m + d - 1, -1)
    if next(t.tokens, None) is not None:
        raise ValueError("trailing certificate data")
    return BoundCertificate(relation, lower, upper, tuple(solutions))
