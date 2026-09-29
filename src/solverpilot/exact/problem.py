"""Rational snapshot of a LinearProblem, preserving its represented binary64 data."""
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, floor, isfinite

from solverpilot.problem import LinearProblem


def rational(value):
    return Fraction(float(value))


@dataclass(frozen=True)
class ExactModel:
    c: tuple
    rows: tuple
    lower: tuple
    upper: tuple
    row_lower: tuple
    row_upper: tuple
    integers: frozenset
    sign: int
    offset: Fraction
    data_hash: str

    @classmethod
    def from_problem(cls, problem):
        if not isinstance(problem, LinearProblem):
            raise TypeError("exact solving accepts LinearProblem (LP/MILP) only")
        if not 0 < problem.n_variables <= 10_000 or problem.nnz > 100_000:
            raise ValueError("exact v1 supports 1..10000 variables and at most 100000 nonzeros")
        if problem.n_constraints > 10_000:
            raise ValueError("exact v1 supports at most 10000 rows")
        # Reconstruct through the public validator, including a fresh immutable CSR copy.
        p = LinearProblem(problem.A, problem.c, problem.variable_lower, problem.variable_upper,
                          problem.constraint_lower, problem.constraint_upper, problem.domains,
                          problem.objective_sense, problem.objective_offset)
        rows = tuple(tuple((int(p.A.indices[k]), rational(p.A.data[k]))
                           for k in range(p.A.indptr[i], p.A.indptr[i + 1]))
                     for i in range(p.n_constraints))
        bounds = lambda values: tuple(rational(x) if isfinite(x) else None for x in values)
        return cls(tuple(map(rational, p.c)), rows, bounds(p.variable_lower),
                   bounds(p.variable_upper), bounds(p.constraint_lower), bounds(p.constraint_upper),
                   frozenset(i for i, d in enumerate(p.domains) if d != "continuous"),
                   1 if p.objective_sense.value == "minimize" else -1,
                   rational(p.objective_offset), p.data_hash)

    def feasible(self, x):
        if len(x) != len(self.c):
            return False
        for j, (v, lo, hi) in enumerate(zip(x, self.lower, self.upper)):
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                return False
            if j in self.integers and v.denominator != 1:
                return False
        for row, lo, hi in zip(self.rows, self.row_lower, self.row_upper):
            v = sum((a * x[j] for j, a in row), Fraction())
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                return False
        return True

    def objective(self, x):
        return sum((a * v for a, v in zip(self.c, x)), self.offset)

    def assumptions(self):
        """Original rows/bounds and valid integer-bound rounding, without presolve trust."""
        result = set()
        for j, (lo, hi) in enumerate(zip(self.lower, self.upper)):
            for bound, sense in ((lo, "G"), (hi, "L")):
                if bound is not None:
                    result.add(constraint_key(((j, Fraction(1)),), sense, bound))
                    if j in self.integers:
                        rounded = Fraction(ceil(bound) if sense == "G" else floor(bound))
                        result.add(constraint_key(((j, Fraction(1)),), sense, rounded))
        for row, lo, hi in zip(self.rows, self.row_lower, self.row_upper):
            if lo is not None:
                result.add(constraint_key(row, "G", lo))
            if hi is not None:
                result.add(constraint_key(row, "L", hi))
            if lo is not None and lo == hi:
                result.add(constraint_key(row, "E", lo))
        return result

    def lp_text(self):
        # Always minimize, with no constant. Restore sign and offset after verification.
        expr = lambda row: " ".join(f"{'+' if a >= 0 else '-'} {abs(a)} x{j}"
                                    for j, a in row if a) or "+ 0 x0"
        lines = ["Minimize", " obj: " + expr(enumerate(a * self.sign for a in self.c)),
                 "Subject To"]
        for i, (row, lo, hi) in enumerate(zip(self.rows, self.row_lower, self.row_upper)):
            if lo is not None and lo == hi:
                lines.append(f" r{i}e: {expr(row)} = {lo}")
            else:
                if lo is not None:
                    lines.append(f" r{i}l: {expr(row)} >= {lo}")
                if hi is not None:
                    lines.append(f" r{i}u: {expr(row)} <= {hi}")
        lines.append("Bounds")
        for j, (lo, hi) in enumerate(zip(self.lower, self.upper)):
            lines.append(f" x{j} free")
            if lo is not None:
                lines.append(f" x{j} >= {lo}")
            if hi is not None:
                lines.append(f" x{j} <= {hi}")
        if self.integers:
            lines.append("Generals")
            lines.extend(f" x{j}" for j in sorted(self.integers))
        lines.append("End")
        return "\n".join(lines) + "\n"


def constraint_key(row, sense, rhs):
    row = tuple(sorted((j, a) for j, a in row if a))
    factor = Fraction(-1 if sense == "G" else 1)
    if sense == "G":
        sense = "L"
    if row:
        factor /= abs(row[0][1])
        if sense == "E" and row[0][1] < 0:
            factor = -factor
    return sense, tuple((j, a * factor) for j, a in row), rhs * factor
