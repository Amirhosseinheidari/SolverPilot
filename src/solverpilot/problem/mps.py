from __future__ import annotations

from dataclasses import dataclass, replace
import gzip
from pathlib import Path
from typing import Iterable, TextIO

import numpy as np
from scipy import sparse

from .enums import ObjectiveSense, VariableDomain
from .linear import LinearProblem


class MPSParseError(ValueError):
    """Raised when an MPS input is malformed or semantically ambiguous."""


class MPSUnsupportedFeatureError(MPSParseError):
    """Raised when an MPS construct cannot be represented by the v0.1 IR."""


_SECTION_NAMES = {
    "NAME",
    "OBJSENSE",
    "OBJNAME",
    "ROWS",
    "COLUMNS",
    "RHS",
    "RANGES",
    "BOUNDS",
    "ENDATA",
}
_UNSUPPORTED_SECTIONS = {
    "QMATRIX",
    "QUADOBJ",
    "QSECTION",
    "DMATRIX",
    "SOS",
    "INDICATORS",
    "GENCONS",
}


def _number(token: str, *, line_no: int) -> float:
    # Fortran-style D exponents are still common in old MPS corpora.
    token = token.replace("D", "E").replace("d", "e")
    try:
        value = float(token)
    except ValueError as exc:
        raise MPSParseError(f"line {line_no}: invalid numeric value {token!r}") from exc
    if not np.isfinite(value):
        raise MPSParseError(f"line {line_no}: non-finite numeric value {token!r}")
    return value


def _clean_marker(token: str) -> str:
    return token.strip().strip("'\"").upper()


@dataclass(slots=True)
class _MPSState:
    name: str | None = None
    objective_sense: ObjectiveSense = ObjectiveSense.MINIMIZE
    objective_row: str | None = None
    explicit_objective_row: str | None = None
    row_types: dict[str, str] = None  # type: ignore[assignment]
    row_order: list[str] = None  # type: ignore[assignment]
    variable_order: list[str] = None  # type: ignore[assignment]
    variable_index: dict[str, int] = None  # type: ignore[assignment]
    entries: dict[tuple[str, str], float] = None  # type: ignore[assignment]
    objective: dict[str, float] = None  # type: ignore[assignment]
    rhs: dict[str, float] = None  # type: ignore[assignment]
    ranges: dict[str, float] = None  # type: ignore[assignment]
    bounds_ops: list[tuple[str, str, float | None]] = None  # type: ignore[assignment]
    integer_marked: set[str] = None  # type: ignore[assignment]
    explicit_binary: set[str] = None  # type: ignore[assignment]
    explicit_integer: set[str] = None  # type: ignore[assignment]
    active_rhs_name: str | None = None
    active_range_name: str | None = None
    active_bound_name: str | None = None
    integer_mode: bool = False

    def __post_init__(self) -> None:
        self.row_types = {}
        self.row_order = []
        self.variable_order = []
        self.variable_index = {}
        self.entries = {}
        self.objective = {}
        self.rhs = {}
        self.ranges = {}
        self.bounds_ops = []
        self.integer_marked = set()
        self.explicit_binary = set()
        self.explicit_integer = set()

    def ensure_variable(self, name: str) -> None:
        if name not in self.variable_index:
            self.variable_index[name] = len(self.variable_order)
            self.variable_order.append(name)


def _iter_lines(source: Iterable[str]):
    for line_no, raw in enumerate(source, start=1):
        # MPS comments are indicated by '*' in column 1. Blank lines are harmless.
        if not raw.strip() or raw.startswith("*"):
            continue
        yield line_no, raw.rstrip("\r\n")


def _parse_stream(source: Iterable[str]) -> LinearProblem:
    state = _MPSState()
    section: str | None = None

    for line_no, raw in _iter_lines(source):
        stripped = raw.strip()
        tokens = stripped.split()
        if not tokens:
            continue

        first_upper = tokens[0].upper()
        # Section cards begin in column 1 in fixed MPS. Accept free-MPS section cards too.
        is_section_card = (not raw[:1].isspace()) and (
            first_upper in _SECTION_NAMES or first_upper in _UNSUPPORTED_SECTIONS
        )
        if is_section_card:
            if first_upper in _UNSUPPORTED_SECTIONS:
                raise MPSUnsupportedFeatureError(
                    f"line {line_no}: section {first_upper} is outside the v0.1 linear/MILP MPS reader"
                )
            section = first_upper
            if section == "NAME":
                if len(tokens) >= 2:
                    state.name = tokens[1]
                continue
            if section == "ENDATA":
                break
            continue

        if section is None:
            raise MPSParseError(f"line {line_no}: data before first MPS section")

        if section == "OBJSENSE":
            sense = first_upper
            if sense in {"MIN", "MINIMIZE", "MINIMUM"}:
                state.objective_sense = ObjectiveSense.MINIMIZE
            elif sense in {"MAX", "MAXIMIZE", "MAXIMUM"}:
                state.objective_sense = ObjectiveSense.MAXIMIZE
            else:
                raise MPSParseError(f"line {line_no}: unknown OBJSENSE value {tokens[0]!r}")
            continue

        if section == "OBJNAME":
            state.explicit_objective_row = tokens[0]
            continue

        if section == "ROWS":
            if len(tokens) < 2:
                raise MPSParseError(f"line {line_no}: ROWS entry requires type and name")
            row_type, row_name = tokens[0].upper(), tokens[1]
            if row_type not in {"N", "E", "L", "G"}:
                raise MPSParseError(f"line {line_no}: unsupported row type {row_type!r}")
            if row_name in state.row_types:
                raise MPSParseError(f"line {line_no}: duplicate row {row_name!r}")
            state.row_types[row_name] = row_type
            state.row_order.append(row_name)
            if row_type == "N" and state.objective_row is None:
                state.objective_row = row_name
            continue

        if section == "COLUMNS":
            # Marker cards tokenize as: MARK0000 'MARKER' 'INTORG' (or INTEND).
            if len(tokens) >= 3 and _clean_marker(tokens[1]) == "MARKER":
                marker = _clean_marker(tokens[-1])
                if marker == "INTORG":
                    state.integer_mode = True
                elif marker == "INTEND":
                    state.integer_mode = False
                else:
                    raise MPSParseError(f"line {line_no}: unknown integer marker {tokens[-1]!r}")
                continue

            if len(tokens) not in {3, 5}:
                raise MPSParseError(
                    f"line {line_no}: COLUMNS entry must contain one or two row/value pairs"
                )
            var = tokens[0]
            state.ensure_variable(var)
            if state.integer_mode:
                state.integer_marked.add(var)
            pairs = [(tokens[1], tokens[2])]
            if len(tokens) == 5:
                pairs.append((tokens[3], tokens[4]))
            objective_row = state.explicit_objective_row or state.objective_row
            for row, raw_value in pairs:
                if row not in state.row_types:
                    raise MPSParseError(f"line {line_no}: unknown row {row!r} in COLUMNS")
                value = _number(raw_value, line_no=line_no)
                if row == objective_row:
                    state.objective[var] = state.objective.get(var, 0.0) + value
                elif state.row_types[row] == "N":
                    # Additional N rows are free rows; they are not constraints in the canonical IR.
                    continue
                else:
                    key = (row, var)
                    state.entries[key] = state.entries.get(key, 0.0) + value
            continue

        if section in {"RHS", "RANGES"}:
            if len(tokens) not in {3, 5}:
                raise MPSParseError(
                    f"line {line_no}: {section} entry must contain one or two row/value pairs"
                )
            vector_name = tokens[0]
            active_attr = "active_rhs_name" if section == "RHS" else "active_range_name"
            active = getattr(state, active_attr)
            if active is None:
                setattr(state, active_attr, vector_name)
                active = vector_name
            if vector_name != active:
                raise MPSUnsupportedFeatureError(
                    f"line {line_no}: multiple {section} vectors are ambiguous in canonical v0.1; "
                    f"active={active!r}, encountered={vector_name!r}"
                )
            pairs = [(tokens[1], tokens[2])]
            if len(tokens) == 5:
                pairs.append((tokens[3], tokens[4]))
            target = state.rhs if section == "RHS" else state.ranges
            for row, raw_value in pairs:
                if row not in state.row_types:
                    raise MPSParseError(f"line {line_no}: unknown row {row!r} in {section}")
                target[row] = _number(raw_value, line_no=line_no)
            continue

        if section == "BOUNDS":
            if len(tokens) < 3:
                raise MPSParseError(f"line {line_no}: malformed BOUNDS entry")
            bound_type = tokens[0].upper()
            vector_name = tokens[1]
            var = tokens[2]
            if state.active_bound_name is None:
                state.active_bound_name = vector_name
            if vector_name != state.active_bound_name:
                raise MPSUnsupportedFeatureError(
                    f"line {line_no}: multiple BOUNDS vectors are ambiguous in canonical v0.1; "
                    f"active={state.active_bound_name!r}, encountered={vector_name!r}"
                )
            state.ensure_variable(var)
            value = _number(tokens[3], line_no=line_no) if len(tokens) >= 4 else None
            if bound_type in {"SC", "SI"}:
                raise MPSUnsupportedFeatureError(
                    f"line {line_no}: semi-continuous/semi-integer bound {bound_type} is unsupported"
                )
            if bound_type not in {"LO", "UP", "FX", "FR", "MI", "PL", "BV", "LI", "UI"}:
                raise MPSUnsupportedFeatureError(
                    f"line {line_no}: unsupported bound type {bound_type!r}"
                )
            if bound_type in {"LO", "UP", "FX", "LI", "UI"} and value is None:
                raise MPSParseError(f"line {line_no}: bound type {bound_type} requires a value")
            if bound_type == "BV":
                state.explicit_binary.add(var)
            if bound_type in {"LI", "UI"}:
                state.explicit_integer.add(var)
            state.bounds_ops.append((bound_type, var, value))
            continue

        # NAME was already handled as a section card; ENDATA exits above.
        if section not in {"NAME"}:
            raise MPSParseError(f"line {line_no}: unexpected data in section {section}")

    if not state.row_types:
        raise MPSParseError("MPS input contains no ROWS entries")
    if not state.variable_order:
        raise MPSParseError("MPS input contains no variables")

    objective_row = state.explicit_objective_row or state.objective_row
    if state.explicit_objective_row is not None:
        if state.explicit_objective_row not in state.row_types:
            raise MPSParseError(f"OBJNAME references unknown row {state.explicit_objective_row!r}")
        if state.row_types[state.explicit_objective_row] != "N":
            raise MPSParseError("OBJNAME must reference an N row")

    constraint_rows = [r for r in state.row_order if state.row_types[r] != "N"]
    row_index = {name: i for i, name in enumerate(constraint_rows)}
    n = len(state.variable_order)
    m = len(constraint_rows)

    rr: list[int] = []
    cc: list[int] = []
    vv: list[float] = []
    for (row, var), value in state.entries.items():
        if value == 0.0:
            continue
        rr.append(row_index[row])
        cc.append(state.variable_index[var])
        vv.append(value)
    A = sparse.coo_matrix((vv, (rr, cc)), shape=(m, n), dtype=np.float64).tocsr()

    c = np.zeros(n, dtype=np.float64)
    for var, value in state.objective.items():
        c[state.variable_index[var]] += value

    rhs = np.zeros(m, dtype=np.float64)
    for row, value in state.rhs.items():
        if row in row_index:
            rhs[row_index[row]] = value

    lower = np.full(m, -np.inf, dtype=np.float64)
    upper = np.full(m, np.inf, dtype=np.float64)
    for row, i in row_index.items():
        b = rhs[i]
        row_type = state.row_types[row]
        if row_type == "E":
            lower[i] = upper[i] = b
        elif row_type == "L":
            upper[i] = b
        elif row_type == "G":
            lower[i] = b

        if row in state.ranges:
            r = state.ranges[row]
            width = abs(r)
            if row_type == "G":
                lower[i], upper[i] = b, b + width
            elif row_type == "L":
                lower[i], upper[i] = b - width, b
            elif row_type == "E":
                if r >= 0:
                    lower[i], upper[i] = b, b + width
                else:
                    lower[i], upper[i] = b - width, b

    # Standard MPS integer markers imply default bounds [0, 1] for marked variables.
    integer_vars = state.integer_marked | state.explicit_integer | state.explicit_binary
    vl = np.zeros(n, dtype=np.float64)
    vu = np.full(n, np.inf, dtype=np.float64)
    for var in state.integer_marked:
        vu[state.variable_index[var]] = 1.0

    explicit_upper = np.zeros(n, dtype=bool)
    for bound_type, var, value in state.bounds_ops:
        j = state.variable_index[var]
        if bound_type == "LO":
            if value is None:
                raise MPSParseError("LO bound requires a numeric value")
            vl[j] = value
        elif bound_type == "UP":
            if value is None:
                raise MPSParseError("UP bound requires a numeric value")
            vu[j] = value
            explicit_upper[j] = True
            # Widely used MPS interpretation: a negative UP releases the default 0 lower bound.
            if value < 0.0 and vl[j] == 0.0:
                vl[j] = -np.inf
        elif bound_type == "FX":
            if value is None:
                raise MPSParseError("FX bound requires a numeric value")
            vl[j] = vu[j] = value
            explicit_upper[j] = True
        elif bound_type == "FR":
            vl[j], vu[j] = -np.inf, np.inf
            explicit_upper[j] = True
        elif bound_type == "MI":
            vl[j] = -np.inf
        elif bound_type == "PL":
            vu[j] = np.inf
            explicit_upper[j] = True
        elif bound_type == "BV":
            vl[j], vu[j] = 0.0, 1.0
            explicit_upper[j] = True
        elif bound_type == "LI":
            if value is None:
                raise MPSParseError("LI bound requires a numeric value")
            vl[j] = value
            # LI explicitly declares a general integer. If no upper bound was already
            # supplied, it overrides the INTORG marker's implicit binary upper bound.
            if not explicit_upper[j]:
                vu[j] = np.inf
        elif bound_type == "UI":
            if value is None:
                raise MPSParseError("UI bound requires a numeric value")
            vu[j] = value
            explicit_upper[j] = True

    domains: list[VariableDomain] = []
    for var in state.variable_order:
        j = state.variable_index[var]
        if var in state.explicit_binary or (
            var in integer_vars and vl[j] == 0.0 and vu[j] == 1.0
        ):
            domains.append(VariableDomain.BINARY)
        elif var in integer_vars:
            domains.append(VariableDomain.INTEGER)
        else:
            domains.append(VariableDomain.CONTINUOUS)

    objective_offset = 0.0
    if objective_row is not None and objective_row in state.rhs:
        # In MPS the RHS value attached to the objective N-row is -c0.
        objective_offset = -float(state.rhs[objective_row])

    return LinearProblem.from_data(
        A=A,
        c=c,
        variable_lower=vl,
        variable_upper=vu,
        constraint_lower=lower,
        constraint_upper=upper,
        domains=domains,
        objective_sense=state.objective_sense,
        objective_offset=objective_offset,
        name=state.name,
        metadata={
            "source_format": "mps",
            "variable_names": tuple(state.variable_order),
            "constraint_names": tuple(constraint_rows),
            "objective_row": objective_row,
            "rhs_name": state.active_rhs_name,
            "ranges_name": state.active_range_name,
            "bounds_name": state.active_bound_name,
        },
    )


def parse_mps(text: str) -> LinearProblem:
    """Parse a linear/MILP MPS document into the canonical sparse IR.

    Supported v0.1 constructs: NAME, OBJSENSE, OBJNAME, ROWS, COLUMNS,
    integer MARKER cards, RHS, RANGES, BOUNDS (LO/UP/FX/FR/MI/PL/BV/LI/UI),
    and ENDATA. Quadratic, SOS, indicator and semi-continuous constructs are rejected
    rather than silently lowered.
    """

    return _parse_stream(text.splitlines())


def read_mps(path: str | Path) -> LinearProblem:
    """Read ``.mps`` or ``.mps.gz`` into the canonical sparse IR."""

    p = Path(path)
    opener = gzip.open if p.suffix.lower() == ".gz" else open
    with opener(p, "rt", encoding="ascii", errors="strict") as f:  # type: ignore[arg-type]
        problem = _parse_stream(f)
    metadata = dict(problem.metadata)
    metadata["source_path"] = str(p)
    return replace(problem, metadata=metadata)
