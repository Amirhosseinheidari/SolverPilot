import gzip
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from solverpilot import (
    MPSParseError,
    MPSUnsupportedFeatureError,
    ObjectiveSense,
    PublicStatus,
    VariableDomain,
    execute,
    parse_mps,
    read_mps,
)
from solverpilot.backends import ScipyHighsBackend, ScipyVendoredHighsDevBackend


BASIC = """NAME          BASIC
ROWS
 N  COST
 G  DEMAND
 L  CAP
COLUMNS
    X1        COST       1        DEMAND     1
    X1        CAP        1
    X2        COST       2        DEMAND     1
    X2        CAP        2
RHS
    RHS1      DEMAND     3        CAP        8
BOUNDS
 UP BND1      X1         5
ENDATA
"""


def test_parse_basic_lp_and_solve():
    p = parse_mps(BASIC)
    assert p.name == "BASIC"
    assert p.A.shape == (2, 2)
    assert p.metadata["variable_names"] == ("X1", "X2")
    assert p.metadata["constraint_names"] == ("DEMAND", "CAP")
    assert np.allclose(p.c, [1.0, 2.0])
    assert np.allclose(p.variable_lower, [0.0, 0.0])
    assert np.isclose(p.variable_upper[0], 5.0)
    assert np.isposinf(p.variable_upper[1])
    assert np.allclose(p.constraint_lower, [3.0, -np.inf])
    assert np.allclose(p.constraint_upper, [np.inf, 8.0])
    result = execute(p, ScipyHighsBackend())
    assert result.status is PublicStatus.VALID_OPTIMAL
    assert result.validation and result.validation.valid
    assert np.isclose(result.objective, 3.0)


def test_objective_rhs_is_negative_offset_and_objsense_maximize():
    text = """NAME OFFMAX
OBJSENSE
 MAX
ROWS
 N OBJ
 L C1
COLUMNS
    X OBJ 2 C1 1
RHS
    R C1 3 OBJ 5
BOUNDS
 LO B X 0
ENDATA
"""
    p = parse_mps(text)
    assert p.objective_sense is ObjectiveSense.MAXIMIZE
    assert p.objective_offset == -5.0
    result = execute(p, ScipyHighsBackend())
    assert result.status is PublicStatus.VALID_OPTIMAL
    assert np.isclose(result.x[0], 3.0)
    assert np.isclose(result.objective, 1.0)


def test_integer_markers_default_to_binary_and_general_integer_bounds_override():
    text = """NAME INTS
ROWS
 N OBJ
 L C1
COLUMNS
    M1 'MARKER' 'INTORG'
    X OBJ 1 C1 1
    Y OBJ 2 C1 1
    M2 'MARKER' 'INTEND'
RHS
    R C1 5
BOUNDS
 LI B Y 0
 UI B Y 4
ENDATA
"""
    p = parse_mps(text)
    assert p.domains.tolist() == [VariableDomain.BINARY.value, VariableDomain.INTEGER.value]
    assert np.allclose(p.variable_lower, [0.0, 0.0])
    assert np.allclose(p.variable_upper, [1.0, 4.0])


def test_li_alone_removes_marker_implicit_binary_upper():
    text = """NAME LI
ROWS
 N OBJ
 L C1
COLUMNS
    M1 'MARKER' 'INTORG'
    X OBJ 1 C1 1
    M2 'MARKER' 'INTEND'
RHS
    R C1 5
BOUNDS
 LI B X 0
ENDATA
"""
    p = parse_mps(text)
    assert p.domains.tolist() == [VariableDomain.INTEGER.value]
    assert np.isposinf(p.variable_upper[0])


def test_ranges_semantics_all_row_types():
    text = """NAME RANGES
ROWS
 N OBJ
 G GROW
 L LROW
 E EPOS
 E ENEG
COLUMNS
    X GROW 1 LROW 1
    X EPOS 1 ENEG 1
RHS
    R GROW 10 LROW 20
    R EPOS 30 ENEG 40
RANGES
    RG GROW -3 LROW 4
    RG EPOS 5 ENEG -6
ENDATA
"""
    p = parse_mps(text)
    assert np.allclose(p.constraint_lower, [10, 16, 30, 34])
    assert np.allclose(p.constraint_upper, [13, 20, 35, 40])


def test_duplicate_column_coefficients_are_summed():
    text = """NAME DUP
ROWS
 N OBJ
 E C1
COLUMNS
    X OBJ 1 C1 2
    X C1 3
RHS
    R C1 5
ENDATA
"""
    p = parse_mps(text)
    assert p.A[0, 0] == 5.0


def test_additional_n_rows_are_ignored_as_free_rows():
    text = """NAME NROWS
ROWS
 N OBJ
 N FREE
 E C1
COLUMNS
    X OBJ 1 FREE 99
    X C1 1
RHS
    R C1 2
ENDATA
"""
    p = parse_mps(text)
    assert p.A.shape == (1, 1)
    assert p.c[0] == 1.0


def test_negative_up_releases_default_zero_lower_bound():
    text = """NAME NEGUP
ROWS
 N OBJ
COLUMNS
    X OBJ 1
BOUNDS
 UP B X -2
ENDATA
"""
    p = parse_mps(text)
    assert np.isneginf(p.variable_lower[0])
    assert p.variable_upper[0] == -2.0


def test_gzip_read(tmp_path: Path):
    path = tmp_path / "basic.mps.gz"
    with gzip.open(path, "wt", encoding="ascii") as f:
        f.write(BASIC)
    p = read_mps(path)
    assert p.metadata["source_format"] == "mps"
    assert p.metadata["source_path"] == str(path)
    assert p.name == "BASIC"


def test_unsupported_semicontinuous_and_quadratic_are_hard_errors():
    semicon = """NAME S
ROWS
 N OBJ
COLUMNS
 X OBJ 1
BOUNDS
 SC B X 5
ENDATA
"""
    with pytest.raises(MPSUnsupportedFeatureError, match="semi-continuous"):
        parse_mps(semicon)

    quad = """NAME Q
ROWS
 N OBJ
COLUMNS
 X OBJ 1
QUADOBJ
 X X 2
ENDATA
"""
    with pytest.raises(MPSUnsupportedFeatureError, match="QUADOBJ"):
        parse_mps(quad)


def test_unknown_row_is_parse_error():
    bad = """NAME BAD
ROWS
 N OBJ
COLUMNS
 X NOPE 1
ENDATA
"""
    with pytest.raises(MPSParseError, match="unknown row"):
        parse_mps(bad)


@pytest.mark.skipif(
    importlib.util.find_spec("scipy.optimize._highspy._core") is None,
    reason="SciPy private vendored HiGHS unavailable",
)
def test_canonical_parser_matches_native_highs_file_reader(tmp_path: Path):
    # Cross-check parser semantics against the native HiGHS MPS reader for objective,
    # objective offset, integrality and ranges on a model representable by both paths.
    text = """NAME CROSS
OBJSENSE
 MIN
ROWS
 N OBJ
 G LO
 L HI
COLUMNS
    M0 'MARKER' 'INTORG'
    X OBJ 3 LO 1
    X HI 1
    M1 'MARKER' 'INTEND'
    Y OBJ 1 LO 1
    Y HI 2
RHS
    R LO 2 HI 5
    R OBJ 4
RANGES
    RG LO 2
BOUNDS
 LI B X 0
 UI B X 3
 LO B Y 0
ENDATA
"""
    path = tmp_path / "cross.mps"
    path.write_text(text, encoding="ascii")

    canonical = read_mps(path)
    ours = execute(canonical, ScipyVendoredHighsDevBackend())
    assert ours.status is PublicStatus.VALID_OPTIMAL

    from scipy.optimize._highspy._core import _Highs

    h = _Highs()
    h.setOptionValue("output_flag", False)
    assert "error" not in str(h.readModel(str(path))).lower()
    h.run()
    native_obj = float(h.getObjectiveValue())
    native_x = np.asarray(h.getSolution().col_value, dtype=float)

    assert np.isclose(float(ours.objective), native_obj, atol=1e-9)
    assert np.allclose(ours.x, native_x, atol=1e-8)
