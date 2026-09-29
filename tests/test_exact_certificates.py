from fractions import Fraction
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import pytest

from solverpilot import LinearProblem
from solverpilot.exact import solve_exact, verify_exact_certificate
from solverpilot.exact.certificate import bind_certificate
from solverpilot.exact.problem import ExactModel
from solverpilot.exact import runtime


def problem(*, sense="minimize", offset=0, integer=False):
    return LinearProblem([[1.0]], [1.0], [0.0], [3.0], [1.0], [np.inf],
                         ["integer" if integer else "continuous"], sense, offset)


CERT = b"""VER 1.0
VAR 1
t_x0
INT 0
OBJ min
1 0 1
CON 3 2
lo G 0 1 0 1
hi L 3 1 0 1
r G 1 1 0 1
RTP range 1 1
SOL 1
s 1 0 1
DER 1
d G 1 OBJ { lin 1 2 1 } -1
"""


def test_exact_binary64_export_and_feasibility():
    p = LinearProblem([[0.1]], [0.3], [-np.inf], [1.5], [-np.inf], [0.2], ["integer"])
    m = ExactModel.from_problem(p)
    assert str(Fraction(0.1)) in m.lp_text()
    assert "Generals" in m.lp_text()
    assert not m.feasible([Fraction(3, 2)])
    assert m.feasible([Fraction(1)])
    assert not m.feasible([Fraction(3)])
    assert not m.feasible([])


def test_valid_binding_and_no_aliasing():
    p = problem(offset=10)
    m = ExactModel.from_problem(p)
    b = bind_certificate(m, CERT)
    assert b.lower == b.upper == 1
    assert m.objective(b.solutions[0]) == 11
    p.c.flags.writeable = True
    p.c[0] = 7
    assert m.c == (1,)


@pytest.mark.parametrize("before,after", [
    (b"VAR 1", b"VAR 2000000000"),
    (b"t_x0", b"t_x1"),
    (b"INT 0", b"INT 1\n0"),
    (b"OBJ min\n1 0 1", b"OBJ min\n1 0 2"),
    (b"hi L 3", b"hi L 2"),
    (b"r G 1", b"r G 2"),
    (b"s 1 0 1", b"s 1 0 0"),
    (b"RTP range 1 1", b"RTP range 2 1"),
    (b"RTP range 1 1", b"RTP range 0 0"),
    (b"RTP range 1 1", b"RTP infeas"),
    (b"lin 1 2 1", b"lin 1 4 1"),
    (b"lin 1 2 1", b"lin 2 2 1 2 1"),
    (b"lin 1 2 1", b"weak 1 2 1"),
    (b"VER 1.0", b"VER 1.1"),
    (b"1 0 1", b"1 0 1/0"),
])
def test_adversarial_binding(before, after):
    with pytest.raises(ValueError):
        bind_certificate(ExactModel.from_problem(problem()), CERT.replace(before, after))


def test_truncation_trailing_and_duplicate_variables():
    m = ExactModel.from_problem(problem())
    for payload in (CERT[:-10], CERT + b"ignored trailing junk", CERT.replace(b"VAR 1\nt_x0", b"VAR 2\nx0 t_x0")):
        with pytest.raises(ValueError):
            bind_certificate(m, payload)


def test_positive_row_scaling_is_equivalent():
    bind_certificate(ExactModel.from_problem(problem()), CERT.replace(b"r G 1 1 0 1", b"r G 2 1 0 2"))


def test_integer_bound_rounding():
    p = LinearProblem([[1]], [1], [0.5], [3.5], [1], [np.inf], ["integer"])
    payload = CERT.replace(b"INT 0", b"INT 1\n0").replace(b"lo G 0", b"lo G 1")
    bind_certificate(ExactModel.from_problem(p), payload)


def test_trusted_checker_contract_and_exact_result(tmp_path, monkeypatch):
    cert = tmp_path / "certificate.vipr"
    cert.write_bytes(CERT)
    def run(command, directory, log_name, deadline):
        (directory / log_name).write_text("Successfully verified.\n")
    monkeypatch.setattr(runtime, "_run", run)
    r = verify_exact_certificate(problem(offset=10), cert, checker_executable=sys.executable,
                                 evidence_directory=tmp_path)
    assert r.status == "optimal" and r.independently_verified
    assert r.objective == r.bound == Fraction(11)
    assert r.absolute_gap == 0 and r.x == (Fraction(1),)
    assert r.certificate_sha256 and r.checker_sha256
    assert Path(r.evidence_directory, "checked.vipr").read_bytes() == CERT


@pytest.mark.parametrize("mode", ["negative", "empty", "crash", "timeout", "mutated"])
def test_checker_failure_never_verifies(tmp_path, monkeypatch, mode):
    cert = tmp_path / "certificate.vipr"
    cert.write_bytes(CERT)
    def run(command, directory, log_name, deadline):
        if mode == "crash":
            raise RuntimeError("checker failed")
        if mode == "timeout":
            raise TimeoutError("checker timed out")
        (directory / log_name).write_text("Verification failed." if mode == "negative" else
                                          "" if mode == "empty" else "Successfully verified.")
        if mode == "mutated":
            (directory / "checked.vipr").write_bytes(CERT + b" ")
    monkeypatch.setattr(runtime, "_run", run)
    r = verify_exact_certificate(problem(), cert, checker_executable=sys.executable)
    assert not r.independently_verified and r.status == "unverified"


def test_real_process_timeout_reaps_child(tmp_path):
    with pytest.raises(TimeoutError):
        runtime._run([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path,
                     "timeout.log", monotonic() + 0.1)
    # Open without sharing conflicts after the child has actually exited.
    (tmp_path / "timeout.log").unlink()


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf"), True, "2"])
def test_invalid_budget(budget):
    with pytest.raises(ValueError):
        solve_exact(problem(), scip_executable=sys.executable, checker_executable=sys.executable,
                    time_limit=budget)


def test_explicit_native_paths_required():
    with pytest.raises(ValueError):
        solve_exact(problem(), scip_executable="scip", checker_executable="viprchk")
