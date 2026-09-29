"""Native qualification: skipped unless explicit exact executables are supplied."""
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.native
def test_real_exact_runtime(tmp_path):
    scip, vipr = os.getenv("SOLVERPILOT_EXACT_SCIP"), os.getenv("SOLVERPILOT_VIPR")
    if not scip or not vipr:
        pytest.skip("explicit exact SCIP and VIPR executables required")
    evidence = Path("exact-native-evidence")
    evidence.mkdir(exist_ok=True)
    (tmp_path / "problem.lp").write_text(
        "Minimize\n obj: + 1 x0 + 1 x1\nSubject To\n"
        " c0: + 2 x0 + 2 x1 >= 3\nBounds\n 0 <= x0 <= 2\n 0 <= x1 <= 2\n"
        "Generals\n x0 x1\nEnd\n", encoding="ascii"
    )
    (tmp_path / "exact.set").write_text(
        'exact/enable = TRUE\ncertificate/filename = "proof.vipr"\n'
        'presolving/maxrounds = 0\npresolving/maxrestarts = 0\nlimits/time = 30\n'
        'separating/maxrounds = 0\nseparating/maxroundsroot = 0\n',
        encoding="ascii",
    )
    run = subprocess.run([scip, "-c", "set load exact.set",
                          "-c", "read problem.lp", "-c", "optimize", "-c", "quit"],
                         cwd=tmp_path, capture_output=True, text=True, timeout=45)
    (evidence / "scip.log").write_text(run.stdout + run.stderr, encoding="utf-8")
    for item in tmp_path.glob("proof*"):
        (evidence / item.name).write_bytes(item.read_bytes())
    assert run.returncode == 0, run.stdout + run.stderr
    checked = subprocess.run([vipr, str(tmp_path / "proof.vipr")], capture_output=True,
                             text=True, timeout=30)
    (evidence / "vipr.log").write_text(checked.stdout + checked.stderr, encoding="utf-8")
    assert checked.returncode == 0, checked.stdout + checked.stderr


@pytest.mark.native
@pytest.mark.parametrize("case", ["milp", "lp", "infeasible", "maximize", "binary", "fractional"])
def test_exact_adapter_original_model(case):
    from fractions import Fraction
    import numpy as np
    from solverpilot import LinearProblem
    from solverpilot.exact import solve_exact, verify_exact_certificate
    scip, vipr = os.getenv("SOLVERPILOT_EXACT_SCIP"), os.getenv("SOLVERPILOT_VIPR")
    if not scip or not vipr:
        pytest.skip("explicit exact SCIP and VIPR executables required")
    cases = {
        "milp": (LinearProblem([[2, 2]], [1, 1], [0, 0], [2, 2], [3], [np.inf], ["integer"] * 2), Fraction(2)),
        "lp": (LinearProblem([[3]], [1], [0], [2], [1], [np.inf], ["continuous"]), Fraction(1, 3)),
        "infeasible": (LinearProblem([[1]], [1], [0], [2], [0.5], [0.5], ["integer"]), None),
        "maximize": (LinearProblem([[1]], [1], [0], [3], [-np.inf], [2.5], ["integer"], "maximize", 7), Fraction(9)),
        "binary": (LinearProblem([[1, 1]], [2, 3], [0, 0], [1, 1], [1], [np.inf], ["binary"] * 2), Fraction(2)),
        "fractional": (LinearProblem([[0.1]], [0.3], [0], [3], [0.2], [np.inf], ["continuous"]), Fraction(0.3) * 2),
    }
    p, expected = cases[case]
    r = solve_exact(p, scip_executable=scip, checker_executable=vipr, time_limit=30,
                    evidence_directory=Path("exact-native-evidence") / case)
    (Path(r.evidence_directory) / "result.txt").write_text(repr(r), encoding="utf-8")
    assert r.independently_verified, r
    assert r.status == ("infeasible" if expected is None else "optimal"), r
    assert r.objective == expected, r
    certificate = Path(r.evidence_directory) / "checked.vipr"
    replay = verify_exact_certificate(p, certificate, checker_executable=vipr)
    assert replay.independently_verified and replay.objective == expected
    # A certificate from this model must not silently prove a different objective.
    wrong = LinearProblem(p.A, p.c * 2 + 1, p.variable_lower, p.variable_upper,
                          p.constraint_lower, p.constraint_upper, p.domains, p.objective_sense)
    assert not verify_exact_certificate(wrong, certificate, checker_executable=vipr).independently_verified


@pytest.mark.native
def test_native_checker_rejects_false_derivation(tmp_path):
    import numpy as np
    from solverpilot import LinearProblem
    from solverpilot.exact import verify_exact_certificate
    vipr = os.getenv("SOLVERPILOT_VIPR")
    if not vipr:
        pytest.skip("explicit VIPR executable required")
    p = LinearProblem([[1]], [1], [0], [3], [1], [np.inf], ["continuous"])
    payload = ("VER 1.0\nVAR 1\nx0\nINT 0\nOBJ min\n1 0 1\nCON 1 0\n"
               "r G 1 1 0 1\nRTP range 1 1\nSOL 1\ns 1 0 1\nDER 1\n"
               "d G 1 OBJ { lin 1 0 1 } -1\n")
    path = tmp_path / "proof.vipr"
    path.write_text(payload, encoding="ascii")
    assert verify_exact_certificate(p, path, checker_executable=vipr).independently_verified
    path.write_text(payload.replace("lin 1 0 1", "lin 1 0 0"), encoding="ascii")
    r = verify_exact_certificate(p, path, checker_executable=vipr)
    assert not r.independently_verified and r.status == "unverified"


@pytest.mark.native
def test_random_small_milps_match_exhaustive_reference():
    from fractions import Fraction
    from itertools import product
    import numpy as np
    from solverpilot import LinearProblem
    from solverpilot.exact import solve_exact
    scip, vipr = os.getenv("SOLVERPILOT_EXACT_SCIP"), os.getenv("SOLVERPILOT_VIPR")
    if not scip or not vipr:
        pytest.skip("explicit exact SCIP and VIPR executables required")
    rng = np.random.default_rng(290926)
    for seed in range(20):
        a = rng.integers(-4, 5, size=(2, 2))
        feasible = rng.integers(-2, 4, size=2)
        upper = a @ feasible + rng.integers(0, 3, size=2)
        c = rng.integers(-4, 5, size=2)
        values = [int(c @ x) for x in product(range(-2, 4), repeat=2)
                  if np.all(a @ x <= upper)]
        sense = "minimize" if seed % 2 else "maximize"
        expected = min(values) if sense == "minimize" else max(values)
        p = LinearProblem(a, c, [-2, -2], [3, 3], [-np.inf] * 2, upper,
                          ["integer"] * 2, sense, 0.5)
        result = solve_exact(p, scip_executable=scip, checker_executable=vipr,
                             evidence_directory=Path("exact-native-evidence") / f"random-{seed}")
        (Path(result.evidence_directory) / "result.txt").write_text(repr(result), encoding="utf-8")
        assert result.status == "optimal" and result.independently_verified, (seed, result)
        assert result.objective == Fraction(expected) + Fraction(1, 2)


@pytest.mark.native
def test_unbounded_and_budget_do_not_claim_proof():
    import numpy as np
    from solverpilot import LinearProblem
    from solverpilot.exact import solve_exact
    scip, vipr = os.getenv("SOLVERPILOT_EXACT_SCIP"), os.getenv("SOLVERPILOT_VIPR")
    if not scip or not vipr:
        pytest.skip("explicit exact SCIP and VIPR executables required")
    p = LinearProblem(np.zeros((0, 1)), [-1], [0], [np.inf], [], [], ["continuous"])
    r = solve_exact(p, scip_executable=scip, checker_executable=vipr,
                    evidence_directory=Path("exact-native-evidence") / "unbounded")
    assert r.status == "unverified" and not r.independently_verified, r
    r = solve_exact(p, scip_executable=scip, checker_executable=vipr, time_limit=0.000001)
    assert r.status == "unverified" and not r.independently_verified, r
