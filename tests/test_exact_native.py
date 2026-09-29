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
        'presolving/maxrounds = 0\npresolving/maxrestarts = 0\nlimits/time = 30\n',
        encoding="ascii",
    )
    run = subprocess.run([scip, "-c", "set load exact.set", "-c", "set separating off",
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
