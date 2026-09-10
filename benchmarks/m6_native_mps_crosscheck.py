from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from solverpilot import read_mps, solve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mps")
    ap.add_argument("--output", default="benchmarks/results/m6-native-mps-crosscheck.json")
    args = ap.parse_args()

    try:
        from scipy.optimize._highspy import _core as core
    except Exception as exc:  # private development verifier only
        raise SystemExit(f"SciPy vendored HiGHS unavailable: {exc}")

    path = Path(args.mps)
    p = read_mps(path)
    ours = solve(p, backend="scipy-highs-ds")

    h = core._Highs()
    h.setOptionValue("output_flag", False)
    read_status = h.readModel(str(path))
    run_status = h.run()
    native_status = h.modelStatusToString(h.getModelStatus())
    native_obj = float(h.getObjectiveValue())
    out = {
        "input_path": str(path.resolve()),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "read_status": str(read_status),
        "run_status": str(run_status),
        "native_model_status": str(native_status),
        "native_objective": native_obj,
        "canonical_ir_objective": ours.objective,
        "objective_abs_diff": abs(native_obj - float(ours.objective)),
        "canonical_validation_valid": bool(ours.validation and ours.validation.valid),
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
