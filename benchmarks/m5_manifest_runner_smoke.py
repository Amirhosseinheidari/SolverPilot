from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
import tempfile

RUNNER_PATH = Path(__file__).parent / "m5_public_mps_runner.py"
SPEC = importlib.util.spec_from_file_location("m5_public_mps_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)

OUT = Path(__file__).parent / "results" / "m5-manifest-runner-smoke.json"

MODELS = {
    "lp_min.mps": ("""NAME LPMIN
ROWS
 N OBJ
 G C
COLUMNS
 X OBJ 1 C 1
RHS
 R C 2
ENDATA
""", "=opt= lp_min 2"),
    "lp_max.mps": ("""NAME LPMAX
OBJSENSE
 MAX
ROWS
 N OBJ
 L C
COLUMNS
 X OBJ 2 C 1
RHS
 R C 3 OBJ 1
ENDATA
""", "=opt= lp_max 5"),
    "binary.mps": ("""NAME BIN
ROWS
 N OBJ
 G C
COLUMNS
 M0 'MARKER' 'INTORG'
 X OBJ 1 C 1
 Y OBJ 2 C 1
 M1 'MARKER' 'INTEND'
RHS
 R C 1
ENDATA
""", "=opt= binary 1"),
    "general_int.mps": ("""NAME GINT
ROWS
 N OBJ
 G C
COLUMNS
 M0 'MARKER' 'INTORG'
 X OBJ 1 C 1
 M1 'MARKER' 'INTEND'
RHS
 R C 3
BOUNDS
 LI B X 0
 UI B X 5
ENDATA
""", "=opt= general_int 3"),
    "ranged.mps": ("""NAME RANGE
ROWS
 N OBJ
 G C
COLUMNS
 X OBJ 1 C 1
RHS
 R C 2
RANGES
 RG C 3
ENDATA
""", "=opt= ranged 2"),
    "infeasible.mps": ("""NAME INF
ROWS
 N OBJ
 G LO
 L HI
COLUMNS
 X OBJ 1 LO 1
 X HI 1
RHS
 R LO 2 HI 1
ENDATA
""", "=inf= infeasible"),
    "best_known_max.mps": ("""NAME BEST
OBJSENSE
 MAX
ROWS
 N OBJ
 L C
COLUMNS
 X OBJ 1 C 1
RHS
 R C 4
ENDATA
""", "=best= best_known_max 3.5"),
}


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="optimind-m5-runner-") as tmp:
        root = Path(tmp)
        manifest = []
        solu = []
        for index, (filename, (text, ref)) in enumerate(MODELS.items()):
            # Exercise transparent gzip ingestion for alternating models.
            if index % 2:
                filename = filename + ".gz"
                with gzip.open(root / filename, "wt", encoding="ascii") as f:
                    f.write(text)
            else:
                (root / filename).write_text(text, encoding="ascii")
            manifest.append(filename)
            solu.append(ref.replace(ref.split()[1], RUNNER._instance_key(filename)))

        payload = RUNNER.run_manifest(
            manifest_text="\n".join(manifest) + "\n",
            solu_text="\n".join(solu) + "\n",
            data_dir=root,
            backend="scipy-highs-bridge",
            time_limit_s=None,
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "instances": payload["instances_in_manifest"],
        "state_counts": payload["state_counts"],
        "reference_checks": payload["reference_checks"],
        "reference_matches": payload["reference_matches"],
        "reference_mismatches": payload["reference_mismatches"],
        "reference_check_counts": payload["reference_check_counts"],
    }, indent=2))
    if payload["state_counts"] != {"solved": len(MODELS)} or payload["reference_mismatches"] != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
