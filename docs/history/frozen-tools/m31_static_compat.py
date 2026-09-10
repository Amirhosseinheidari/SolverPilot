from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

errors: list[dict[str, object]] = []
files = sorted(SRC.rglob("*.py"))
for path in files:
    text = path.read_text(encoding="utf-8")
    try:
        ast.parse(text, filename=str(path), feature_version=(3, 10))
    except SyntaxError as exc:
        errors.append({"path": str(path.relative_to(ROOT)), "line": exc.lineno, "message": exc.msg})

payload = {
    "schema": "solverpilot.m31.static_python_compat.v1",
    "target": "python-3.10-syntax",
    "files_checked": len(files),
    "syntax_errors": errors,
    "passed": not errors,
    "claim_boundary": "Syntax compatibility only; this is not runtime compatibility evidence.",
}
print(json.dumps(payload, indent=2, sort_keys=True))
raise SystemExit(0 if not errors else 2)
