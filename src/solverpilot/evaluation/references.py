from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class ObjectiveReference:
    name: str
    status: str
    objective: float | None


def parse_miplib_solu(text: str) -> dict[str, ObjectiveReference]:
    """Parse a MIPLIB-style ``.solu`` objective/status file.

    Supported records are ``=opt=``, ``=best=``, ``=inf=``, and ``=unkn=``.
    The parser is intentionally strict about duplicate names and malformed objectives so
    benchmark validation cannot silently use ambiguous reference data.
    """

    out: dict[str, ObjectiveReference] = {}
    tag_map = {
        "=opt=": "optimal",
        "=best=": "best_known",
        "=inf=": "infeasible",
        "=unkn=": "unknown",
    }
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2 or parts[0] not in tag_map:
            raise ValueError(f"line {line_no}: malformed .solu record")
        tag, name = parts[0], parts[1]
        if name in out:
            raise ValueError(f"line {line_no}: duplicate .solu instance {name!r}")
        status = tag_map[tag]
        objective = None
        if status in {"optimal", "best_known"}:
            if len(parts) != 3:
                raise ValueError(f"line {line_no}: {tag} record requires one objective")
            try:
                objective = float(parts[2])
            except ValueError as exc:
                raise ValueError(f"line {line_no}: invalid objective {parts[2]!r}") from exc
            if not isfinite(objective):
                raise ValueError(f"line {line_no}: objective must be finite")
        elif len(parts) != 2:
            raise ValueError(f"line {line_no}: {tag} record must not contain an objective")
        out[name] = ObjectiveReference(name=name, status=status, objective=objective)
    return out


def parse_test_manifest(text: str) -> tuple[str, ...]:
    """Parse a one-instance-per-line benchmark manifest such as MIPLIB ``.test``."""

    names: list[str] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if any(ch.isspace() for ch in line):
            raise ValueError(f"line {line_no}: manifest entry must be a single filename")
        if line in seen:
            raise ValueError(f"line {line_no}: duplicate manifest entry {line!r}")
        seen.add(line)
        names.append(line)
    if not names:
        raise ValueError("manifest contains no instances")
    return tuple(names)
