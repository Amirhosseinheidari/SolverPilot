"""Comparable reuse observations, without inferring hidden native factorization."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReuseEvidence:
    workspace: str = "unknown"
    primal_dual_start: str = "unknown"
    symbolic_factorization: str = "unknown"
    numeric_factorization: str = "unknown"
    matrix_update: bool | None = None
    reason: str = "adapter did not provide reuse observations"


def reuse_evidence(result):
    """Read adapter observations; enabled warm starts are not observed reuse."""
    raw = getattr(result, "raw_statistics", None)
    report = raw.get("reuse_report") if isinstance(raw, Mapping) else None
    if not isinstance(report, Mapping):
        return ReuseEvidence()
    states = {"observed", "not_used", "unknown", "automatic_enabled", "not_applicable"}
    fields = {
        name: report.get(name, "unknown")
        for name in (
            "workspace",
            "primal_dual_start",
            "symbolic_factorization",
            "numeric_factorization",
        )
    }
    fields = {
        name: value if isinstance(value, str) and value in states else "unknown"
        for name, value in fields.items()
    }
    update = report.get("matrix_update")
    return ReuseEvidence(
        **fields,
        matrix_update=update if type(update) is bool else None,
        reason=str(report.get("reason", "adapter-reported observation")),
    )
