"""Fail-closed auditing for predictive feature boundaries."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .errors import LeakageDetectedError
from .schema import FeatureRecord, FeatureSchema

_FORBIDDEN_FRAGMENTS = (
    "solver", "backend_result", "solve_result", "selected_backend", "selected_solver", "winner",
    "runtime", "wall_clock", "cpu_time", "regret", "target", "label", "prediction",
    "benchmark_result", "oracle", "quality_score", "best_known", "timeout", "solve_status",
    "optimality_status", "validation_result", "rank",
)


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


@dataclass(frozen=True, slots=True)
class LeakageFinding:
    code: str
    location: str
    fragment: str
    message: str


@dataclass(frozen=True, slots=True)
class LeakageAuditReport:
    passed: bool
    findings: tuple[LeakageFinding, ...]
    audited_feature_count: int
    audited_record_count: int

    def require_pass(self) -> None:
        if not self.passed:
            raise LeakageDetectedError(f"feature leakage audit failed with {len(self.findings)} finding(s)")


def audit_feature_boundary(schema: FeatureSchema, records: Iterable[FeatureRecord] = ()) -> LeakageAuditReport:
    rows = tuple(records)
    findings: list[LeakageFinding] = []
    for definition in schema.definitions:
        locations = [(f"definition.{definition.name}.name", definition.name)]
        locations.extend((f"definition.{definition.name}.source_fields", field) for field in definition.source_fields)
        for location, raw in locations:
            normalized = _norm(raw)
            for fragment in _FORBIDDEN_FRAGMENTS:
                if fragment in normalized:
                    findings.append(LeakageFinding(
                        "FORBIDDEN_PREDICTIVE_INPUT", location, fragment,
                        "feature boundary references solver outcome, runtime, target, oracle, or other post-solve evidence",
                    ))
    for record in rows:
        if record.feature_schema_id != schema.feature_schema_id:
            findings.append(LeakageFinding(
                "SCHEMA_MISMATCH", "record.feature_schema_id", "schema_mismatch",
                "record does not belong to the audited feature schema",
            ))
        undeclared = sorted(set(record.values) - set(schema.names))
        for name in undeclared:
            findings.append(LeakageFinding(
                "UNDECLARED_FEATURE", "record.values", name,
                "record contains an undeclared feature",
            ))
    return LeakageAuditReport(not findings, tuple(findings), len(schema.definitions), len(rows))
