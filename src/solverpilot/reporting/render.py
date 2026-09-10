from __future__ import annotations

import html
import json

from .model import ExplanationReport


def render_json(report: ExplanationReport, *, indent: int | None = 2) -> str:
    if not isinstance(report, ExplanationReport):
        raise TypeError("render_json expects an ExplanationReport")
    if not report.claim_safe:
        raise ValueError("refusing to render an explanation report that violates claim-safety invariants")
    return json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True, indent=indent, allow_nan=False)


def _md(value: object) -> str:
    text = html.escape(str(value), quote=False)
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def render_markdown(report: ExplanationReport) -> str:
    if not isinstance(report, ExplanationReport):
        raise TypeError("render_markdown expects an ExplanationReport")
    if not report.claim_safe:
        raise ValueError("refusing to render an explanation report that violates claim-safety invariants")
    lines = [
        "# SolverPilot explanation",
        "",
        f"**Status:** `{_md(report.status)}`",
        f"**Problem class:** `{_md(report.problem_class)}`",
        f"**Backend:** `{_md(report.backend if report.backend is not None else 'unknown')}`",
        f"**Objective:** `{_md(report.objective if report.objective is not None else 'not available')}`",
        "",
        "## Summary",
        _md(report.summary),
        "",
        "## Status interpretation",
        _md(report.status_explanation),
        "",
        "## Claims",
        "| Claim | Disposition | Statement |",
        "| --- | --- | --- |",
    ]
    for claim in report.claims:
        lines.append(
            f"| `{_md(claim.claim_id)}` | `{_md(claim.disposition.value)}` | {_md(claim.statement)} |"
        )
        if claim.qualifiers:
            for qualifier in claim.qualifiers:
                lines.append(f"  - **Qualifier:** {_md(qualifier)}")
    if report.warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {_md(warning)}" for warning in report.warnings)
    return "\n".join(lines) + "\n"
