"""Structured, claim-safe explanations for SolverPilot solve results.

This package is intentionally not re-exported from the frozen top-level ``solverpilot``
namespace. It explains existing ``SolveResult`` evidence without changing solve behavior.
"""

from .explain import explain_result, explain_status
from .model import (
    REPORT_SCHEMA_VERSION,
    ClaimDisposition,
    ClaimKind,
    ExplanationClaim,
    ExplanationReport,
)
from .render import render_json, render_markdown

__all__ = [
    "REPORT_SCHEMA_VERSION",
    "ClaimDisposition",
    "ClaimKind",
    "ExplanationClaim",
    "ExplanationReport",
    "explain_result",
    "explain_status",
    "render_json",
    "render_markdown",
]

from .quality import solution_quality
__all__.append("solution_quality")
