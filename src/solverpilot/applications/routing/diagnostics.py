from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .model import VRPInstance
from .solution import VRPSolution
from .validation import VRPValidationReport, validate_vrp_solution


@dataclass(frozen=True, slots=True)
class VRPDiagnosticAction:
    code: str
    target: str
    severity: str
    suggestion: str
    evidence_code: str


_ACTIONS = {
    "capacity_exceeded": ("blocking", "Split the route, reassign customers, or use a vehicle with more capacity."),
    "time_window_violation": ("blocking", "Resequence the stop earlier, relax its time window, split the route, or reassign it."),
    "vehicle_shift_late_return": ("blocking", "Split or shorten the route, extend the vehicle shift, or reassign late stops."),
    "max_route_duration_exceeded": ("blocking", "Split the route, add a vehicle, or explicitly increase the permitted route duration."),
    "unassigned_customer": ("blocking", "Add capacity/vehicles or relax the blocking routing constraints before retrying assignment."),
    "duplicate_customer_assignment": ("blocking", "Remove duplicate service and recompute the route plan."),
    "objective_mismatch": ("blocking", "Discard the reported objective and recompute it from the route and distance matrix."),
    "unknown_vehicle": ("blocking", "Repair the vehicle reference before using the plan."),
    "unknown_customer": ("blocking", "Repair the customer reference before using the plan."),
}


def diagnose_vrp_validation(report: VRPValidationReport) -> tuple[VRPDiagnosticAction, ...]:
    actions: list[VRPDiagnosticAction] = []
    seen: set[tuple[str, str]] = set()
    for issue in report.issues:
        severity, suggestion = _ACTIONS.get(issue.code, ("warning", "Review the validation evidence and repair the route before operational use."))
        target = issue.customer_id or issue.vehicle_id or "solution"
        key = (issue.code, target)
        if key in seen:
            continue
        seen.add(key)
        actions.append(VRPDiagnosticAction(issue.code, target, severity, suggestion, issue.code))
    return tuple(actions)


def diagnose_vrp_solution(instance: VRPInstance, solution: VRPSolution, *, waiting_warning_threshold: float = 30.0) -> tuple[VRPDiagnosticAction, ...]:
    """Build actionable diagnostics without promoting suggestions to proofs.

    Validation blockers are emitted first.  Feasible waiting time is reported as
    informational/warning evidence because it can be operationally useful but is
    not itself an infeasibility.
    """
    if isinstance(waiting_warning_threshold, bool):
        raise ValueError("waiting_warning_threshold must be numeric")
    threshold = float(waiting_warning_threshold)
    if not isfinite(threshold) or threshold < 0.0:
        raise ValueError("waiting_warning_threshold must be finite and non-negative")
    report = validate_vrp_solution(instance, solution)
    actions = list(diagnose_vrp_validation(report))
    seen = {(a.code, a.target) for a in actions}
    for timeline in report.timelines:
        for visit in timeline.visits:
            if visit.waiting <= 1e-12:
                continue
            key = ("waiting_time_detected", visit.customer_id)
            if key in seen:
                continue
            seen.add(key)
            severity = "warning" if visit.waiting >= threshold else "info"
            actions.append(VRPDiagnosticAction(
                "waiting_time_detected",
                visit.customer_id,
                severity,
                "Accept the wait if it protects the time window, or consider resequencing; this is a suggestion, not an automatic improvement claim.",
                "timeline_waiting",
            ))
    return tuple(actions)
