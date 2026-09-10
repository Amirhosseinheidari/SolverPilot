from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isclose, isfinite
from statistics import fmean, median, pstdev
from typing import Iterable, Mapping, Any

from .oracle import ObjectiveSense, OracleEntry, OracleStatus, OracleTable


class RegretStatus(str, Enum):
    COMPARED = "compared"
    ORACLE_UNAVAILABLE = "oracle_unavailable"
    INSTANCE_MISMATCH = "instance_mismatch"
    SELECTED_RUN_MISSING = "selected_run_missing"
    SELECTED_RUN_INELIGIBLE = "selected_run_ineligible"


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    decision_id: str
    instance: str
    selected_run_id: str | None
    selector_id: str = "policy"
    selector_version: str = "unknown"
    selection_overhead_s: float = 0.0

    def __post_init__(self) -> None:
        for name in ("decision_id", "instance", "selector_id", "selector_version"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        if self.selected_run_id is not None:
            value = str(self.selected_run_id).strip()
            if not value:
                raise ValueError("selected_run_id must be non-empty when provided")
            object.__setattr__(self, "selected_run_id", value)
        overhead = float(self.selection_overhead_s)
        if not isfinite(overhead) or overhead < 0:
            raise ValueError("selection_overhead_s must be finite and non-negative")
        object.__setattr__(self, "selection_overhead_s", overhead)


@dataclass(frozen=True, slots=True)
class RegretVector:
    decision_id: str
    selector_id: str
    selector_version: str
    instance: str
    selected_run_id: str | None
    oracle_run_ids: tuple[str, ...]
    status: RegretStatus
    comparable: bool
    quality_regret: float | None
    relative_quality_regret: float | None
    runtime_delta_s: float | None
    selection_overhead_s: float
    runtime_regret_s: float | None
    failure_regret: int
    timeout_regret: int
    limit_regret: int
    infeasibility_regret: int
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MetricSummary:
    observed_count: int
    missing_count: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    median: float | None
    population_stddev: float | None


@dataclass(frozen=True, slots=True)
class SelectorRegretSummary:
    selector_id: str
    selector_version: str
    decision_count: int
    compared_count: int
    incomparable_count: int
    quality_regret: MetricSummary
    relative_quality_regret: MetricSummary
    runtime_delta_s: MetricSummary
    runtime_regret_s: MetricSummary
    failure_regret_count: int
    timeout_regret_count: int
    limit_regret_count: int
    infeasibility_regret_count: int
    significance_claim_supported: bool = False


def _summary(values: list[float | None]) -> MetricSummary:
    observed = [float(v) for v in values if v is not None]
    missing = len(values) - len(observed)
    if not observed:
        return MetricSummary(0, missing, None, None, None, None, None)
    return MetricSummary(
        observed_count=len(observed),
        missing_count=missing,
        minimum=min(observed),
        maximum=max(observed),
        mean=fmean(observed),
        median=float(median(observed)),
        population_stddev=pstdev(observed) if len(observed) > 1 else 0.0,
    )


def _row_runtime(row: Mapping[str, Any], field: str) -> float | None:
    current: Any = row
    for part in field.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    try:
        value = float(current)
    except (TypeError, ValueError):
        return None
    return value if isfinite(value) and value >= 0 else None


def _binary_failure_dimensions(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    state = str(row.get("state") or "")
    public_status = str(row.get("public_status") or "")
    timeout = int(state == "hard_timeout")
    limit = int(public_status == "feasible_limit")
    infeasible = int(public_status == "infeasible")
    failure = int(
        state in {"missing_instance", "parse_error", "solve_error", "worker_error"}
        or public_status in {"invalid_solution", "error", "unknown", "infeasible_or_unbounded"}
    )
    return failure, timeout, limit, infeasible


def _directional_regret(selected: float, oracle: float, sense: ObjectiveSense, *, atol: float, rtol: float) -> float:
    if isclose(selected, oracle, abs_tol=atol, rel_tol=rtol):
        return 0.0
    raw = selected - oracle if sense is ObjectiveSense.MINIMIZE else oracle - selected
    # An eligible oracle is extremal, but clamping tiny negative numerical drift keeps the
    # metric non-negative if values sit just outside the explicit tolerance boundary.
    return max(0.0, float(raw))


def evaluate_selection_decisions(
    decisions: Iterable[SelectionDecision],
    rows: Iterable[Mapping[str, Any]],
    oracle_table: OracleTable,
) -> tuple[RegretVector, ...]:
    decision_rows = tuple(decisions)
    benchmark_rows = tuple(rows)
    if len({d.decision_id for d in decision_rows}) != len(decision_rows):
        raise ValueError("decision_id values must be unique")
    by_run: dict[str, Mapping[str, Any]] = {}
    for row in benchmark_rows:
        run_id = str(row.get("run_id") or "")
        if not run_id:
            raise ValueError("every benchmark row must provide run_id")
        if run_id in by_run:
            raise ValueError(f"duplicate benchmark run_id: {run_id}")
        by_run[run_id] = row
    entries = oracle_table.by_instance()
    vectors: list[RegretVector] = []

    def emit(decision: SelectionDecision, entry: OracleEntry | None, status: RegretStatus, reasons: tuple[str, ...], *, selected_row: Mapping[str, Any] | None = None) -> None:
        failure = timeout = limit = infeasible = 0
        if selected_row is not None:
            failure, timeout, limit, infeasible = _binary_failure_dimensions(selected_row)
        vectors.append(RegretVector(
            decision_id=decision.decision_id,
            selector_id=decision.selector_id,
            selector_version=decision.selector_version,
            instance=decision.instance,
            selected_run_id=decision.selected_run_id,
            oracle_run_ids=() if entry is None else entry.oracle_run_ids,
            status=status,
            comparable=False,
            quality_regret=None,
            relative_quality_regret=None,
            runtime_delta_s=None,
            selection_overhead_s=decision.selection_overhead_s,
            runtime_regret_s=None,
            failure_regret=failure,
            timeout_regret=timeout,
            limit_regret=limit,
            infeasibility_regret=infeasible,
            reasons=reasons,
        ))

    for decision in sorted(decision_rows, key=lambda d: d.decision_id):
        entry = entries.get(decision.instance)
        if entry is None:
            emit(decision, None, RegretStatus.INSTANCE_MISMATCH, ("decision instance is absent from oracle table",))
            continue
        if entry.status is not OracleStatus.AVAILABLE:
            emit(decision, entry, RegretStatus.ORACLE_UNAVAILABLE, entry.reasons or (f"oracle status is {entry.status.value}",))
            continue
        if decision.selected_run_id is None:
            emit(decision, entry, RegretStatus.SELECTED_RUN_MISSING, ("selector abstained or did not provide a run_id",))
            continue
        selected = by_run.get(decision.selected_run_id)
        if selected is None:
            emit(decision, entry, RegretStatus.SELECTED_RUN_MISSING, ("selected run is absent from benchmark rows",))
            continue
        if str(selected.get("instance") or "") != decision.instance:
            emit(decision, entry, RegretStatus.INSTANCE_MISMATCH, ("selected run belongs to a different instance",), selected_row=selected)
            continue
        candidate_by_id = {c.run_id: c for c in entry.candidates}
        candidate = candidate_by_id.get(decision.selected_run_id)
        if candidate is None:
            exclusion = next((x for x in entry.exclusions if x.run_id == decision.selected_run_id), None)
            reason = "selected run is not oracle-eligible" if exclusion is None else f"selected run excluded: {exclusion.reason.value}: {exclusion.detail}"
            emit(decision, entry, RegretStatus.SELECTED_RUN_INELIGIBLE, (reason,), selected_row=selected)
            continue
        if entry.oracle_objective is None or entry.fastest_oracle_runtime_s is None:
            raise RuntimeError("oracle-eligible candidate is missing objective/runtime invariants")
        quality = _directional_regret(
            candidate.objective,
            entry.oracle_objective,
            entry.objective_sense,
            atol=oracle_table.policy.objective_atol,
            rtol=oracle_table.policy.objective_rtol,
        )
        if abs(entry.oracle_objective) <= oracle_table.policy.objective_atol:
            relative = None
            reasons = ("relative quality regret is undefined because oracle objective is zero",)
        else:
            relative = quality / abs(entry.oracle_objective)
            reasons = ()
        runtime_delta = candidate.runtime_s - entry.fastest_oracle_runtime_s
        runtime_regret = max(0.0, candidate.runtime_s + decision.selection_overhead_s - entry.fastest_oracle_runtime_s)
        vectors.append(RegretVector(
            decision_id=decision.decision_id,
            selector_id=decision.selector_id,
            selector_version=decision.selector_version,
            instance=decision.instance,
            selected_run_id=decision.selected_run_id,
            oracle_run_ids=entry.oracle_run_ids,
            status=RegretStatus.COMPARED,
            comparable=True,
            quality_regret=quality,
            relative_quality_regret=relative,
            runtime_delta_s=runtime_delta,
            selection_overhead_s=decision.selection_overhead_s,
            runtime_regret_s=runtime_regret,
            failure_regret=0,
            timeout_regret=0,
            limit_regret=0,
            infeasibility_regret=0,
            reasons=reasons,
        ))
    return tuple(vectors)


def summarize_regret_vectors(vectors: Iterable[RegretVector]) -> tuple[SelectorRegretSummary, ...]:
    rows = tuple(vectors)
    grouped: dict[tuple[str, str], list[RegretVector]] = {}
    for row in rows:
        grouped.setdefault((row.selector_id, row.selector_version), []).append(row)
    summaries: list[SelectorRegretSummary] = []
    for (selector_id, selector_version), group in sorted(grouped.items()):
        summaries.append(SelectorRegretSummary(
            selector_id=selector_id,
            selector_version=selector_version,
            decision_count=len(group),
            compared_count=sum(int(v.comparable) for v in group),
            incomparable_count=sum(int(not v.comparable) for v in group),
            quality_regret=_summary([v.quality_regret for v in group]),
            relative_quality_regret=_summary([v.relative_quality_regret for v in group]),
            runtime_delta_s=_summary([v.runtime_delta_s for v in group]),
            runtime_regret_s=_summary([v.runtime_regret_s for v in group]),
            failure_regret_count=sum(v.failure_regret for v in group),
            timeout_regret_count=sum(v.timeout_regret for v in group),
            limit_regret_count=sum(v.limit_regret for v in group),
            infeasibility_regret_count=sum(v.infeasibility_regret for v in group),
            significance_claim_supported=False,
        ))
    return tuple(summaries)
