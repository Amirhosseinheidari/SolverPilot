"""An auditable, finite continuous-production scenario workflow.

The caller supplies the business contract. Checks establish agreement with that
explicit contract, not completeness of unstated business requirements.
"""

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from collections.abc import Mapping

import numpy as np

from solverpilot._identity import new_execution_id
from solverpilot._immutability import deep_freeze, readonly_array
from solverpilot.problem import LinearProblem, ObjectiveSense
from solverpilot.runtime.evidence import evidence_bundle, verify_evidence_bundle
from solverpilot.runtime.manifest import _seal, _verify_integrity, json_value, _replay_payload
from solverpilot.runtime.options import SolveOptions
from solverpilot.scenarios import scenario_sweep
from .templates import production_model


def _array(value, name, *, shape=None, nonnegative=False, allow_inf=False):
    array = np.asarray(value, dtype=float)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if np.isnan(array).any() or (not allow_inf and not np.isfinite(array).all()):
        raise ValueError(f"{name} must contain finite numbers")
    if nonnegative and np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    return readonly_array(array)


def _names(values, count, prefix):
    result = tuple(f"{prefix}[{i}]" for i in range(count)) if values is None else tuple(values)
    if len(result) != count or any(not isinstance(v, str) or not v.strip() for v in result):
        raise ValueError(f"{prefix} names must be nonempty and match dimensions")
    if len(set(result)) != len(result):
        raise ValueError(f"{prefix} names must be unique")
    return result


@dataclass(frozen=True, slots=True, eq=False)
class ProductionContract:
    """Immutable dense contract: maximize profit, continuous quantities, fixed units.

    Resource coefficients are nonnegative; minimum commitments and maximum
    quantities are fixed across a study. Scenarios may change capacity or profit.
    Unit labels are supplied by the caller; dimensional correctness is not inferred.
    """

    profit: object
    resources: object
    capacity: object
    maximum: object = None
    minimum: object = None
    product_names: tuple[str, ...] | None = None
    resource_names: tuple[str, ...] | None = None
    objective_unit: str = "unspecified"

    def __post_init__(self):
        profit = _array(self.profit, "profit")
        capacity = _array(self.capacity, "capacity", nonnegative=True)
        if profit.ndim != 1 or capacity.ndim != 1 or not profit.size or not capacity.size:
            raise ValueError("profit and capacity must be nonempty vectors")
        resources = _array(self.resources, "resources", shape=(capacity.size, profit.size), nonnegative=True)
        minimum = _array(np.zeros(profit.size) if self.minimum is None else self.minimum,
                         "minimum", shape=profit.shape, nonnegative=True)
        maximum = _array(np.full(profit.size, np.inf) if self.maximum is None else self.maximum,
                         "maximum", shape=profit.shape, nonnegative=True, allow_inf=True)
        if np.any(maximum < minimum):
            raise ValueError("maximum must be at least minimum")
        if not isinstance(self.objective_unit, str) or not self.objective_unit.strip():
            raise ValueError("objective_unit must be nonempty")
        for name, value in (("profit", profit), ("resources", resources), ("capacity", capacity),
                            ("minimum", minimum), ("maximum", maximum)):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "product_names", _names(self.product_names, profit.size, "product"))
        object.__setattr__(self, "resource_names", _names(self.resource_names, capacity.size, "resource"))

    def to_dict(self):
        return json_value({"schema": "solverpilot.production-contract.v1", **{
            name: getattr(self, name) for name in (
                "profit", "resources", "capacity", "maximum", "minimum",
                "product_names", "resource_names", "objective_unit",
            )
        }})

    @classmethod
    def from_dict(cls, payload):
        data = dict(payload)
        if data.pop("schema", "solverpilot.production-contract.v1") != "solverpilot.production-contract.v1":
            raise ValueError("unsupported production contract schema")
        return cls(**data)

    @property
    def data_hash(self):
        from solverpilot.benchmark.integrity import canonical_sha256
        return canonical_sha256(self.to_dict())

    def updated(self, updates):
        if not isinstance(updates, Mapping) or set(updates) - {"profit", "capacity"}:
            raise ValueError("production scenarios only support profit and capacity updates")
        candidate = replace(self, **dict(updates))
        if candidate.profit.shape != self.profit.shape or candidate.capacity.shape != self.capacity.shape:
            raise ValueError("scenario dimensions cannot change")
        return candidate

    def build_model(self):
        return production_model(self.profit, self.resources, self.capacity,
                                maximum=self.maximum, minimum=self.minimum)


@dataclass(frozen=True, slots=True)
class FormulationCheck:
    matches: bool
    issues: tuple[str, ...]
    scope: str = "exact ordered continuous-production template contract v1"


def check_production_formulation(contract, problem):
    """Audit canonical data independently of the template/compiler implementation.

    Both a <= b and -a >= -b row orientations are recognized. Reordering,
    scaling or another reformulation is outside this deliberately narrow audit.
    A mismatch is not a claim that an arbitrary equivalent model is incorrect.
    """
    if not isinstance(contract, ProductionContract):
        raise TypeError("contract must be ProductionContract")
    if not isinstance(problem, LinearProblem):
        return FormulationCheck(False, ("expected a canonical LinearProblem",))
    issues = []
    if problem.A.shape != contract.resources.shape:
        issues.append("resource/variable dimensions differ")
    if problem.objective_sense != ObjectiveSense.MAXIMIZE:
        issues.append("objective must maximize profit")
    if not np.array_equal(problem.c, contract.profit) or problem.objective_offset != 0:
        issues.append("profit coefficients or objective offset differ")
    if not np.array_equal(problem.variable_lower, contract.minimum):
        issues.append("minimum production commitments differ")
    if not np.array_equal(problem.variable_upper, contract.maximum):
        issues.append("maximum production quantities differ")
    if not np.all(problem.domains == "continuous"):
        issues.append("production variables must be continuous")
    if problem.A.shape == contract.resources.shape:
        for i, name in enumerate(contract.resource_names):
            row = problem.A.getrow(i).toarray().ravel()
            forward = (np.array_equal(row, contract.resources[i])
                       and np.isneginf(problem.constraint_lower[i])
                       and problem.constraint_upper[i] == contract.capacity[i])
            reverse = (np.array_equal(row, -contract.resources[i])
                       and problem.constraint_lower[i] == -contract.capacity[i]
                       and np.isposinf(problem.constraint_upper[i]))
            if not (forward or reverse):
                issues.append(f"resource capacity row differs: {name}")
    return FormulationCheck(not issues, tuple(issues))


def _tolerance(atol, rtol):
    if any(isinstance(v, bool) or not np.isfinite(v) or v < 0 for v in (atol, rtol)):
        raise ValueError("semantic tolerances must be finite and nonnegative")


def check_production_candidate(contract, quantities, objective, *, atol=1e-7, rtol=1e-9):
    """Check raw contract inequalities and profit without trusting the compiled model."""
    _tolerance(atol, rtol)
    output = {"candidate_present": quantities is not None, "feasible": False,
              "objective_consistent": False, "objective": None, "resource_usage": None,
              "resource_slack": None, "violations": [], "atol": atol, "rtol": rtol}
    if quantities is None:
        return deep_freeze(output)
    x = np.asarray(quantities, dtype=float)
    if x.shape != contract.profit.shape or not np.isfinite(x).all():
        output["violations"].append("candidate must be a finite vector matching products")
        return deep_freeze(output)
    with np.errstate(over="ignore", invalid="ignore"):
        usage, profit = contract.resources @ x, float(contract.profit @ x)
    if not np.isfinite(usage).all() or not np.isfinite(profit):
        output["violations"].append("contract evaluation overflow")
        return deep_freeze(output)
    for i, name in enumerate(contract.product_names):
        if x[i] < contract.minimum[i] - (atol + rtol * abs(contract.minimum[i])):
            output["violations"].append(f"minimum commitment violated: {name}")
        if np.isfinite(contract.maximum[i]) and x[i] > contract.maximum[i] + (atol + rtol * abs(contract.maximum[i])):
            output["violations"].append(f"maximum quantity violated: {name}")
    for i, name in enumerate(contract.resource_names):
        if usage[i] > contract.capacity[i] + atol + rtol * abs(contract.capacity[i]):
            output["violations"].append(f"resource capacity violated: {name}")
    consistent = (objective is not None and np.isfinite(objective)
                  and abs(profit - objective) <= atol + rtol * max(abs(profit), abs(objective)))
    output.update(feasible=not output["violations"], objective_consistent=bool(consistent),
                  objective=profit, resource_usage=usage, resource_slack=contract.capacity - usage)
    if not consistent:
        output["violations"].append("reported objective differs from contract profit")
    return deep_freeze(output)


def _capacity_diagnosis(contract, *, atol, rtol):
    """Nonnegative consumption gives a direct necessary lower resource bound."""
    with np.errstate(over="ignore", invalid="ignore"):
        required = contract.resources @ contract.minimum
        tolerated_lower = contract.minimum - (atol + rtol * np.abs(contract.minimum))
        tolerated_required = contract.resources @ tolerated_lower
        # Guard dot-product roundoff as well as the candidate check's permitted
        # lower-bound and resource errors; this is not an exact certificate.
        rounding = (contract.resources @ np.abs(tolerated_lower)) * np.finfo(float).eps * (2 * contract.profit.size + 4)
    shortfalls = []
    for i, name in enumerate(contract.resource_names):
        if (np.isfinite(required[i]) and np.isfinite(tolerated_required[i]) and np.isfinite(rounding[i])
                and tolerated_required[i] - rounding[i] > contract.capacity[i] + atol + rtol * abs(contract.capacity[i])):
            shortfalls.append({"resource": name, "minimum_required": float(required[i]),
                               "available": float(contract.capacity[i]),
                               "tolerated_minimum_required": float(tolerated_required[i]),
                               "shortfall": float(required[i] - contract.capacity[i])})
    return deep_freeze({
        "kind": "nonnegative_resource_lower_bound", "shortfalls": shortfalls,
        "infeasibility_established": bool(shortfalls),
        "evidence_level": "tolerance_qualified_arithmetic" if shortfalls else "not_established",
        "protected_constraints": ["minimum commitments", "maximum quantities", "resource capacities"],
        "interpretation": "A shortfall is incompatible with fixed minimum commitments. No constraint was relaxed.",
    })


@dataclass(frozen=True, slots=True)
class ProductionScenario:
    index: int
    name: str
    contract: ProductionContract | None
    execution: object = None
    formulation: FormulationCheck | None = None
    semantics: object = None
    diagnosis: object = None
    comparison: object = None
    error: str | None = None
    state: str = "completed"

    @property
    def accepted(self):
        return bool(self.execution is not None and self.execution.summary.feasible
                    and self.formulation.matches and self.semantics["feasible"]
                    and self.semantics["objective_consistent"]
                    and not self.diagnosis["infeasibility_established"])


@dataclass(frozen=True, slots=True)
class ProductionStudy:
    contract: ProductionContract
    scenarios: tuple[ProductionScenario, ...]
    study_id: str = field(default_factory=new_execution_id)
    input_complete: bool = True

    def to_dict(self, *, include_model=False):
        """Export decisions and labels; full contract/replay input is opt-in."""
        rows = []
        for case in self.scenarios:
            row = {"index": case.index, "name": case.name, "state": case.state,
                   "accepted": case.accepted, "error": case.error,
                   "contract_hash": None if case.contract is None else case.contract.data_hash,
                   "formulation": case.formulation, "semantics": case.semantics,
                   "diagnosis": case.diagnosis, "comparison": case.comparison}
            if case.execution is not None:
                row["evidence"] = evidence_bundle(case.execution.problem, case.execution.result,
                                                  include_model=include_model)
            if include_model and case.contract is not None:
                row["contract"] = case.contract.to_dict()
            rows.append(row)
        return _seal({
            "schema": "solverpilot.production-study.v1", "study_id": self.study_id,
            "baseline_contract_hash": self.contract.data_hash,
            "input_complete": self.input_complete,
            "objective_sense": "maximize", "objective_unit": self.contract.objective_unit,
            "export_policy": {"full_inputs": bool(include_model), "decisions_and_labels": True},
            "scope": "explicit continuous-production contract; no guarantee about unstated requirements",
            "scenarios": rows,
        })

    def save(self, path, *, include_model=False):
        Path(path).write_text(json.dumps(self.to_dict(include_model=include_model), indent=2,
                                        ensure_ascii=False, allow_nan=False), encoding="utf-8")
        return Path(path)

    def render_markdown(self):
        from solverpilot.reporting.render import _md
        lines = ["# SolverPilot production study", "", f"Study: `{self.study_id}`",
                 f"Objective: maximize profit ({_md(self.contract.objective_unit)})", "",
                 "Contract checks and numerical optimality evidence are distinct.", "",
                 "| Scenario | State | Accepted | Profit | Change vs baseline | Optimality evidence |",
                 "| --- | --- | --- | --- | --- | --- |"]
        for case in self.scenarios:
            summary = None if case.execution is None else case.execution.summary
            comparison = case.comparison or {}
            lines.append("| " + " | ".join(_md(v) for v in (
                case.name, case.state if summary is None else summary.status, case.accepted,
                comparison.get("objective") if comparison.get("objective") is not None else "unavailable",
                comparison.get("objective_delta") if comparison.get("objective_delta") is not None else "not comparable",
                "not established" if summary is None else summary.optimality,
            )) + " |")
        for case in self.scenarios:
            comparison = case.comparison or {}
            if case.error:
                lines.extend(["", f"{_md(case.name)}: {_md(case.error)}", ""])
            if comparison.get("decisions"):
                lines.extend(["", f"**{_md(case.name)} — quantities:** " + "; ".join(
                    f"{_md(d['product'])}={d['quantity']:g} (change {d['delta'] if d['delta'] is not None else 'unavailable'})"
                    for d in comparison["decisions"]), ""])
            if case.diagnosis:
                for item in case.diagnosis["shortfalls"]:
                    lines.extend(["", f"{_md(case.name)}: {_md(item['resource'])} requires at least "
                                  f"{item['minimum_required']:g}; available {item['available']:g}. "
                                  "Fixed minimum commitments cannot be met; no constraint was relaxed.", ""])
        lines.extend(["", "Objective changes are omitted when profit coefficients differ or either candidate is unaccepted.",
                      "Contract checks cover supplied assumptions only. Integrity hashes are not digital signatures."])
        return "\n".join(lines) + "\n"


def run_production_scenarios(contract, scenarios=(), *, backend=None, options=None,
                             cancellation=None, on_error="record"):
    """Run baseline plus finite independent scenarios using the existing session sweep.

    Inputs are validated before solving. Invalid entries remain error rows when
    requested. Omitted parameters reset to baseline; comparison never substitutes
    zero for errors/infeasibility or compares different profit definitions.
    """
    if not isinstance(contract, ProductionContract):
        raise TypeError("contract must be ProductionContract")
    if on_error not in {"record", "raise"}:
        raise ValueError("on_error must be record or raise")
    options = options or SolveOptions()
    atol, rtol = options.tolerances.feasibility, options.tolerances.feasibility_rel
    entries, errors, seen = [("baseline", {}, contract)], {}, {"baseline"}
    token = cancellation if cancellation is not None else options.cancellation
    source, input_complete = iter(scenarios), False
    while token is None or not token.cancelled:
        try:
            item = next(source)
        except StopIteration:
            input_complete = True
            break
        i = len(entries)
        name = f"scenario-{i}"
        try:
            name, updates = item if isinstance(item, tuple) else (name, item)
            if not isinstance(name, str) or not name.strip() or name in seen:
                raise ValueError("scenario names must be nonempty, unique and different from baseline")
            seen.add(name)
            effective = contract.updated(updates)
            entries.append((name, {k: getattr(effective, k) for k in updates}, effective))
        except (TypeError, ValueError) as exc:
            if on_error == "raise":
                raise
            # Keep a unique display name even for malformed or duplicate labels.
            error_name = f"invalid-{i}"
            while error_name in seen:
                error_name += "-error"
            seen.add(error_name)
            entries.append((error_name, {}, None))
            errors[i] = f"{type(exc).__name__}: {exc}"
    valid = [(i, entry) for i, entry in enumerate(entries) if entry[2] is not None]
    executions = {}
    stream = scenario_sweep(contract.build_model(), [(name, updates) for _, (name, updates, _) in valid],
                            backend=backend, options=options, cancellation=token, on_error=on_error)
    try:
        for execution in stream:
            executions[valid[execution.index][0]] = execution
    finally:
        stream.close()
    cases = []
    baseline = None
    for i, (name, _, effective) in enumerate(entries):
        execution = executions.get(i)
        if i in errors or (execution is not None and execution.error is not None):
            cases.append(ProductionScenario(i, name, effective, error=errors.get(i) or execution.error, state="error"))
            continue
        if execution is None:
            cases.append(ProductionScenario(i, name, effective, state="cancelled"))
            continue
        if execution.summary.problem_data_hash != execution.problem.data_hash:
            raise ValueError("scenario execution provenance does not match its problem")
        formulation = check_production_formulation(effective, execution.problem)
        semantics = check_production_candidate(effective, execution.result.x, execution.summary.objective,
                                               atol=atol, rtol=rtol)
        diagnosis = _capacity_diagnosis(effective, atol=atol, rtol=rtol)
        case = ProductionScenario(i, name, effective, execution, formulation, semantics, diagnosis)
        if i == 0:
            baseline = case
        comparable = bool(case.accepted and baseline is not None and baseline.accepted
                          and np.array_equal(effective.profit, contract.profit))
        decisions = []
        if case.accepted:
            for j, product in enumerate(contract.product_names):
                value = float(execution.result.x[j])
                delta = None if baseline is None or not baseline.accepted else value - float(baseline.execution.result.x[j])
                decisions.append({"product": product, "quantity": value, "delta": delta})
        comparison = deep_freeze({
            "objective": semantics["objective"] if case.accepted else None,
            "comparable_objective": comparable,
            "objective_delta": semantics["objective"] - baseline.semantics["objective"] if comparable else None,
            "decisions": decisions,
            "binding_resources": [contract.resource_names[j] for j, slack in enumerate(semantics["resource_slack"])
                                  if abs(slack) <= atol + rtol * abs(effective.capacity[j])] if case.accepted else [],
        })
        cases.append(replace(case, comparison=comparison))
    return ProductionStudy(contract, tuple(cases), input_complete=input_complete)


def replay_production_scenario(path, name="baseline", *, require_versions=True, require_source=True):
    """Replay a full-input study row, then re-check its original contract.

    This executes a new solve and returns (result, formulation_check, semantic_check).
    Loaded prior evidence alone never qualifies a new result.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "solverpilot.production-study.v1":
        raise ValueError("unsupported production study schema")
    _verify_integrity(payload)
    rows = [row for row in payload["scenarios"] if row["name"] == name]
    if len(rows) != 1 or "contract" not in rows[0] or "evidence" not in rows[0]:
        raise ValueError("replay requires one executed scenario with full input export")
    row = rows[0]
    contract = ProductionContract.from_dict(row["contract"])
    if contract.data_hash != row["contract_hash"]:
        raise ValueError("production contract hash mismatch")
    bundle = verify_evidence_bundle(row["evidence"])
    result = _replay_payload(bundle["run"], require_versions=require_versions, require_source=require_source)
    # Rebuild only for provenance; the independent audit below checks against raw
    # contract coefficients, so a mutated template cannot certify itself.
    problem = contract.build_model().compile().execution_ir
    if problem.data_hash != bundle["problem_data_hash"]:
        raise ValueError("replayed canonical model does not match the production contract")
    formulation = check_production_formulation(contract, problem)
    semantics = check_production_candidate(contract, result.x, result.objective,
        atol=row["semantics"]["atol"], rtol=row["semantics"]["rtol"])
    return result, formulation, semantics
