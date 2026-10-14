import json
from dataclasses import replace

import numpy as np
import pytest

from solverpilot import LinearProblem, solve
from solverpilot.applications.production import (
    ProductionContract, check_production_candidate, check_production_formulation,
    run_production_scenarios, replay_production_scenario,
)
from solverpilot.runtime.batch import CancellationToken
from solverpilot.scenarios import scenario_sweep


def contract(**kwargs):
    return ProductionContract([3., 2.], [[1., 1.], [2., 1.]], [4., 5.],
        maximum=[10., 10.], product_names=("A", "B"), resource_names=("labor", "material"),
        objective_unit="USD/day", **kwargs)


def test_analytic_scenarios_reset_inputs_and_do_not_compare_changed_objectives():
    study = run_production_scenarios(contract(), [
        ("more labor", {"capacity": [5., 5.]}),
        ("new prices", {"profit": [1., 4.]}), ("reset", {}),
    ], backend="scipy-highs-ds")
    assert [c.execution.summary.objective for c in study.scenarios] == pytest.approx([9., 10., 16., 9.])
    assert all(c.accepted for c in study.scenarios)
    assert study.scenarios[0].comparison["decisions"][0]["quantity"] == pytest.approx(1.)
    assert study.scenarios[1].comparison["objective_delta"] == pytest.approx(1.)
    assert study.scenarios[2].comparison["objective_delta"] is None
    assert not study.scenarios[2].comparison["comparable_objective"]
    assert study.scenarios[3].comparison["objective_delta"] == pytest.approx(0.)
    assert len({c.execution.summary.run_id for c in study.scenarios}) == 4
    assert "material" in study.scenarios[0].comparison["binding_resources"]


def test_minimum_commitment_shortfall_is_distinct_from_invalid_input():
    study = run_production_scenarios(contract(minimum=[1., 0.]), [
        ("shortage", {"capacity": [.5, 5.]}),
        ("negative", {"capacity": [-1., 5.]}), ("reset", {}),
    ], backend="scipy-highs-ds")
    shortage, invalid, reset = study.scenarios[1:]
    assert shortage.execution.summary.status == "infeasible"
    assert not shortage.accepted and shortage.comparison["objective"] is None
    assert shortage.comparison["objective_delta"] is None
    assert shortage.diagnosis["infeasibility_established"]
    assert shortage.diagnosis["shortfalls"][0]["resource"] == "labor"
    assert shortage.diagnosis["shortfalls"][0]["shortfall"] == pytest.approx(.5)
    assert invalid.state == "error" and invalid.execution is None
    assert reset.accepted and reset.comparison["objective"] == pytest.approx(9.)
    assert study.contract.minimum.tolist() == [1., 0.]


@pytest.mark.parametrize("row", [0, 1])
@pytest.mark.parametrize("mutation", ["omit", "reverse"])
def test_missing_or_reversed_capacity_cannot_pass_contract(row, mutation):
    spec = contract()
    p = spec.build_model().compile().execution_ir
    data = {"A": p.A.toarray(), "c": p.c, "variable_lower": p.variable_lower,
            "variable_upper": p.variable_upper, "constraint_lower": p.constraint_lower.copy(),
            "constraint_upper": p.constraint_upper.copy(), "objective_sense": p.objective_sense}
    if mutation == "omit":
        data["A"] = np.delete(data["A"], row, axis=0)
        data["constraint_lower"] = np.delete(data["constraint_lower"], row)
        data["constraint_upper"] = np.delete(data["constraint_upper"], row)
    else:
        data["constraint_upper"][row] = data["constraint_lower"][row]
        data["constraint_lower"][row] = -np.inf
    corrupt = LinearProblem.from_data(**data)
    result = solve(corrupt, backend="scipy-highs-ds")
    assert result.validation.valid  # Solving a wrong submitted model is not a solver defect.
    assert not check_production_formulation(spec, corrupt).matches
    checked = check_production_candidate(spec, result.x, result.objective)
    assert not checked["feasible"]


def test_wrong_objective_sense_requires_formulation_audit():
    spec = contract()
    p = spec.build_model().compile().execution_ir
    wrong = replace(p, objective_sense="minimize")
    result = solve(wrong, backend="scipy-highs-ds")
    assert result.objective == 0 and result.validation.valid
    assert check_production_candidate(spec, result.x, result.objective)["feasible"]
    assert not check_production_formulation(spec, wrong).matches


def test_equivalent_positive_row_orientation_is_accepted():
    spec = contract()
    p = LinearProblem.from_data(A=spec.resources, c=spec.profit,
        variable_lower=spec.minimum, variable_upper=spec.maximum,
        constraint_lower=[-np.inf, -np.inf], constraint_upper=spec.capacity, objective_sense="maximize")
    assert check_production_formulation(spec, p).matches


@pytest.mark.parametrize("quantities", [[np.nan, 0], [np.inf, 0], [1], [3, 1], [-1, 3], [11, 0]])
def test_invalid_candidates_do_not_get_accepted(quantities):
    assert not check_production_candidate(contract(), quantities, 9.)["feasible"]


def test_objective_sign_tampering_is_distinct_from_feasibility():
    check = check_production_candidate(contract(), [1, 3], -9.)
    assert check["feasible"] and not check["objective_consistent"]


def test_contract_owns_inputs_and_roundtrips():
    profits = np.array([3., 2.])
    resources = np.array([[1., 1.], [2., 1.]])
    spec = ProductionContract(profits, resources, [4, 5])
    original = spec.data_hash
    profits[:] = -100
    resources[:] = 0
    assert spec.data_hash == original
    with pytest.raises(ValueError):
        spec.resources.flags.writeable = True
    assert ProductionContract.from_dict(spec.to_dict()).data_hash == spec.data_hash


@pytest.mark.parametrize("updates", [{"typo": 1}, {"profit": [1]}, {"capacity": [np.nan, 5]}, None])
def test_invalid_scenario_is_retained_and_next_scenario_works(updates):
    spec = contract()
    study = run_production_scenarios(spec, [("bad", updates), ("after", {})], backend="scipy-highs-ds")
    assert study.scenarios[1].state == "error"
    assert study.scenarios[2].comparison["objective"] == pytest.approx(9.)
    with pytest.raises((ValueError, TypeError)):
        run_production_scenarios(spec, [("bad", updates)], on_error="raise")


def test_cancellation_has_no_invented_execution_or_zero_profit():
    token = CancellationToken()
    token.cancel()
    study = run_production_scenarios(contract(), [("next", {})], cancellation=token)
    assert all(c.state == "cancelled" and c.execution is None and not c.accepted for c in study.scenarios)


def test_full_study_export_replay_and_default_minimization(tmp_path):
    study = run_production_scenarios(contract(), [("changed", {"capacity": [5, 5]})], backend="scipy-highs-ds")
    path = tmp_path / "study.json"
    study.save(path)
    payload = json.loads(path.read_text())
    assert "contract" not in payload["scenarios"][0]
    assert "model" not in payload["scenarios"][0]["evidence"]["run"]
    with pytest.raises(ValueError, match="full input"):
        replay_production_scenario(path)
    study.save(path, include_model=True)
    result, formulation, semantics = replay_production_scenario(path, "changed")
    assert result.objective == pytest.approx(10.)
    assert formulation.matches and semantics["feasible"] and semantics["objective_consistent"]
    assert result.trace.replay_of == study.scenarios[1].execution.summary.run_id
    payload = json.loads(path.read_text())
    payload["scenarios"][1]["contract"]["capacity"][0] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash"):
        replay_production_scenario(path, "changed")
    markdown = study.render_markdown()
    assert "9.0" in markdown and "A=" in markdown and "numerical optimality" in markdown


def test_generic_sweep_records_malformed_item_and_keeps_problem_snapshot():
    rows = list(scenario_sweep(contract().build_model(), [("bad", None), ("ok", {})],
        on_error="record", backend="scipy-highs-ds"))
    assert rows[0].error is not None
    assert rows[1].problem.data_hash == rows[1].summary.problem_data_hash


def test_duplicate_names_are_errors_not_ambiguous_replay():
    study = run_production_scenarios(contract(), [("baseline", {}), ("x", {}), ("x", {})], backend="scipy-highs-ds")
    assert [c.state for c in study.scenarios] == ["completed", "error", "completed", "error"]
    assert len({c.name for c in study.scenarios}) == len(study.scenarios)


def test_generated_error_name_cannot_shadow_a_replayable_scenario(tmp_path):
    study = run_production_scenarios(contract(), [("invalid-2", {}), ("bad", None)], backend="scipy-highs-ds")
    assert len({c.name for c in study.scenarios}) == 3
    path = study.save(tmp_path / "study.json", include_model=True)
    result, _, _ = replay_production_scenario(path, "invalid-2")
    assert result.objective == pytest.approx(9.)


def test_diagnosis_respects_tolerated_minimum_quantity():
    from solverpilot.applications.production import _capacity_diagnosis
    spec = ProductionContract([1], [[1e9]], [999999950], minimum=[1], maximum=[2])
    assert check_production_candidate(spec, [.99999995], .99999995)["feasible"]
    assert not _capacity_diagnosis(spec, atol=1e-7, rtol=1e-9)["infeasibility_established"]


def test_precancelled_study_does_not_consume_scenario_source():
    token = CancellationToken()
    token.cancel()
    def source():
        raise AssertionError("cancelled workflow consumed input")
        yield {}
    study = run_production_scenarios(contract(), source(), cancellation=token)
    assert not study.input_complete
    assert study.scenarios[0].state == "cancelled"


def test_options_cancellation_matches_direct_token_and_preserves_options():
    from solverpilot.runtime.options import SolveOptions
    from solverpilot.validate import ValidationTolerances
    token = CancellationToken()
    options = SolveOptions(cancellation=token, tolerances=ValidationTolerances(feasibility_rel=0))
    direct = run_production_scenarios(contract(), [("reset", {})], cancellation=token,
                                      backend="scipy-highs-ds")
    through_options = run_production_scenarios(contract(), [("reset", {})], options=options,
                                               backend="scipy-highs-ds")
    assert all(row.accepted for row in through_options.scenarios)
    assert [row.execution.summary.objective for row in through_options.scenarios] == [
        row.execution.summary.objective for row in direct.scenarios]
    assert options.cancellation is token and options.tolerances.feasibility_rel == 0
    assert through_options.scenarios[0].semantics["rtol"] == 0


def test_options_precancellation_does_not_consume_inputs():
    from solverpilot.runtime.options import SolveOptions
    token = CancellationToken()
    token.cancel()
    def source():
        raise AssertionError("cancelled workflow consumed input")
        yield {}
    study = run_production_scenarios(contract(), source(), options=SolveOptions(cancellation=token))
    assert not study.input_complete
    assert study.scenarios[0].state == "cancelled"


def test_options_cancellation_between_solves_keeps_completed_evidence(monkeypatch):
    import solverpilot.applications.production as production
    from solverpilot.runtime.options import SolveOptions
    token = CancellationToken()
    original = production.scenario_sweep
    def cancel_after_first(*args, **kwargs):
        for row in original(*args, **kwargs):
            yield row
            token.cancel()
    monkeypatch.setattr(production, "scenario_sweep", cancel_after_first)
    study = run_production_scenarios(contract(), [("next", {}), ("last", {})],
        options=SolveOptions(cancellation=token), backend="scipy-highs-ds")
    assert study.scenarios[0].accepted
    assert [row.state for row in study.scenarios] == ["completed", "cancelled", "cancelled"]
    assert all(row.execution is None for row in study.scenarios[1:])


@pytest.mark.parametrize("field", ["capacity", "profit"])
def test_conversion_overflow_records_only_bad_scenario(field):
    updates = {field: [10**1000, 1]}
    study = run_production_scenarios(contract(), [("bad", updates), ("reset", {})],
        backend="scipy-highs-ds", on_error="record")
    assert [row.state for row in study.scenarios] == ["completed", "error", "completed"]
    assert study.scenarios[0].accepted and study.scenarios[2].accepted
    assert "float64 range" in study.scenarios[1].error
    with pytest.raises(ValueError, match="float64 range"):
        run_production_scenarios(contract(), [("bad", updates)], on_error="raise")


def test_initial_contract_conversion_overflow_is_validation_error():
    with pytest.raises(ValueError, match="capacity.*float64 range"):
        ProductionContract([1], [[1]], [10**1000])


@pytest.mark.parametrize("kwargs", [
    {"profit": []}, {"resources": [[-1, 1], [2, 1]]}, {"maximum": [0, 10], "minimum": [1, 0]},
    {"product_names": ["same", "same"]}, {"objective_unit": ""},
])
def test_contract_rejects_invalid_specifications(kwargs):
    base = dict(profit=[3, 2], resources=[[1, 1], [2, 1]], capacity=[4, 5])
    with pytest.raises(ValueError):
        ProductionContract(**(base | kwargs))
