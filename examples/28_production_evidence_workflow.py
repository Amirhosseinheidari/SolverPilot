"""Capacity planning, contract checks, reports and replay; no optional solvers needed."""

from pathlib import Path
from tempfile import TemporaryDirectory

from solverpilot.applications.production import (
    ProductionContract, run_production_scenarios, replay_production_scenario,
)
from solverpilot.history import HistoryStore


def main():
    contract = ProductionContract(
        profit=[3., 2.], resources=[[1., 1.], [2., 1.]], capacity=[4., 5.],
        minimum=[1., 0.], maximum=[10., 10.], product_names=("A", "B"),
        resource_names=("labor", "material"), objective_unit="USD/day",
    )
    study = run_production_scenarios(contract, [
        ("more labor", {"capacity": [5., 5.]}),
        ("new prices", {"profit": [1., 4.]}),
        ("shortage", {"capacity": [.5, 5.]}),
        ("reset", {}),
    ], backend="scipy-highs-ds")
    assert study.scenarios[0].accepted
    assert abs(study.scenarios[0].comparison["objective"] - 9.) < 1e-7
    assert study.scenarios[2].comparison["objective_delta"] is None
    assert study.scenarios[3].diagnosis["infeasibility_established"]
    assert abs(study.scenarios[4].comparison["objective"] - 9.) < 1e-7
    with TemporaryDirectory() as folder:
        path = study.save(Path(folder) / "production.json", include_model=True)
        Path(folder, "production.md").write_text(study.render_markdown(), encoding="utf-8")
        with HistoryStore(Path(folder) / "history.sqlite") as history:
            for case in study.scenarios:
                if case.execution is not None:
                    history.record_problem_object(case.execution.problem)
                    record = history.record_solve_result(case.execution.result)
                    assert record.run_id == case.execution.summary.run_id
        result, formulation, semantics = replay_production_scenario(path)
        assert formulation.matches and semantics["feasible"]
        assert result.trace.replay_of == study.scenarios[0].execution.summary.run_id
    print(study.render_markdown())


if __name__ == "__main__":
    main()
