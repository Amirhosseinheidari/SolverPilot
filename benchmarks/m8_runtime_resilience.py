from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from solverpilot import HealthPolicy, LinearProblem, execute_portfolio, solve
from solverpilot.backends import BackendRegistry, BackendSolveResult, ScipyHighsLPBackend, probe_backends
from solverpilot.capabilities import BackendManifest, Capability, SupportLevel


@dataclass(slots=True)
class PoisonedBackend:
    name: str = "poisoned-lp"
    time_limit_s: float | None = None

    @property
    def manifest(self):
        return BackendManifest(
            name=self.name,
            version="fault-injection",
            capabilities={Capability.LP: SupportLevel.NATIVE},
            metadata={"fault_injection": True},
        )

    def is_available(self):
        return True

    def solve(self, problem):
        x = np.zeros(problem.n_variables, dtype=float)
        return BackendSolveResult("optimal", x, float(problem.c @ x + problem.objective_offset), {})


@dataclass(slots=True)
class RaisingBackend:
    name: str = "raising-lp"
    time_limit_s: float | None = None

    @property
    def manifest(self):
        return BackendManifest(
            name=self.name,
            version="fault-injection",
            capabilities={Capability.LP: SupportLevel.NATIVE},
            metadata={"fault_injection": True},
        )

    def is_available(self):
        return True

    def solve(self, problem):
        raise RuntimeError("injected backend failure")


def make_problem(seed: int) -> LinearProblem:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 12))
    # x >= lower > 0 makes the poisoned all-zero candidate definitely invalid.
    lower = rng.uniform(0.25, 1.25, size=n)
    upper = lower + rng.uniform(0.5, 2.0, size=n)
    c = rng.uniform(-2.0, 2.0, size=n)
    # Add a harmless aggregate constraint that preserves box feasibility.
    A = np.ones((1, n), dtype=float)
    rhs_lo = float(lower.sum())
    rhs_hi = float(upper.sum())
    return LinearProblem.from_data(
        A=A,
        c=c,
        variable_lower=lower,
        variable_upper=upper,
        constraint_lower=[rhs_lo],
        constraint_upper=[rhs_hi],
        name=f"m8_resilience_{seed}",
    )


def main() -> None:
    out = Path(__file__).resolve().parent / "results" / "m8-runtime-resilience.json"
    real = ScipyHighsLPBackend("highs-ds")
    poisoned = PoisonedBackend()
    raising = RaisingBackend()

    health_reports = probe_backends([poisoned, real])
    health_by_name = {r.backend: r.status.value for r in health_reports}

    capability_invalid = 0
    health_valid = 0
    portfolio_recovered = 0
    portfolio_invalid_selected = 0
    cases = []
    t0 = perf_counter()

    for seed in range(100):
        problem = make_problem(seed)

        registry = BackendRegistry()
        registry.register(PoisonedBackend())
        registry.register(ScipyHighsLPBackend("highs-ds"))

        baseline = solve(problem, registry=registry)
        if baseline.status.value == "invalid_solution":
            capability_invalid += 1

        guarded = solve(
            problem,
            registry=registry,
            health_reports=health_reports,
            health_policy=HealthPolicy.REQUIRE_HEALTHY,
        )
        if guarded.validation is not None and guarded.validation.valid:
            health_valid += 1

        portfolio = execute_portfolio(problem, [raising, real])
        if portfolio.chosen is not None and portfolio.chosen.validation is not None and portfolio.chosen.validation.valid:
            portfolio_recovered += 1
        if portfolio.chosen is not None and portfolio.chosen.status.value == "invalid_solution":
            portfolio_invalid_selected += 1

        if seed < 5:
            cases.append({
                "seed": seed,
                "capability_only_status": baseline.status.value,
                "health_aware_status": guarded.status.value,
                "health_aware_backend": guarded.trace.backend,
                "portfolio_backend": portfolio.chosen_backend,
                "portfolio_attempts": [asdict(a) for a in portfolio.attempts],
            })

    payload = {
        "schema_version": "1.0",
        "cases": 100,
        "health_reports": health_by_name,
        "capability_only_invalid": capability_invalid,
        "health_aware_valid": health_valid,
        "portfolio_recovered_after_exception": portfolio_recovered,
        "portfolio_invalid_selected": portfolio_invalid_selected,
        "sample_cases": cases,
        "elapsed_s": perf_counter() - t0,
        "interpretation": (
            "Fault-injection resilience test. It validates health gating and failure isolation; "
            "it is not a solver-performance benchmark."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
