# Production contracts and execution evidence

This is unreleased development after published 0.4. It completes a scoped
continuous-production workflow using the existing semantic template, scenario
sweep, owned session, core solver, explanation and history APIs. It does not
enable learned routing, change solver proof categories or add a cloud service.

## Run a complete capacity study

```python
from solverpilot.applications.production import (
    ProductionContract, run_production_scenarios, replay_production_scenario,
)

contract = ProductionContract(
    profit=[3, 2], resources=[[1, 1], [2, 1]], capacity=[4, 5],
    minimum=[1, 0], maximum=[10, 10],
    product_names=("A", "B"), resource_names=("labor", "material"),
    objective_unit="USD/day",
)
study = run_production_scenarios(contract, [
    ("more labor", {"capacity": [5, 5]}),
    ("new prices", {"profit": [1, 4]}),
    ("shortage", {"capacity": [.5, 5]}),
    ("reset", {}),
], backend="scipy-highs-ds")
print(study.render_markdown())
study.save("production.json", include_model=True)
result, formulation, semantics = replay_production_scenario("production.json", "baseline")
assert formulation.matches and semantics["feasible"]
```

The baseline produces A=1, B=3 and profit 9. Minimum commitments are protected;
the shortage scenario cannot meet A>=1 with labor capacity 0.5. No constraint is
silently relaxed. A negative capacity is invalid input, not an infeasibility
result. Without positive minimum commitments, zero production is feasible for
every valid capacity input.

The contract owns immutable copies of dense numerical inputs. Product/resource
names must be unique and nonempty. Quantities are continuous, resource coefficients and
capacities nonnegative, and maximum quantities at least the minimum commitments.
Maximum may contain positive infinity. `ProductionContract.from_dict()` accepts
the versioned JSON contract returned by `to_dict()`. Unit labels are caller
declarations; the library does not infer dimensional consistency or omitted
business rules. This template excludes setup costs, integer batches, stock,
lead times, networks and stochastic recourse.

Scenarios are finite named mappings that change only `capacity` and/or `profit`.
They are validated before solving. Omitted parameters reset to the original
baseline. `on_error="record"` retains distinct input/solve error rows; `"raise"`
propagates errors. Names, including generated error names, remain unique.
Cancellation stops input collection and subsequent solves; `input_complete`
reports whether the finite source was exhausted. Already collected unexecuted
rows are cancelled, with no invented execution or zero profit.
The token can be supplied as `cancellation=token` or inside `SolveOptions`;
both provide cancellation between scenarios, not interruption of an active
native solve. A direct token takes precedence when both are supplied.
Numeric conversion overflow is an input validation error and follows the same
`on_error` policy as other malformed scenario values.

`SolveOptions` controls the existing solver. Its feasibility tolerances also
govern candidate checks and the capacity-shortfall diagnostic. Each scenario's
budget belongs to that solve, not the entire study; contract checks, collection,
reporting and export add work outside that per-solve budget.

## Three different checks

1. `check_production_formulation(contract, problem)` independently compares the
   canonical LP with raw contract coefficients, bounds, continuous domains,
   objective direction and offset, and every ordered resource row. It recognizes
   both `a @ x <= b` and `-a @ x >= -b`. It does not build the expected model
   through the same compiler. Other equivalent scaling/reordering is outside
   this narrow contract, so a mismatch is not a universal inequivalence proof.
2. `check_production_candidate` recomputes resource usage, quantity bounds and
   profit directly from the contract. Missing/nonfinite candidates and evaluation
   overflow are unavailable/failed, never feasible zero-production substitutes.
3. The existing numerical feasibility/optimality evidence remains separate.
   Passing a contract check does not establish optimality, exact arithmetic,
   completeness of the business specification or industrial suitability.

A scenario is accepted only when canonical candidate validation, formulation
agreement and contract candidate/objective checks pass without an established
capacity contradiction. The shortfall diagnostic uses nonnegative consumption
and tolerated minimum quantities, with a roundoff margin. It is a scoped
numerical arithmetic check, not a generic exact infeasibility certificate, IIS,
minimum repair, or authorization to change commitments.

Comparison includes named decisions/deltas, binding resource names, and profit
changes only when both candidates are accepted and the profit coefficients are
identical. Changed prices, infeasibility, missing results and failures have no
objective delta. No percentage divides by a zero baseline. Numerical optimality
levels are retained; an accepted feasible incumbent need not be optimal.

## One execution identity

`SolveTrace.execution_id` is created once and survives `dataclasses.replace`.
`SolveResult.execution_id`, `summarize(result).run_id`, explanation
`runtime.execution_id`, default history IDs and run/evidence exports agree.
Two solves of identical model data have different execution IDs. Model hashes
do not depend on execution IDs. Common conic/NLP/MINLP/global/CP/exact/batch result
objects own IDs too; batch transports retain the completed worker identity.
Third-party results without identity summarize with an empty ID, not a fabricated
execution. Application-specific TSP/VRP and aggregate orchestration adapters are
outside this new evidence-envelope scope.

Explicit history `run_id=` overrides remain supported as storage aliases; record
metadata retains the actual `execution_id`. Trace fields are additive optional
keyword-only constructor parameters; frozen historical API snapshots are intact.

## Export and replay policy

```python
from solverpilot.runtime.evidence import save_evidence_bundle, replay_evidence_bundle

case = study.scenarios[0]
save_evidence_bundle("evidence.json", case.execution.problem, case.execution.result)
# Explicit full-model permission is necessary for executable replay:
save_evidence_bundle("replay.json", case.execution.problem, case.execution.result,
                     include_model=True)
replayed = replay_evidence_bundle("replay.json")
assert replayed.execution_id != case.execution.result.execution_id
assert replayed.trace.replay_of == case.execution.result.execution_id
```

`solverpilot.run.v2` is the default `run_manifest`/`save_run` format. It includes
the execution ID, summary, package versions, code identity, export policy and a
content integrity hash. By default it excludes model arrays, parameters and raw
backend statistics. `include_model=True` includes model and effective parameters;
`include_raw_statistics=True` separately opts into unfiltered backend data.
These replace v1's implicit raw-statistics export. Consumers depending on raw
statistics must explicitly opt in.

`solverpilot.evidence.v1` links the run, summary and structured explanation.
Its explanation allowlist excludes free-text warning/rationale fields, including
nested validation warnings. Invalid results whose explanation cannot be safely
represented retain an explicit unavailable explanation with no added claims.
Default exports still disclose objective, metrics, identities and backend names;
they are not anonymization. A production-study export additionally includes
decision values, labels, comparisons, diagnostics and input-error messages.
Review these before sharing. Full production contracts and replay models require
`include_model=True`; nothing is uploaded automatically.

Core evidence envelopes and replay currently cover `SolveResult` and canonical
LP/MILP/convex QP. They do not claim universal specialized-result replay.
Production replay checks full-input contract/hash linkage, performs a new solve,
and reruns contract checks. Replaying never reuses the original execution ID.

Code identity hashes package Python source bytes and is cached at the first trace
in a process; restart after editing source. Readiness uses the same hashing
algorithm. This records a source snapshot, not binary/loaded-code attestation or
native solver executable identity. Unreadable source is marked unavailable.
V2 replay checks versions and source by default; `require_source=False` explicitly
permits source drift. Legacy v1 remains readable with version-only checks when
the argument is omitted; explicit `require_source=True` rejects v1 because it
has no source evidence. V1 cannot acquire missing evidence retroactively.

Integrity hashes detect changed/corrupted content and linked-field disagreement.
They are not signatures and cannot authenticate a bundle whose author can replace
both content and hash. Loading a bundle checks integrity/consistency, not the
truth of stored mathematical claims. Only a new checked solve establishes new
execution evidence. Use trusted local JSON; never pickle or executable payloads.

## Validation and next qualification

The [pilot kit](pilot/README.md) adds an independently formulated direct-SciPy reference,
public educational and explicitly synthetic inputs, balanced repeated machine timings,
two clean wheel installations and blank participant feedback forms. It does not substitute
automated execution for independent users or confirmed business requirements.

`tests/test_execution_evidence.py` covers identity, history aliases, export
sentinels, lineage, source/version checks, legacy reading, batch transport and
invalid-candidate export. `tests/test_production_workflow.py` uses analytic
optima, deliberately removed/reversed rows, wrong objective sense, malformed
scenarios, minimum-commitment conflicts, tolerance consistency and replay.
`examples/28_production_evidence_workflow.py` runs the complete workflow using
only the base SciPy backend and temporary local files.

These automated checks establish the specified software behavior. Time saved,
user comprehension, repeat adoption and market value still require independent
users with their own data; they are not claimed by passing tests.
