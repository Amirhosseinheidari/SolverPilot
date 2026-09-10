# M8 Status — Health-Gated Planning + Validated Portfolio Execution

**Version:** 0.0.9  
**Milestone state:** completed for M8's local-runtime scope. Public native solver packages and large public benchmark suites remain blocked/unexecuted in this runtime.

## Completed and executed

- explicit planner health policies: `ignore`, `prefer_healthy`, `require_healthy`;
- health reports can now gate or abstain planner selection without enabling any empirical performance rule;
- health evidence is version-specific; stale backend-version reports are ignored and cannot authorize execution;
- `CandidatePlan` records backend health status and `SolvePlan` records the health policy;
- trace schema bumped to **0.3** and records `planner_health_policy`;
- sequential validated portfolio executor with per-attempt isolation;
- portfolio ignores invalid candidate solutions and backend exceptions;
- portfolio respects minimization/maximization when choosing the best validated feasible result;
- portfolio does not silently hide conflicting terminal claims (`infeasible` vs `unbounded`);
- portfolio can require active healthy probes before execution;
- global wall-time/thread budgets are propagated only when a backend can enforce them;
- backward-compatible `_apply_budget` alias retained while public internal helper moved to `optimind.runtime.apply_budget`;
- fault-injection resilience benchmark executed over 100 randomized LPs;
- M1/M2/M5/M6/M7 correctness and performance regressions rerun on final M8 source;
- package version bumped to 0.0.9.

## Fault-injection resilience evidence

The benchmark deliberately registers a backend that advertises LP capability but returns an invalid all-zero solution. This tests runtime trust boundaries, not solver performance.

- cases: **100**;
- active probe classification: poisoned backend = **unhealthy**, SciPy HiGHS DS = **healthy**;
- capability-only planner selected an invalid result in **100/100** cases because registration order is its documented tie-break;
- health-gated planner returned a validated result in **100/100** cases;
- portfolio recovered after an injected backend exception in **100/100** cases;
- portfolio selected an invalid solution in **0** cases.

Interpretation: active health evidence prevents an available-but-broken backend from being trusted. This is a reliability result, not an empirical speed-selection claim.

## Latest selector research regression

- solver calls: **240**;
- median winners: DS **12**, IPM **12**;
- policy/SBS aggregate cost ratio: **0.923483**;
- bootstrap 95% interval: **[0.876471, 0.971705]**;
- SBS→VBS gap closure: **0.865801**;
- maximum DS/IPM objective difference: **0.0**.

This remains synthetic within-HiGHS evidence. The production planner does **not** enable this rule.

## Latest repeated-LP reuse regression

Median stateful/cold end-to-end wall ratios:

- small: **0.671828**;
- medium: **0.424940**;
- large: **0.344548**.

All paired objectives agree and all expected reuse steps were backend-confirmed in the recorded runs.

## Public native package attempt

Exact locked `files.pythonhosted.org` URLs were attempted for `highspy`, `osqp`, and `pyscipopt`. All three failed before download with curl return code 6 (`Could not resolve host: files.pythonhosted.org`). No wheel was accepted, installed, or counted as verified.

This is an environment DNS/egress failure, not evidence that the adapters pass public-package integration.

## Not completed

- public `highspy 1.15.1` execution in this runtime;
- public `OSQP 1.1.3` execution/update/warm-start integration;
- public `PySCIPOpt 6.2.1` execution/IIS integration;
- cross-solver SBS/VBS on a public corpus;
- full Netlib, MIPLIB 2017, or QPLIB compatible-subset benchmark;
- learned selector;
- final public package name/legal clearance.
