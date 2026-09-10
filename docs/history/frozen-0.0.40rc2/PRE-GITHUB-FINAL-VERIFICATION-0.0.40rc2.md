# Pre-GitHub Final Verification — SolverPilot 0.0.40rc2

**Classification:** TRACK P P0–P9 INTEGRATED PRE-RELEASE — PRIVATE GITHUB STAGING ELIGIBLE; PUBLIC RELEASE FAIL-CLOSED

## Integrated release scope

`0.0.40rc2` forward-ports the frozen Track P P0–P9 feature line onto the hardened SolverPilot rc8 baseline. It does **not** overwrite the M27–M33/pre-GitHub hardening line.

The canonical distribution and namespace are `solverpilot`. No live `optimind` compatibility package is shipped. The existing 78-symbol top-level public API remains frozen; Track P capabilities are exposed through explicit submodules such as `solverpilot.model`, `solverpilot.conic`, `solverpilot.nlp`, `solverpilot.minlp`, and `solverpilot.cp`.

## Authoritative integrated regression

The final rc2 hardening regression inventory is:

- **593 collected executable tests**;
- **591 passed**;
- **2 skipped** inside the collected suite;
- **0 failures**;
- **0 errors**;
- duplicate collected test IDs: **0**.

In addition, three optional native-integration modules are skipped at collection because `highspy`, Python `osqp`, and `pyscipopt` are not installed in this container. Those module-level skips cover six native test functions and are recorded separately rather than being silently omitted or counted as passes. The two collected-suite skips are the known external M22 evidence boundary and unavailable M25 prepared cache.

## Installed-wheel qualification

A diagnostic wheel built from the final rc2 hardening source was installed outside the source tree and exercised successfully for:

- LP / MILP / continuous QP core runtime;
- conservative production planner with learned performance routing disabled;
- semantic modeling and conic IR construction;
- CasADi 3.7.2 / Ipopt local NLP solve and independent validation;
- certified convex binary MINLP smoke;
- CP reference path;
- OR-Tools CP-SAT 9.15.6755 solve with independent parent-process revalidation;
- all **14/14** public examples.

The local wheel hardening smoke also verifies fail-closed unconfirmed/nonconvex QP routing, scale-aware primal validation, CP IR semantic rejection, MPS multi-vector rejection, NLP domain-hazard accounting, and the MINLP proof-scope split.

No `optimind` package is present in the installed wheel.

## CP-SAT native isolation boundary

Integration testing found a real Linux native-library collision: loading the SciPy/HiGHS path before OR-Tools 9.15 CP-SAT could make CP-SAT import fail with an undefined HiGHS C++ symbol. The integrated backend therefore executes CP-SAT in a clean subprocess and transfers only canonical JSON problem/result data across the boundary. The parent SolverPilot process independently validates the returned assignment/objective.

The critical `HiGHS -> CP-SAT` ordering regression passes on the integrated wheel.

## sdist round-trip

The diagnostic source distribution was extracted, rebuilt into a wheel, installed outside the extracted source tree, and re-exercised for Core, Conic, NLP, MINLP, and CP-SAT. All smoke gates passed. The rebuilt wheel was byte-identical to the direct diagnostic wheel.

## Package and repository audit

- wheel top-level package: `solverpilot` only;
- legacy `optimind` wheel members: **0**;
- wheel test/cache pollution: **0**;
- sdist contains all **14** public example scripts;
- sdist contains current API/backend snapshots and Track P merge provenance;
- sdist excludes tests, release tools, GitHub workflows, caches, and binary artifacts;
- secret-pattern scan: **0 hits**;
- active GitHub workflows: `ci.yml`, `release-qualification.yml`;
- publication workflow remains `publish.yml.disabled`;
- all referenced GitHub Actions are pinned to full 40-character commit SHAs;
- qualification includes Linux/macOS/Windows and Python 3.12/3.13/3.14 plus extended `conic`, `nlp`, `minlp`, and `cp` cells.

## Build qualification boundary

`pyproject.toml` requires the exact build backend:

- `setuptools==84.0.0`
- `wheel==0.48.0`

Those exact build requirements could not be materialized in this container. Local distributions were therefore built with no build isolation for **diagnostic verification only**. The exact-pinned backend remains mandatory in the GitHub release-qualification workflow; local diagnostic artifacts are not authorized for PyPI publication.

## Publication boundary

Local rc2 hardening artifact: **VERIFIED WITH EXTERNAL GATES PENDING**.

Private GitHub staging source: **ELIGIBLE**.

Public GitHub/PyPI/1.0: **NOT AUTHORIZED** until repository URLs/settings are real and the external exact-artifact GitHub qualification matrix succeeds.
