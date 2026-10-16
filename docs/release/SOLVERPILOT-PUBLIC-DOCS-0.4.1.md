# SolverPilot 0.4.1

This release includes the accumulated changes on main since published 0.4,
including PR #27's LP verification repairs. It preserves the 78 top-level names;
new APIs remain in explicit submodules or optional keyword arguments. The full
current signatures are recorded in `PUBLIC-API-V0_4_1.json`. The original 0.4
snapshots and historical benchmark files remain unchanged.

## Install

```bash
python -m pip install "solverpilot==0.4.1"
python -m pip install "solverpilot[highs]==0.4.1"
python -m solverpilot.cli.doctor --compact
```

Python 3.12-3.14 is supported. Base dependencies remain NumPy `>=2.2,<3` and
SciPy `>=1.15,<2`. Native integrations are optional and platform dependent;
installing an extra does not establish independent mathematical proof or hardware
qualification. `pip check` detects inconsistent installed dependencies.

## LP certificate and repeated-solve repairs

Exact dyadic arithmetic and invocation-local caches reduce repeated Fraction
construction during implied-bound propagation. The LP lower bound keeps variable
bounds in the domain rather than also adding their rounded multipliers to the
Lagrangian. Full dual sign, stationarity, complementarity and primal checks remain.
If simpler bounds fail, equality-only recovery is capped at 256 pivots, 50,000
nonzero visits and 8,192 rational bits. Shared proof work/time policies remain
fail-closed. Numerical tolerances are unchanged and no small residual is rounded
to zero. QP certificate formulas are unchanged by PR #27.

The pre-release development comparison recovered all 68 incomplete observations
from the frozen 256-observation public corpus. These are eight related public
families and repeated/transformed instances, not 256 independent datasets. The
[complete report](../LP-CERTIFICATE-PERFORMANCE.md) retains timing regressions,
source hashes, proof scope and OS variability. Its measured 0.4 development wheels
must not be confused with the newly built 0.4.1 distribution. Performance is not
guaranteed across hosts or problem families.

Opt-in fixed-matrix continuous-simplex HiGHS reuse retains native models when
eligible and rebuilds safely otherwise. Trace phases separate primal validation
and certificate work. Native model or basis observations do not establish
factorization reuse. Verification remains substantial when native solves are cheap.

## Other accumulated changes included

- Semantic NumPy/CasADi shape and ownership repairs; immutable result/evidence
  snapshots and original-coordinate termination witness checks.
- Optional proof policies and bounded coefficient-preparation reuse, aggregate
  scenario/lexicographic/MINLP budgets and isolated CPU deadline controls.
- Continuous convex QCQP-to-RSOC lowering using bounded exact decomposition,
  explicit PWL epigraph/graph helpers and independently checked inventory examples.
- Scoped convex-QP sensitivity, sparse PSD evidence, vector convex OA, local NLP
  maximization, domain enclosures and risk/uncertainty helpers.
- Production evidence identity, explicit export/replay, a public educational pilot
  kit and recorded independent comparisons. Real-user productivity is unqualified.
- Explicit exact SCIP/VIPR and cuOpt adapters, subject to separate executable and
  hardware qualification. External executables and CUDA are not bundled.
- Experimental learned LP training/calibration/replay tools. Automatic learned
  production routing remains disabled; failed promotion gates remain failures.

See [semantic contracts](../SEMANTIC-CONTRACTS.md), [convex modeling](../CONVEX-MODELING-NEXT.md),
[production evidence](../PRODUCTION-EVIDENCE-WORKFLOW.md), [exact solving](../EXACT-MILP.md),
[GPU scope](../GPU-LP.md) and the complete [changelog](../../CHANGELOG.md).

## Guarantee boundaries

Continuous LP/QP validation remains tolerance-qualified numerical evidence unless
an explicit exact certificate path says otherwise. Generic exact primal feasibility,
generic global NLP/MINLP proof, independent transcendental-cone certificates,
arbitrary infeasible/unbounded termination certificates and automatic factorization
reuse are not promised. Restricted convex binary MINLP retains its declared proof
scope. Historical `included_in_published_0_4` readiness fields describe 0.4, not
whether an integration is present or qualified in this release.

## Release identity and verification

Publication uses one wheel/sdist pair from an attested successful qualification
of the final main commit. The workflow checks 105 jobs, including nine core
OS/Python cells, minimum dependencies, 90 optional integration cells, security,
artifact provenance and the technical gate. TestPyPI and PyPI receive the same
qualified bytes. Registry file hashes and fresh wheel/sdist installation checks
are recorded in the matching GitHub release. Package version alone does not
identify old development artifacts or prove installed-source ownership.
