# Audit repairs and comparative evaluation

Development after published 0.4. This work repairs the audit findings at source
commit `736fafc`; it does not publish a new PyPI version or enable learned routing.

## Correctness repairs

* `OptimalityCheck.dual_bound` is expressed in the original objective: a lower
  bound for minimization and an upper bound for maximization, including the
  objective constant. Conversion uses exact represented coefficients before
  outward binary64 rounding. Internal gap/tolerance qualification is unchanged.
* Evidence loading checks agreement between recorded summary, validation,
  optimality flags, report claims and budget metadata, including strict boolean
  types. Existing valid evidence v1 records remain loadable. These checks do not
  authenticate a record or rerun a mathematical proof.
* Production cancellation works both as an explicit argument and through
  `SolveOptions`. The sweep consumes the token between scenarios; active native
  solves are not interrupted. Original options remain immutable.
* Numeric-conversion overflow follows the scenario input-error policy. A bad
  scenario no longer discards the valid baseline and subsequent cases when
  `on_error="record"` is used.

## Reproducible mutation challenge

```powershell
python benchmarks/challenge_production_evidence.py --output NEW-CHALLENGE.json
```

The runner creates independent LP representations and explicitly constructed
feasible points. It checks missing/reversed constraints, objective mistakes,
bound/domain mistakes, resource and quantity violations, nonfinite candidates,
tolerance boundaries and contradictory resealed evidence. Positive controls
prevent a validator that rejects everything from succeeding. Tests also disable
individual checks to confirm the challenge can detect those regressions.

Seed, protocol, runner and package-source hashes are recorded. This is a seeded
synthetic development challenge, not an independently held-out cohort. Unit
labels and a source mistake repeated consistently in both contract and model
are explicit blind-spot demonstrations, excluded from detector accuracy.

## Matched native-solver comparison

Use a separate environment with SolverPilot, highspy, CVXPY and Pyomo installed.
The reviewed adapter versions are highspy 1.15.1, CVXPY 1.9.3 and Pyomo 6.10.1.
The runner records installed versions and does not install or silently skip a
requested interface.

```powershell
python benchmarks/compare_lp_workflows.py --preset standard --repeats 8 --warmups 1 --steps 4 --max-wall-s 300 --output NEW-COMPARISON.json
```

It compares SolverPilot's matrix-first `execute`/HighspyNativeBackend path,
direct highspy, CVXPY+HiGHS and Pyomo/APPSI+HiGHS. All use the same highspy native
library, simplex, presolve off, one thread, and the recorded native defaults for
feasibility tolerances, simplex strategy and seed. Native extension bytes are
hashed. This isolates wrapper differences; shared-backend agreement is not an
independent numerical solver validation.

Sparse production and network families span three size settings and two row
scaling ranges, with constructed primal/dual reference witnesses. Objective and
right-hand-side updates genuinely change data. Cold means a fresh model after
imports. Repeated sequences retain CVXPY DPP parameters, Pyomo's persistent
interface and direct HiGHS models. SolverPilot's selected public highspy adapter
currently rebuilds its native model; this limitation is recorded rather than
described as native warm-start reuse. The benchmark does not evaluate every
SolverPilot session/backend combination. JuMP is not included in this runner.

Each workflow receives the same additional primal/objective/reference check.
SolverPilot's internal checks remain enabled, so its pre-check time still
includes validation. Construction/update and solve API times are separate, but
compilation can occur inside a solve call. Evidence export, application contract
checking and human effort are outside this numerical benchmark; it is not a
comparison of complete production-evidence workflows.

All outcomes, candidates, order positions and failures remain inspectable.
Warmups are excluded from timing summaries and reported separately. Cyclic and
reversed orders reduce position bias; eight repetitions complete both cycles
for four tools. Minimum, median and p95 are descriptive values, not significance
tests. A missing requested tool or a failed measured case makes the comparison
incomplete. The global limit is a cooperative stop, not a hard process deadline.

The planted optima retain their support/tight rows across updates. These cases
do not qualify active-set transitions, difficult degeneracy, MILP/global solving,
real industrial scalability or user productivity. Row-scale spread is not a
measured matrix condition number. Background machine activity is uncontrolled.
CI runs a small functional smoke matrix, not a hardware performance gate.

For competitor behavior, see the primary documentation for
[CVXPY DPP](https://www.cvxpy.org/tutorial/dpp/index.html),
[CVXPY solver options](https://www.cvxpy.org/tutorial/solvers/index.html), and
[Pyomo APPSI](https://pyomo.readthedocs.io/en/stable/api/pyomo.contrib.appsi.base.PersistentSolver.html).

## Independent participant study

The [comparison protocol](pilot/COMPARISON-PROTOCOL.md) specifies paired task
variants, alternating tool order, reference scoring, observed human durations,
failures and interventions. Its blank template is included in the portable kit.
Actual participants and their operating data/feedback are still required. The
technical runner never fabricates observations or sends participant messages.

## Recorded local observations

The [machine-readable summary](evidence/audit-remediation/summary.json) binds the
tested package source, candidate wheel, benchmark scripts and full local raw
outputs by SHA-256. Regression finished with 1,866 passed and 49 skipped tests;
line coverage was 85.75%, branch coverage 71.65% (81.91% combined). The configured
three-file mypy check and correctness-focused Ruff check passed. Native/hardware
skip requirements remain; these figures do not qualify every optional backend.

The mutation challenge accepted all 46 positive controls and rejected all 188
invalid cases, with no unexpected errors. Two clean wheel installations replayed
both pilot fixtures successfully. Independent review loaded 27 prior saved
evidence bundles unchanged.

The installed candidate wheel was compared on this Windows machine with HiGHS
1.15.1, CVXPY 1.9.3 and Pyomo 6.10.1. Twelve synthetic sequences, four states,
two execution modes and eight measured repetitions per tool produced 3,072
accepted measured solves. All 384 warmup solves also passed. The largest network
model had 1,535 variables and 512 rows. These are constructed optima, not an
industrial dataset.

| Workflow | Fresh-model median, ms | Repeated-update median, ms |
| --- | ---: | ---: |
| SolverPilot / native HiGHS | 6.828 | 6.208 |
| Direct highspy | 1.841 | 0.340 |
| CVXPY / HiGHS | 8.808 | 2.522 |
| Pyomo / APPSI HiGHS | 19.297 | 3.363 |

These are pooled medians of `checked_total_s`, with equal counts per case: 384
fresh-model and 288 update observations per tool. Sequence initialization is
retained separately in JSON. Imports/process startup are excluded; background
activity was uncontrolled. The full regression suite and installation work had
finished before this timing run. The common additional checker is identical,
but SolverPilot also performs its built-in validation/certificate work.

Within this limited workload, SolverPilot's fresh-model median is lower than
the two modeling interfaces, while its repeated-update median is higher. This
supports investigating native reuse and validation overhead separately. It does
not establish a general ranking, a learned-routing benefit, an independent
holdout result or a human productivity improvement.
