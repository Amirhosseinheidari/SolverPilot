# Opt-in HiGHS reuse and measured validation cost

> Release inclusion: the implementation described here is included in 0.4.1.
> The original development measurements and artifact versions below are historical
> evidence, not measurements of the new 0.4.1 distribution.

Development after published 0.4. This work extends the [September audit](AUDIT-REMEDIATION.md).
Historical measurements remain unchanged and learned routing remains experimental.

The final October 13 qualification confirms the scoped reuse behavior and
preserved numerical checks. It does **not** establish a general speedup: the
stateless path regressed in the final comparison, and certificate work dominates
the public cases. Counts, phase statistics, per-case comparisons and raw-file
hashes are in the [machine-readable evidence](evidence/repeated-lp/summary.json).

## Retain a continuous simplex LP

This example requires the development candidate and the optional `highspy`
dependency. It is not available by installing the unchanged published 0.4 wheel.
The review bundle records the exact wheel and dependency pins used below.

```python
from dataclasses import replace
from solverpilot import LinearProblem, execute
from solverpilot.backends import HighspyNativeBackend

problem = LinearProblem.from_data(
    A=[[1., 1.]], c=[1., 2.], variable_lower=[0., 0.],
    variable_upper=[10., 10.], constraint_lower=[1.], constraint_upper=[float("inf")],
)
with HighspyNativeBackend(reuse=True, solver="simplex", threads=1) as backend:
    first = execute(problem, backend)
    second = execute(replace(problem, c=[2., 1.], constraint_lower=[2.]), backend)
    print(second.trace.reuse_applied)
    print(second.trace.timings.primal_validate_s)
    print(second.trace.timings.certificate_s)
    print(second.optimality_evidence.independently_verified_optimal)
```

Reuse requires an identical numerical matrix, sparsity, variable domains and
objective sense. Costs, row bounds, variable bounds and objective offset are
updated through the public highspy interface. Changes to matrix data, structure,
presolve, threads or solver rebuild the native model. A conflicting process-global
HiGHS thread scheduler is detected through the native diagnostic callback and
executes the requested configuration in a fresh, one-shot process. This path
retains no workspace and reports `reuse_applied=False`; startup and transport
consume its remaining allowance. The existing global scheduler is never reset.
QP, MILP and implicit solver
selection retain the previous stateless behavior. `reuse=False` remains default.
The same explicitly owned backend can be used with `solve()`, `SolveOptions`,
semantic reoptimization and production scenario sweeps.

Native update errors discard partially changed state and attempt a fresh model;
failed solves cannot return an earlier optimal result. `reset()` discards state;
`close()` is terminal. Lifecycle operations are serialized. Caller-owned options
must not be mutated concurrently. Dataclass replacement creates an independent
backend; native objects and locks never become configuration hashes.

Call-local time limits can change without losing a compatible model. HiGHS's
native clock accumulates across solves, so each run receives the prior clock plus
its new allowance. Removing a limit restores the native unlimited setting.
These remain native soft limits; construction, validation and cleanup can add
latency. The existing process-isolated deadline API has its own supported scope.

`reuse_applied` means the native model was retained. Raw statistics record the
reason, changed fields and observed basis validity before/after the run. They do
not assert factorization reuse or guarantee a faster solve.

## Preserve correctness while measuring cost

`PhaseTimings.primal_validate_s` and `certificate_s` are additive keyword-only
fields. `validate_s` remains their total. Native build, update, solve and total
backend times remain separate. Older traces default the new fields to zero;
that is missing historical breakdown, not proof of zero cost. Standard evidence
envelopes retain their existing timing allowlist; detailed trace dictionaries
and the qualification runners preserve the new breakdown.

The executor owns snapshots of configuration, candidate and raw certificate data
within the backend's reentrant lock. Concurrent budgeted calls cannot lend each
other temporary settings or overwrite a retained workspace during capture. Its
primal report can then be reused by an internal
certificate helper. Public `verify_optimality()` still validates independently.
Proof arithmetic and tolerances are unchanged. An inconsistent backend objective
claim retains its invalid result semantics and triggers a separately timed
canonical primal check where needed. Each changed model gets fresh validation;
no previous candidate's proof is reused.
Third-party backends retain ownership of their synchronization contract; an
arbitrary private `_lock` is not assumed to be reentrant by the executor.

## Evaluation protocols

* `benchmarks/qualify_highs_reuse.py` freezes a development-visible analytic
  challenge before execution. Independent two-variable block references cover
  active-row/bound changes, degeneracy, maximization, infeasible-to-feasible and
  unbounded-to-bounded transitions, rebuilds and small native budgets. Every
  outcome remains recorded. This is bounded work, not a hard wall-time guarantee.
* `benchmarks/qualify_public_reuse.py prepare` admits fresh Netlib families after
  prior name/family/raw/MPS/canonical-hash exclusions and independent MPS-reader
  agreement. A separate `run` cannot substitute cases based on solve outcomes.
  Bijective objective/RHS/bound transforms are checked by mapping candidates back
  to the original arrays and historical Netlib numerical objective references.
  The variants are related, and the references are not exact certificates.
* `benchmarks/compare_lp_workflows.py` v2 adds `solverpilot-reuse`, a presolve
  switch, actual optimality-evidence flags and split phase statistics. Its
  original four-workflow default is retained. All comparisons use one native
  HiGHS version/options; Pyomo's cumulative native clock is handled by a narrow
  adapter qualified for 6.10.1. CI runs functional observations, not speed gates.

```powershell
python benchmarks/qualify_highs_reuse.py --output NEW-TRANSITIONS
python benchmarks/compare_lp_workflows.py --preset standard --repeats 10 --warmups 1 --steps 4 --presolve --workflows solverpilot,solverpilot-reuse,direct-highspy,cvxpy-highs,pyomo-appsi-highs --output NEW-COMPARISON.json
```

Native model reuse, candidate acceptance and independently verified optimality
are counted separately. A time limit, solver-only termination or missing proof
cannot become an independent certificate merely because a workflow completed.
Background OS activity is uncontrolled; medians/p95 are descriptive. The public
cases broaden correctness evidence without establishing industrial performance,
human productivity, global nonlinear proof or learned-routing value.

## Production pilot

The public educational/synthetic pilot remains the current input scope. The
portable kit includes the [participant run checklist](pilot/PARTICIPANT-RUN.md),
paired task protocol and blank observation forms. Actual operating data and a
participant are pending; no feedback or human time saving is inferred from these
machine runs. The next real session must first confirm business rules and units,
freeze an independent reference and record every intervention and failure.

## Recorded final observations — October 13, 2026

The measured candidate is an **unreleased development wheel labelled 0.4**,
not another PyPI release. Its SHA-256 is
`3b8c5337507ab7a5ca609edc06163433a6364dc6b2afc44aac8c135ab5892196`;
its Python-source digest is
`ee0e8ae320ddf4f6a46ceace3d6f49811dab778a974639731b4fc6b1b7dab7b2`.
All 246 Python files match the measured Windows checkout. Seven pre-existing
files use CRLF locally and LF in Git; every Python AST matches. The Git source
byte digest is `b45028829db169c3cd82464c4e918f9416414de872f9ede40f0e6cdcc810da30`.
The JSON evidence lists the seven files. Use the exact supplied wheel for strict
replay rather than treating these different byte identities as interchangeable.
Result tables were written after the wheel build; wheel metadata therefore
predates these tables. These hashes
identify source bytes, not author identity or the running native binary's trust.

Windows 11 / Python 3.12.14 / highspy 1.15.1 observations:

| Check | Outcome |
| --- | --- |
| Complete local regression | 1,995 passed, 49 skipped; no failures or errors |
| Coverage | 85.61% lines, 71.49% branches, 81.77% combined |
| Analytic state-transition challenge | All 240 paired cases / 480 solves pass their gates |
| Analytic numerical optimality | 192 per mode; remaining 48 per mode not established |
| Fresh-public admission set | All 256 intended candidates accepted; 188 independently numerically verified |
| Version and five-interface comparisons | 11,136 measured calls and 1,344 warmups accepted |
| Clean wheel installations | Both environments pass; strict replay of all 8 production-fixture scenarios |
| Participant pilot | Pending operating data and an actual participant |

The skipped tests need unavailable optional hardware/native tooling, external
evidence or platform facilities. Ruff correctness rules and configured mypy
checks also pass. The analytic challenge explicitly includes infeasible,
unbounded and tiny-limit outcomes; passing its gates does not mean 480 optimal
solutions or independent termination certificates.

### Matched repeated-call comparison

Each five-interface run contains 12 synthetic sequences, four steps, ten measured
repetitions and one warmup per sequence/workflow/mode. All use the same native
library, solver options and column-bound formulation. The table reports pooled
**checked-total medians in milliseconds**; imports/startup and cleanup are outside
the per-call timing. SolverPilot retains its internal verification, and all five
receive the same independent candidate/reference check. This is not a complete
comparison of the packages' evidence-export workflows.

| Workflow | Fresh, presolve off | Update, off | Fresh, presolve on | Update, on |
| --- | ---: | ---: | ---: | ---: |
| SolverPilot stateless | 12.830 | 13.096 | 17.937 | 20.287 |
| SolverPilot reuse | 12.496 | 11.392 | 17.796 | 14.518 |
| Direct highspy | 2.825 | 0.707 | 5.197 | 0.816 |
| CVXPY / HiGHS | 15.403 | 4.521 | 25.781 | 5.676 |
| Pyomo / APPSI HiGHS | 39.942 | 7.931 | 58.463 | 12.840 |

Reuse improves all 12 update-case medians. Taking each case's reuse/stateless
median ratio, then the geometric mean across cases, gives **26.1% less time with
presolve off and 32.9% with it on**. Pooled median reductions are 13.0% and 28.4%;
these are different aggregations, not conflicting estimates. Cold-case geometric
ratios are 1.013 and 1.009, providing no cold-solve benefit. Each SolverPilot
variant independently verifies all 960 measured candidates per presolve setting;
all 360 retained updates per setting record native reuse. Other wrappers'
certificate work is unobserved, not measured as zero. Direct highspy and Pyomo
remain faster on every update case; CVXPY is faster on 10 of 12.

Within reuse updates, the median per-observation certificate/checked-total share
is about 56%; native solve share is about 2%. Component medians must not be added
as though they equal the median total.

### Stateless regression and repeatability

The separately frozen ABBA order compares the previous development wheel from
`11c4aeb` with the candidate, using unchanged inputs, dependencies and runner.
It includes 1,536 measured calls; all are accepted and independently numerically
verified. Pooled checked-total medians change **10.756 to 11.530 ms for fresh
models (+7.2%)**, and **9.739 to 10.600 ms for repeated stateless calls (+8.8%)**.
Equal-case geometric ratios of pooled case medians are 1.072 and 1.061; only
three of twelve case medians improve in each phase.

The October 12 precursor campaign instead observed pooled reductions of 17.0%
and 12.7%. Its source predates the final concurrency/configuration-binding fixes;
both campaigns and the reason for repeating are preserved. Unchanged rival
adapters also changed markedly between campaigns: direct highspy update medians
rose 28.1%/47.9% with presolve off/on. Background OS activity and temporal
variation are uncontrolled. We therefore make no causal claim that the final
fixes caused the regression, or that eliminating a duplicate primal check
establishes a stable stateless speedup. The regression remains an explicit cost
to investigate; the safer snapshots and verification are retained.

### Public families and verification limits

The outcome-independent admission selected GANGES, STOCFOR1, DEGEN3, FFFFF800,
TUFF, VTP.BASE, SEBA and BRANDY from [Netlib](https://www.netlib.org/lp/data/readme).
The frozen exclusion scope covers 15 earlier registers, 267 prior names,
348 model/content hashes and 75 previously downloaded MPS files, including
rejected imports. The eight admitted raw models and 32 transformed model hashes
match the freeze. Independent review loaded all 256 evidence records, checked
unique execution IDs and replayed all original-data candidate checks.

Five families verify all 32 numerical optima, FFFFF800 verifies 28/32, and
STOCFOR1 and BRANDY each verify 0/32. The 68 incomplete verifications report
stationarity error without a finite lower-bound correction; no wrong candidate
was observed. Cold/reuse pairs have **zero verification losses and four gains**;
coverage is unchanged from the precursor campaign. No result has the separate
`certificate_verified` flag, and no generic exact-certificate claim is made.

All 96 retained updates reuse the model and valid basis. Native solve medians
fall from **8.173 to 0.433 ms** with presolve off, and **8.336 to 0.380 ms** with
it on. Across the 96 update pairs, reuse is faster in 65; the median paired
checked-time ratio is 0.961. Summed update time is **48.86 s cold / 46.69 s reuse**.
Certificate work consumes **92.75% of summed checked time over all 256 calls**.
The related affine variants favor retained bases and do not qualify general
active-set-change performance. These observations establish neither industrial
speed nor population-wide superiority.

The next measured improvement should target certificate cost and the 68
incomplete numerical checks while preserving conservative arithmetic. Native
reuse stays opt-in; learned-routing promotion and human-productivity claims
remain disabled. The portable pilot kit now includes the participant checklist
and blank forms, ready for the operating data and participant the user will
provide later.
