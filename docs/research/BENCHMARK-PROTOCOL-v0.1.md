# Benchmark Protocol v0.1

**Date:** 2026-08-30  
**Status:** Proposed; no benchmark runs have been executed yet.

The benchmark is not a marketing demo. It is the mechanism that decides whether the project’s core hypotheses survive.

---

## 1. Questions the benchmark must answer

### Q1 — Correctness
Can every supported adapter produce a solution that validates against the canonical model?

### Q2 — Runtime overhead
How much time do inspection, planning, compilation and validation add?

### Q3 — Solver-selection value
Does the rule-based policy outperform a fixed Single Best Solver on held-out instances after paying feature/planning cost?

### Q4 — Reoptimization value
On sequences of related problems, does safe state reuse reduce **end-to-end sequence time** compared with cold rebuild/solve?

### Q5 — Robustness
Can the runtime distinguish feasible, infeasible, unbounded, numerical-failure, timeout-with-incumbent and invalid-solution cases reliably?

### Q6 — Diagnostics
When an infeasibility explanation is produced, is it valid and correctly typed?

---

## 2. Benchmark infrastructure

Use **Benchopt 1.9.x** rather than building experiment orchestration from zero.

Useful documented properties:
- reproducible seeding;
- solver/dataset parameter grids;
- caching;
- multiple repetitions;
- parallel and cluster execution;
- convergence tracking;
- fresh-environment benchmark tests;
- result serialization.

Sources:
- https://benchopt.github.io/stable/
- https://benchopt.github.io/stable/get_started.html
- https://benchopt.github.io/stable/user_guide/controlling_randomness.html
- https://benchopt.github.io/stable/benchmark_workflow/test_benchmark.html

Repository suggestion:

```text
benchmarks/
    optimind-benchmark/
        objective.py
        datasets/
            micro_correctness.py
            netlib_lp.py
            qplib_convex.py
            miplib2017.py
            repeated_qp.py
            repeated_lp.py
            repeated_milp.py
            robustness.py
        solvers/
            direct_highs.py
            direct_osqp.py
            direct_scip.py
            direct_gurobi.py
            optimind_fixed.py
            optimind_auto.py
```

---

## 3. Dataset tiers

### Tier 0 — correctness micro-corpus

Hand-constructed tiny instances with exact expected behavior.

Must include:
- feasible bounded LP;
- infeasible LP;
- unbounded LP;
- degenerate LP;
- badly scaled but solvable LP;
- convex QP;
- infeasible convex QP;
- binary MILP;
- mixed integer/continuous MILP;
- MILP with known incumbent under a tight time limit;
- models with fixed/free/ranged variables and constraints;
- empty/singleton rows;
- objective offset;
- max/min sense;
- invalid/nonconvex QP input.

Goal: contract testing, not performance.

### Tier 1 — public static corpora

#### MILP — MIPLIB 2017 Benchmark Set
Use current benchmark set version 2.

Official benchmark set contains **240 instances** and was selected using feature/performance coverage criteria while screening benchmark suitability.

Sources:
- https://miplib.zib.de/set_benchmark.html
- https://miplib.zib.de/download
- https://miplib.zib.de/Selection_Methodology.html

Do not silently cherry-pick only easy instances.
If using a subset for CI, publish the exact list and selection rule.

#### Convex QP — QPLIB filtered subset
QPLIB currently contains:
- 134 continuous instances, of which the statistics page reports 32 convex;
- 319 discrete instances, of which 31 are convex.

v0.1 is **continuous convex QP only**, so filter using QPLIB metadata/convexity classification rather than assuming the entire library is supported.

Sources:
- https://qplib.zib.de/
- https://qplib.zib.de/statistics.html

#### LP
Use an established LP corpus such as Netlib for static LP tests, plus our own modern sparse parametric sequences.

Before bundling any third-party instances in a wheel/repository, review redistribution terms. If terms are ambiguous, provide a downloader/manifest rather than redistributing data.

### Tier 2 — repeated / parametric workloads

This tier is critical because reoptimization is one of the product’s core differentiators.

#### Repeated convex QP
Generate MPC-style sequences:
- fixed `P` sparsity;
- fixed `A` sparsity;
- changing `q`, `l`, `u`;
- occasional numeric-value changes in `P/A` with unchanged sparsity;
- controlled horizon and conditioning.

Measure OSQP/update paths vs cold construction and other backends.

#### Rolling-horizon LP
Keep structure mostly fixed while changing:
- costs;
- demands/RHS;
- capacity bounds.

Then introduce structural-change checkpoints to verify that stale basis/reuse is invalidated correctly.

#### Repeated MILP
Families:
- knapsack variants;
- assignment/generalized assignment variants;
- facility-location demand updates;
- rolling scheduling-like MILPs without CP-specific primitives.

Mutate:
- objective coefficients;
- RHS/bounds;
- known incumbent quality.

Use HiGHS/SCIP/Gurobi optional MIP starts where supported.
Do not assume SCIP reoptimization or MIP starts behave identically.

### Tier 3 — robustness / diagnostics

Construct controlled pathologies:
- contradictory bounds;
- duplicate conflicting equalities;
- hidden conflicts requiring multiple rows;
- near-infeasible systems;
- coefficient ranges spanning increasing powers of ten;
- nearly dependent rows;
- large Big-M cases;
- primal unboundedness;
- models where solver terminates before proof.

Ground truth must be known or independently checked.

### Tier 4 — later ML/OOD benchmark

Only after enough trace data exists.

Use family/group holdouts.
Never randomly split variants of the same generator across train/test.

BenLOC specifically warns that inconsistent datasets/splits and leakage can produce over-optimistic claims in learned MIP configuration.

Source:
https://arxiv.org/abs/2506.02752

---

## 4. Baselines

Every adaptive benchmark must include:

### Direct backend baselines
- direct HiGHS default;
- direct OSQP default on supported QP;
- direct SCIP default;
- optional direct Gurobi default;
- later direct Clarabel/cuOpt where relevant.

### Fixed runtime baselines
Use our runtime plumbing but force exactly one backend.
This isolates **orchestration overhead** from solver-selection value.

Example:
- `runtime(force="highs")`
- `runtime(force="scip")`

### SBS — Single Best Solver
The single candidate with best aggregate metric on the training/reference population.

### VBS — Virtual Best Solver
Oracle that chooses the best candidate per instance after seeing outcomes.
VBS is not deployable; it represents the ceiling available to a selector for the given portfolio.

### Policy
- deterministic rules v0;
- later learned selector.

---

## 5. Primary metrics

### Correctness
- validated solve rate;
- invalid-solution rate;
- false optimal rate **must be zero** in accepted tests;
- false infeasible rate;
- objective error vs known/reference solution;
- max constraint violation;
- max bound violation;
- max integrality violation.

### Time
Record separately:
- load/parse;
- inspect;
- plan;
- transform;
- backend model build;
- backend solve;
- validation;
- diagnostics;
- end-to-end.

### MILP quality under budget
- time to first feasible;
- incumbent objective over time;
- best bound over time;
- absolute gap;
- relative gap;
- time to target;
- nodes where exposed.

### Resource usage
- peak RSS;
- CPU threads;
- GPU memory for GPU path when available.

### Reoptimization
- total sequence wall time;
- per-step wall time;
- speedup vs cold;
- rebuild count;
- reuse-mode distribution;
- validation failure count;
- stale-state bug count (must remain zero).

---

## 6. Algorithm-selection metrics

### PAR10
For runtime-selection studies, timeout runs may be penalized as 10× cutoff when producing PAR10-style analyses.

### SBS/VBS gap closure

For a cost metric where lower is better:

```text
gap_closure =
    (SBS_cost - policy_cost) /
    (SBS_cost - VBS_cost)
```

Interpretation:
- 0: no gain over SBS;
- 1: matches VBS;
- <0: worse than SBS.

Always include feature/inspection cost in `policy_cost`.

For solution-quality objectives, define a normalized direction-aware counterpart rather than blindly reusing the runtime formula.

### Coverage
Also report:
- fraction of instances where each solver wins;
- policy confusion/win matrix;
- abstention/fallback rate.

---

## 7. Timeout policy

Use multiple suites:

### CI smoke
- tiny curated corpus;
- seconds per instance;
- purpose: correctness/regression only.

### Nightly
- representative public subsets;
- tens of seconds per instance.

### Research/release
- full or large benchmark sets;
- meaningful longer cutoffs;
- hardware reserved and controlled.

Never compare methods with different cutoffs.

Record both timeout and whether a validated incumbent exists.

---

## 8. Threading and hardware protocol

For algorithmic comparison:
- default primary suite: 1 solver thread where supported;
- pin/record CPU allocation;
- no concurrent Benchopt workers that oversubscribe the solver;
- separate multi-thread scaling suite.

Benchopt itself warns that parallel benchmark jobs can reduce lower-level thread counts and therefore should not be conflated with sequential solver timings.

Source:
https://benchopt.github.io/stable/user_guide/distributed_run.html

Record:
- CPU model;
- physical/logical core count;
- RAM;
- OS/kernel;
- Python version;
- package versions;
- solver versions;
- BLAS if relevant;
- GPU model;
- driver/CUDA/cuOpt version;
- thread limits;
- environment variables affecting parallelism.

---

## 9. Repetitions and randomness

Deterministic solver modes:
- at least warm-up + repeated timing where variance is meaningful.

Stochastic/nondeterministic solver configurations:
- minimum 5 seeds for research comparisons;
- same intended random-seed schedule across comparable candidates where semantics permit;
- publish medians and uncertainty intervals.

Benchopt provides deterministic seed derivation across experiment axes.

Source:
https://benchopt.github.io/stable/user_guide/controlling_randomness.html

---

## 10. Train / validation / test policy for future selector

Do not create random row-level splits of generated sibling instances.

Preferred:
- hold out entire generator families;
- for MIPLIB, respect instance `Group` metadata where relevant;
- keep configuration/tuning set separate from final test;
- never use final test outcomes to alter planner rules;
- preserve a frozen external test set.

For benchmark evolution:
- version dataset manifests;
- hash each instance;
- publish excluded instances and reasons.

---

## 11. Performance claim gates

### Claim: “low overhead”
Allowed only if the measured orchestration overhead on the target suite satisfies the release target and raw timing components are published.

### Claim: “automatic solver selection improves performance”
Allowed only if:
- policy beats SBS on held-out benchmark aggregate;
- uncertainty supports the gain;
- inspection cost is counted;
- no hidden solver-specific tuning is applied only to our path.

### Claim: “faster repeated solves”
Allowed only per documented mutation classes/backend combinations where cold-vs-reuse sequence benchmark confirms it.

### Claim: “diagnoses infeasibility”
Must be qualified by mechanism:
- IIS;
- infeasibility certificate;
- elastic relaxation;
- heuristic conflict localization.

Do not present all four as equivalent.

---

## 12. First experiments to run once code exists

Experiment A — adapter correctness
- Tier 0 across HiGHS/OSQP/SCIP.

Experiment B — orchestration overhead
- direct backend vs forced runtime backend.

Experiment C — repeated QP
- OSQP cold vs OSQP update/warm path;
- compare solution validity and total sequence time.

Experiment D — LP basis reuse
- HiGHS cold vs basis reuse under small vs large perturbations.

Experiment E — MILP starts
- HiGHS/SCIP/(Gurobi optional) with and without prior incumbent.

Experiment F — deterministic policy
- MIPLIB representative subset;
- policy vs SBS/VBS;
- no learned model.

Experiment G — diagnostics
- curated infeasible/ill-conditioned corpus.

---

## 13. What this protocol does NOT claim yet

No experiment has been executed as of 2026-08-30.

Therefore we do not yet know:
- the best default MILP backend;
- whether the proposed selector closes any SBS→VBS gap;
- how large the runtime overhead is;
- which fingerprint features are predictive;
- how much reoptimization improves real workloads;
- whether pilot races justify their overhead;
- whether GPU routing is beneficial at any specific size threshold.

Those are experimental questions, not design assumptions.
