# Benchmark Report — M2

**Date:** 2026-08-31  
**Scope:** synthetic research smoke only. **No market/SOTA performance claim.**

## Why M2 changes the benchmark methodology

M1 mostly checked that multiple solver paths agreed on solutions. M2 adds a portfolio-selection question:

> Is there a meaningful per-instance performance gap between candidates, and if so does an automatic policy close any of the SBS→VBS gap after paying inspection/planning cost?

The benchmark code now computes:

- SBS (Single Best Solver): best fixed solver by mean cost;
- VBS (Virtual Best Solver): oracle best solver per instance;
- deployable policy cost including policy overhead;
- SBS→VBS gap closure;
- solver wins/ties;
- performance profiles.

## Experiment A — 48 small synthetic LPs

Command:

```bash
python benchmarks/m2_lp_portfolio.py
```

Latest recorded result:

```text
instances: 48
DS wins: 46
IPM wins: 2
max |objective_DS - objective_IPM|: 0.0
DS mean cost:  0.024312389 s
IPM mean cost: 0.036415985 s
SBS: scipy-highs-ds
SBS cost: 0.024312389 s
VBS cost: 0.024266828 s
current auto policy cost: 0.024864984 s
```

The SBS→VBS difference is only ~45.6 microseconds on average. As a result, a small amount of inspection/planning/runtime overhead overwhelms the tiny oracle opportunity and the numerical gap-closure ratio becomes very negative.

**Interpretation:** this is not a failure of the metric. It tells us this dataset is poor for testing algorithm selection: there is almost no portfolio complementarity to exploit.

## Experiment B — family-separated synthetic selection smoke

Command:

```bash
python benchmarks/m2_selector_research.py
```

The generator families used for fitting and testing are disjoint. The train set and holdout each have 8 instances.

Winner balance:

```text
train: DS 4 / IPM 4
holdout: DS 4 / IPM 4
```

Latest holdout metrics:

| Metric | Current capability-only auto | Benchmark-only depth-2 tree |
|---|---:|---:|
| SBS cost | 0.192080606 s | same |
| VBS cost | 0.175536233 s | same |
| Policy cost | 0.195784551 s | 0.180566106 s |
| SBS→VBS gap closure | -22.39% | 69.60% |
| strict wins | 4 | 6 |
| ties within tolerance | 4 | 6 |

The tiny learned rule fitted by this synthetic run is effectively:

```text
log10(number_of_constraints) <= 2.80 -> DS
otherwise                           -> IPM
```

This rule is deliberately confined to `benchmarks/`; it is not imported by the runtime planner.

### What the result supports

It supports only a narrow empirical statement:

> In a deliberately complementary synthetic LP portfolio, cheap structural instance features contained enough signal for a trivial held-out selector to capture a substantial portion of the SBS→VBS timing opportunity in this particular run.

### What the result does NOT support

It does not show that:

- the rule generalizes to real LP workloads;
- the same features select between different solver vendors;
- learned selection is already production-ready;
- the project beats HiGHS, SCIP, Gurobi, OSQP, or any modeling framework;
- the measured percentage is stable across hardware/runs.

Timing variance is visible across repeated executions of these small suites. Larger controlled benchmark runs and uncertainty estimates are required.

## Public benchmark targets prepared for the next phase

MIPLIB 2017:

- official benchmark set version 2;
- 240 instances;
- benchmark archive approximately 317.3 MB;
- current solution file v36 released 2026-01-26.

QPLIB:

- 134 continuous instances;
- 32 classified convex;
- v0.1 should filter to supported continuous convex QPs rather than claiming the entire library.

The current container cannot download the corpora, so **zero MIPLIB/QPLIB problem instances have been run in M2**.

## Raw results

- `benchmarks/results/m2-lp-portfolio.json`
- `benchmarks/results/m2-selector-research.json`
- `benchmarks/results/m2-osqp-translation-checks.json`
- `benchmarks/results/m1-property-checks.json`
