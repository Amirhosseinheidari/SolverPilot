# Benchmark Report — M27 Selective Absolute-Regret Selector

## Motivation

M26 failed because relative log-cost prediction did not align with absolute product regret: one expensive wrong decision could erase many small wins. M27 therefore models absolute saving and adds a reject/abstain path: use a default solver unless the evidence for switching is strong enough.

This benchmark deliberately separates **development evidence** from the **one-time held-out promotion test**.

## Development-only evaluation

The original M26 train and validation rows (32 total) are treated as development data because M26's negative validation result influenced the M27 design. A group-safe leave-one-group-out audit of the frozen model family reported:

| Metric | Result |
|---|---:|
| development SBS | `scipy-highs-ipm` |
| SBS mean | 0.1044861 s |
| policy mean | 0.1024160 s |
| policy/SBS | **0.9801884** |
| switches | 6 / 32 |
| harmful switches | 0 |
| mean inspection | 0.0010007 s |
| p90 policy/default | 1.0759 |

The development gate passed. This result only allowed the selector to be **frozen**; it did not authorize production ranking.

## One-time held-out protocol

The recorded 16-instance held-out run was executed after hashes of the protocol, split, and model were fixed. Final audit confirms this artifact chronology and the write-once runner guard, while explicitly avoiding a stronger cryptographic one-time-execution claim.

- 16 exact test instances
- two interleaved rounds per backend
- `scipy-highs-ds` and `scipy-highs-ipm`
- solver threads = 1
- controller workers = 4
- cutoff = 1 s
- controller hard timeout = 3 s
- policy cost includes measured inspection time
- static SBS does not pay selector overhead
- paired bootstrap: 20,000 draws, seed 270027

## Held-out result

| Metric | Result |
|---|---:|
| outcomes | **64 / 64** |
| worker errors | **0** |
| hard timeouts | **0** |
| solver-level error outcomes | **2** (DS, one instance, cutoff-charged) |
| objective mismatches | **0** |
| test SBS | `scipy-highs-ipm` |
| SBS mean | **0.1442225 s** |
| policy mean | **0.1453332 s** |
| policy/SBS | **1.0077011** |
| bootstrap 95% | **[0.99643, 1.02718]** |
| switches | **2 / 16** |
| p90 policy/default | **1.04257** |
| max harmful switch | **0.0154641 s** |

Required promotion values were policy/SBS <= 0.97 and bootstrap upper < 1. Both failed.

## The opportunity still exists

Post-test analysis, performed only after the test became consumed, shows that failure was **not** caused by absence of a DS/IPM opportunity:

- DS wins: 9
- IPM wins: 7
- VBS mean: 0.1274337 s
- VBS gain vs test SBS: **11.64%**
- oracle + measured inspection mean: 0.1279107 s
- oracle + inspection gain vs test SBS: **11.31%**

So M27 is a selector-generalization failure, not an opportunity failure.

## Switch-level analysis

The model switched to DS on only two test instances:

1. `neos-4954672-berkel.mps.gz`
   - IPM - DS saving: +0.005325 s
   - useful switch
2. `uct-subprob.mps.gz`
   - IPM - DS saving: -0.015464 s
   - harmful switch

Net solver saving from the two switches was **-0.010139 s** before selector overhead. The harmful `uct-subprob` decision was enough to turn the policy negative.

The abstention/support guard did prevent several risky switches, but it also abstained on multiple profitable DS cases. This indicates the current five-feature geometry is not sufficient to separate safe-switch and unsafe-switch regions reliably.

## Scientific conclusion

M27 is a valid negative held-out result:

- correctness and accounting are clean,
- the test was used once,
- the opportunity remains large,
- the frozen selector failed to generalize enough to beat SBS,
- no post-test tuning is allowed,
- no performance-routing claim is made.

A future selector needs a fresh held-out cohort.
