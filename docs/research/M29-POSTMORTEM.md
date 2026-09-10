# M29 POSTMORTEM

## Decision

M29 does **not** authorize M30 selector research on the current LP representation. Learned LP performance routing is closed for OptiMind 1.0.

## What failed

The failure is not one-dimensional:

1. **Sparse stable signal:** only one of 161 features passes the pre-registered cross-cohort univariate stability screen.
2. **Signal is mostly non-independent:** residualizing that feature against base size/density nearly removes its M25 relationship and weakens M28.
3. **Poor transfer:** no cumulative feature family reaches the required cross-cohort rank correlation in both directions.
4. **Policy failure persists at zero feature cost:** every transfer policy remains worse than target SBS even when extraction cost is artificially removed.
5. **Rich feature cost is material:** static/topology/presolve families consume a significant fraction of the solve budget in the current Python implementation.

## What did not explain the failure

### Timing noise alone

Repeated M25 solver deltas have Spearman ~0.936 and >91% sign agreement. M28 round-level deltas have median pairwise Spearman ~0.922. Labels are not perfectly stable, but they are far too stable for timing noise to explain the cross-cohort collapse by itself.

### A single outlier alone

Opportunity is concentrated, especially in M28, but no single instance accounts for a majority of VBS savings in either cohort.

### Feature computation cost alone

The zero-cost counterfactual still fails. A faster implementation would reduce overhead, but would not fix the observed lack of cross-cohort portability.

## Likely explanation

The current cheap/static representation mostly describes problem size, sparsity and coarse coefficient/bound structure. Those characteristics correlate with DS/IPM behavior inside some cohorts, but they do not define stable enough algorithm-performance regions across the M25 and M28 distributions.

The base-feature distribution itself shifts materially between cohorts (especially `n`, `m`, and `nnz`), making simple extrapolation unreliable. This is exactly the setting where post-outcome feature mining would be most dangerous.

## Product consequence

For 1.0, the correct engineering decision is to stop spending complexity budget on learned LP routing. OptiMind still retains:

- deterministic proof-safe baselines;
- structural inspection;
- exact presolve/reductions where justified;
- reoptimization/session intelligence;
- diagnostics and independent validation;
- specialized exact algorithms;
- evidence-aware routing without an unsupported speed claim.

A future learned LP selector should restart only with materially new data/representation, explicit censored-runtime handling, and a new untouched public/OOD evaluation corpus.
