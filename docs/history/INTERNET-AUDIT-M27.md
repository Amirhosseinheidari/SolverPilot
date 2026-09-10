# M27 Internet / Literature Audit — Selective Absolute-Regret Algorithm Selection

This audit distinguishes **literature findings** from **OptiMind design inferences**. It was used to freeze M27 before the one-time held-out test. The negative held-out result did not trigger any post-test model or threshold changes.

## 1. Per-instance algorithm selection must optimize the actual performance measure

**Literature finding.** The Rice-style algorithm-selection problem maps each instance to an algorithm in order to optimize a performance measure. Modern reviews and meta-level formulations treat the single-best solver (SBS) as the default baseline and the virtual-best solver (VBS) as an oracle bound. Most learned selectors approximate solver performance rather than optimizing classification accuracy alone.

Sources:
- Kerschke et al., *Automated Algorithm Selection: Survey and Perspectives*, arXiv:1811.11597.
- Tornede et al., *Algorithm selection on a meta level*, Machine Learning (2022), DOI 10.1007/s10994-022-06161-4.

**OptiMind implication.** M27 predicts an **absolute wall-clock saving** relative to a default solver and evaluates the complete policy cost against SBS. Winner-label accuracy is not a promotion metric.

## 2. Performance prediction can be useful, but errors have asymmetric downstream cost

**Literature finding.** Empirical performance models predict algorithm runtime from instance features. SATzilla-style portfolio selection moved toward cost-sensitive predictors because not all solver-selection mistakes have the same consequence.

Sources:
- Hutter, Xu, Hoos, Leyton-Brown, *Algorithm Runtime Prediction: Methods & Evaluation*, arXiv:1211.0906.
- SATzilla project / SATzilla 2011: cost-sensitive decision forests for performance prediction.

**OptiMind implication.** M26's relative log-ratio target underweighted catastrophic absolute regret. M27 instead regresses the absolute saving of switching away from the development SBS. Squared error deliberately gives large wall-clock errors more influence.

## 3. Abstention / reject-option prediction is the correct conceptual model

**Literature finding.** Selective prediction trades coverage for lower risk. Foundational work characterizes risk-coverage trade-offs; later reject-option work shows that optimal rejection decisions are driven by conditional risk/uncertainty, not by blindly emitting a prediction for every input.

Sources:
- El-Yaniv & Wiener, *On the Foundations of Noise-free Selective Classification*, JMLR 2010.
- Franc, Prusa & Voracek, *Optimal Strategies for Reject Option Classifiers*, JMLR 2023.
- Geifman & El-Yaniv, *SelectiveNet*, ICML 2019.

**OptiMind implication.** The learned selector does **not** have to choose the alternate LP method. It defaults to the development SBS and switches only when an empirical lower-bound statistic exceeds a positive saving margin. Out-of-support inputs abstain to the default.

## 4. Risk aversion and censored/time-limited observations matter in algorithm selection

**Literature finding.** Run2Survive models censored runtimes using survival analysis and explicitly advocates risk-averse algorithm selection where avoiding timeouts has high priority.

Source:
- Tornede et al., *Run2Survive: A Decision-theoretic Approach to Algorithm Selection based on Survival Analysis*, arXiv:2007.02816.

**OptiMind implication.** M27 keeps the conservative cutoff-valued accounting inherited from M25. It does not impute a hidden exact runtime beyond a timeout. The selected rule is low-capacity and abstains under extrapolation rather than chasing average speedup.

## 5. Feature cost must be part of the selector cost

**Literature finding.** Recent work directly studies feature-computation budget in per-instance algorithm selection and reports that feature cost can materially explain the gap between a learned selector and the VBS; the best feature budget is scenario-dependent.

Source:
- van der Blom & Vermetten, *On the Influence of the Feature Computation Budget on Per-Instance Algorithm Selection for Black-Box Optimization*, arXiv:2605.04954 (2026).

**OptiMind implication.** Every M27 policy decision is charged the measured `inspect_problem` cost. The static SBS baseline is not charged selector feature cost.

## 6. Small datasets and portfolio/model complexity create overfitting risk

**Literature finding.** Portfolio-based algorithm selection has a generalization trade-off: larger/more complex portfolios and selectors can increase overfitting. Modern evaluation work likewise emphasizes benchmark variance and the need to keep model-selection degrees of freedom under control.

Sources:
- Balcan, Sandholm & Vitercik, *Generalization in portfolio-based algorithm selection*, arXiv:2012.13315.
- Bouthillier et al., *Accounting for Variance in Machine Learning Benchmarks*, arXiv:2103.03098.

**OptiMind implication.** With only 32 development rows, M27 intentionally uses a depth-2 CART with minimum leaf size 8. Random forests, boosted trees and large hyperparameter sweeps were rejected for this milestone.

## 7. Leakage and benchmark protocol quality are first-class concerns

**Literature finding.** BenLOC identifies data leakage, homogeneous datasets, inconsistent train/test splits and weak baselines as causes of over-optimistic claims in learned MIP configuration.

Source:
- Li et al., *BenLOC: A Benchmark for Learning to Configure MIP Optimizers*, arXiv:2506.02752 (2025).

**OptiMind implication.** M27 acknowledges that M26's validation set became adaptively used development data after the M26 failure. Therefore all 32 train+validation rows are development-only; the original 16 test rows are the sole promotion evidence. Once M27 opens them, they are permanently consumed.

## 8. Selective-system evaluation itself can be misleading

**Literature finding.** Recent selective-classification work shows that evaluation at one working point can mischaracterize confidence/rejection quality and motivates richer risk-coverage analysis.

Source:
- Traub et al., *Overcoming Common Flaws in the Evaluation of Selective Classification Systems*, arXiv:2407.01032 (2024).

**OptiMind implication.** M27 does not claim a generally calibrated uncertainty model from one 16-instance test. The promotion gate combines average policy/SBS, a paired bootstrap interval, switch count, p90 tail ratio and maximum harmful-switch regret. The empirical leaf bootstrap is explicitly **not** claimed as a formal conformal guarantee.

## 9. Easy-instance bypass is a relevant future direction

**Literature finding.** Recent algorithm-selection work shows that identifying easy instances can avoid unnecessary feature-computation overhead and reserve selection effort for harder instances.

Source:
- Renau & Hart, *Identifying Easy Instances to Improve Efficiency of ML Pipelines for Algorithm-Selection*, arXiv:2406.16999 (2024).

**OptiMind implication.** M27 still pays inspection on every selection decision. If a future selector is attempted, a cheap pre-inspection bypass may be more valuable than making the selector itself more complex. This is a future hypothesis, not an M27 claim.

# Post-test literature-consistent interpretation

The one-time M27 test failed the pre-registered promotion gate: point policy/SBS was above 1 and the paired bootstrap interval crossed and extended above 1. Correctness/accounting gates remained green. The result is consistent with the literature's central warnings: small-sample confidence estimates need not generalize, feature costs matter, and selective abstention reduces but does not eliminate routing risk.

No M27 post-test retuning is permitted. Any next learned selector must use a **new untouched test cohort**.

## 10. Native HiGHS auto-selection is a mandatory future baseline

**Current implementation/documentation finding.** The current SciPy `linprog` API defaults to `method='highs'`. Its documentation states that `highs-ds` and `highs-ipm` expose the HiGHS dual-simplex and interior-point solvers and that `highs` chooses between the two automatically; it also explicitly notes that which explicit method is faster is problem-dependent.

Source:
- SciPy 1.18.0 manual, `scipy.optimize.linprog`, current documentation accessed 2026-09-06.

**OptiMind implication.** A custom learned DS/IPM selector must not be judged only against static DS/IPM SBS. Before another ML selector is justified, a fresh benchmark must compare:

`highs-ds` vs `highs-ipm` vs native `highs` auto vs VBS.

If native HiGHS auto closes most of the VBS gap without external feature/planning cost, an external learned router has little product value even if it can beat a static SBS.
