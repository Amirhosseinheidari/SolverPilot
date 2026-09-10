# M27 Post-hoc Literature Review — What the Negative Test Means

**Boundary:** this review was written after the one-time M27 test was consumed. It did not change the M27 model, threshold, features, split, or promotion gates. It is interpretation only.

## 1. Algorithm selection should be judged on downstream cost, not label accuracy

The modern algorithm-selection literature treats per-instance selection as choosing among algorithms to optimize a performance measure. The Single Best Solver (SBS) is the relevant deployable baseline and the Virtual Best Solver (VBS) is an oracle bound. This supports M27's use of policy wall-clock cost, including inspection overhead, rather than winner-label accuracy.

Primary references:
- Kerschke, Hoos, Neumann, Trautmann, *Automated Algorithm Selection: Survey and Perspectives*, Evolutionary Computation 27(1), 2019, DOI 10.1162/evco_a_00242; arXiv:1811.11597.
- Balcan, Sandholm, Vitercik, *Generalization in Portfolio-Based Algorithm Selection*, AAAI 2021; arXiv:2012.13315.

## 2. Risk-sensitive selection is more appropriate than average-runtime chasing

Run2Survive models censored runtimes and explicitly advocates risk-aware selection when timeouts or expensive mistakes matter. M27's failure is consistent with this: one harmful switch can erase a useful switch even when average predicted saving inside a development leaf was positive.

Reference:
- Tornede, Wever, Werner, Mohr, Hüllermeier, *Run2Survive: A Decision-theoretic Approach to Algorithm Selection based on Survival Analysis*, arXiv:2007.02816.

## 3. Selective prediction / reject-option framing is conceptually correct

Reject-option methods trade coverage for lower conditional risk. The key idea is not to force a model decision on every input. M27 implements this as a safe fallback to the development SBS and only switches inside a supported leaf with a positive empirical lower saving.

References:
- Geifman & El-Yaniv, *SelectiveNet: A Deep Neural Network with an Integrated Reject Option*, ICML 2019, PMLR 97.
- Franc & Prusa, *On Discriminative Learning of Prediction Uncertainty*, ICML 2019, PMLR 97.
- Traub et al., *Overcoming Common Flaws in the Evaluation of Selective Classification Systems*, NeurIPS 2024; arXiv:2407.01032.

## 4. M27 must not claim conformal guarantees

Recent conformal risk-control work can provide finite-sample or high-probability risk control under explicit assumptions. M27's leaf bootstrap statistic is not a conformal method and was fitted after adaptive tree partitioning on only 32 development rows; therefore it must remain an empirical safety heuristic, not a formal coverage/risk guarantee.

References:
- Angelopoulos et al., *Conformal Risk Control*, arXiv:2208.02814.
- Bai & Jin, *Conformal Selective Prediction with General Risk Control*, arXiv:2603.24704 (2026).
- Laghuvarapu, Deb, Sun, *KMM-CP: Practical Conformal Prediction under Covariate Shift via Selective Kernel Mean Matching*, UAI 2026, PMLR 337.

## 5. Distribution shift and support overlap matter

Recent work on conformal/selective prediction under covariate shift emphasizes that reliable decisions require support overlap or explicit shift handling. M27's min/max leaf support guard is directionally aligned with this concern, but it is only a crude geometry check. The held-out result shows why: the guard prevented both very bad and useful switches. It reduced risk but did not produce sufficient discrimination.

## 6. Feature-cost bypass is a serious future direction

Algorithm-selection pipelines can lose their gains simply by paying analysis/feature cost on easy instances. Renau & Hart (2024) explicitly study bypassing algorithm selection for easy instances. M27 still pays inspection on every decision, so a future system may need an extremely cheap first-stage gate before richer features are computed.

Reference:
- Renau & Hart, *Identifying Easy Instances to Improve Efficiency of ML Pipelines for Algorithm-Selection*, arXiv:2406.16999.

## 7. What M27 actually falsified

M27 did **not** falsify the existence of DS/IPM complementarity. The consumed test still has a material VBS gap. It falsified this specific deployment hypothesis:

> A depth-2 CART on five cheap structural features, with a 5 ms empirical lower-saving threshold and leaf-support abstention, generalizes well enough to beat the held-out SBS after inspection cost.

The one-time test rejected that hypothesis. No threshold or tree retuning is permitted on those 16 instances.

## 8. Design implication for the next research milestone

A future learned selector must start with a **new untouched public/OOD cohort**. The consumed M27 test may be used only for historical postmortem, never model selection. Before opening a new test, development work should focus on:
- stronger but still cheap structure features,
- direct conditional-regret/risk modeling,
- explicit solver-failure/censoring handling,
- an easy-instance bypass,
- calibration of switching risk on development/calibration data,
- and pre-registered risk/coverage metrics, not only one threshold working point.

No future model may claim improvement unless it passes a fresh held-out test with full environment and runner provenance embedded at execution time.
