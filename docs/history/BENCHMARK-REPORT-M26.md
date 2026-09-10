# Benchmark Report — M26 Cost/Regret-Aware Selector Validation

## Question

M25 established ~10% oracle-with-inspection headroom on a 48-instance public/OOD cohort. M26 asks the next question: can a simple, low-capacity learned selector capture enough of that opportunity on validation to justify opening the final held-out test?

## Leakage-controlled split

The M25 cohort was split by SHA-256 of a group identifier, independent of timing and winner labels:
- train: 20
- validation: 12
- test: 16
- groups: 45

`csched007/008`, `markshare*`, and `ic97/icir97` were group-locked. Native split validation reports complete coverage and no group leakage.

The derived `m26-trainval-only.json` contains exactly 32 rows and deliberately omits all test costs/winners. The frozen split hash is:

`6e94069451a07a8ec7ef786bff70ec844afeedd5282c93258fe47615ab0415e6`

## Model

One pre-registered model family was allowed:

`ridge regression -> log(IPM_cost / DS_cost)`

Features:
- `log1p(n)`
- `log1p(m)`
- `log1p(nnz)`
- `log(density)`
- `log1p(m/n)`

Grid:
- lambda: 0.1, 1.0, 10.0
- selection threshold: -0.05, 0.0, 0.05

Only validation policy cost selected the hyperparameters. Selector policy cost includes measured inspection overhead; SBS does not pay selector feature cost.

## Result

All 9 candidates lost to validation SBS:

- best policy/SBS: **1.1886926884**
- worst policy/SBS: **1.2217607543**
- required gate: <= 0.98

Frozen best candidate:
- lambda = 10.0
- threshold = 0.0
- DS choices = 10
- IPM choices = 2

Therefore the validation gate failed and the final test was not evaluated.

## Failure analysis

The failure is not explained by feature overhead alone. The dominant issue is cost asymmetry: many validation instances favor DS by a few milliseconds, while one expensive mistake can erase those gains.

The largest example is `seymour1.mps.gz`:
- predicted `log(IPM/DS)` = +0.1313 -> model selects DS
- actual `log(IPM/DS)` = -0.8312 -> IPM is much faster
- DS cost = 0.6935 s
- IPM cost = 0.3020 s
- policy excess versus validation SBS route ~= **0.3928 s**

That one mistake accounts for about **99.2% of the model's positive validation regret** before offsetting savings on other instances.

This reveals an objective mismatch: predicting relative log speed equally across instances is not the same as minimizing absolute runtime regret. Winner/classification accuracy is also insufficient; expensive mistakes deserve much larger weight.

## Scientific conclusion

M26 is a successful negative-result milestone. It prevents a plausible low-capacity selector from consuming the sealed test and from entering production. The M25 opportunity is real, but this model family does not capture it robustly enough.

No performance claim or production promotion is made from M26.
