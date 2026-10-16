# Public LP routing admission and actual-call gates

> Release inclusion: the implementation described here is included in 0.4.1.
> The original development measurements and artifact versions below are historical
> evidence, not measurements of the new 0.4.1 distribution.

Development after published 0.4. These changes make a failed or insufficient study
explicit; they do not enable automatic learned routing or publish a package.

## Repairs

* Public and development manifests have different schemas. Public runs require
  frozen consumed-identity registers and disjoint training/calibration/test sets.
* Netlib admission and pool inventory share conservative family aliases, including
  PEROLD/PILOT, BP/25FV47, POWELL/FFFFF800, GUB/CZPROB and the prior forest,
  staircase and MAROS groups. WOOD variants are excluded together conservatively;
  this is an inference, not a claim that Netlib proves they are identical.
* Public preflight needs at least four training families, four fresh calibration
  families and twenty fresh test families, before any solver call. Repeats must
  be even. Inputs must be continuous MPS with exact native-reader agreement and
  declared source and objective-reference URLs. Integer relaxations must first
  be materialized and hashed as continuous models.
* No qualified learned leaf, an all-single-solver policy or failed session setup
  stops the public run before **any** test strategy executes. Development mode
  can still exercise fallback deliberately. Consumed history and artifacts are
  checked again before execution.
* The runner checks loaded package/module locations against the frozen checkout
  and hashes all benchmark helpers along with package sources. Run in a fresh
  process with `PYTHONPATH=src`; a stale installed package is rejected.
* Public reports score measured `solve_robust_lp` calls, including availability,
  decision, validation and return overhead. Shadow predictions remain development
  evidence. Setup is charged fully for cold calls and at the frozen reuse count
  for amortized calls, before applying the deadline and PAR10 penalty.

## Acceptance

Every test repeat is retained. Reports compare against actual default and
production calls and the best single solver frozen from training. Each family
has equal weight, with a 2,000-draw family bootstrap (seed 271828). For each cost
scope all three comparisons must show at least 3% mean improvement, an upper
95% bootstrap ratio below one, p90 instance slowdown at most 25%, and no loss of
verified within-budget repeats. A conflicting verified objective/reference is a
global failure even if both routes receive identical penalties.

At least four test families must have successful in-support learned switches
**away from the frozen single solver**, with matching model/guard identities.
Fallback calls and constant single-solver selection do not satisfy this gate.
Passing means scoped research qualification; `production_authorized` remains
false. Public datasets and caller-declared family/reference provenance do not
by themselves demonstrate operating-site fitness, security authenticity or
exhaustive semantic independence. Native limits remain cooperative.

## Current admission evidence

The October 13 admission-only audit checks the recorded prior registers and
downloaded models before considering unused Netlib families. It finds four
remaining families within the existing structural caps: ETAMACRO, NESM, RECIPE
and SHELL. Without those caps, conservative grouping yields seven, still fewer
than the 24 needed for separate calibration and test. This is a finding about
this recorded Netlib pool, not about every public optimization dataset.

The audit performs **zero solves** and preserves the remaining holdout inputs.
The complete register identities, metadata/source hashes, exclusions and limits
are in [the pool audit](evidence/lp-v2-public-gate/pool-audit.json). The source for
documented aliases is the [official Netlib README](https://www.netlib.org/lp/data/readme).
The next qualification needs additional independent public families or external
operating models with audited references. Do not relabel consumed examples,
related variants or new random seeds as fresh families.

The next public source to audit is the [MIPLIB 2017 benchmark table](https://miplib.zib.de/tag_benchmark.html)
and its [individual/archive downloads](https://miplib.zib.de/download.html).
Its 240 instances have repeated or missing group labels, so that count is not
240 independent families. Continuous relaxations need separate materialized
files and LP objective references; the published integer optima are not valid
LP references. Larger structural caps require a new resource budget and protocol,
not a retrospective change to this audit's acceptance conditions.

```powershell
$env:PYTHONPATH='src'
python benchmarks/audit_lp_v2_pool.py --workspace D:/ARTICLE/solverpilot --metadata D:/ARTICLE/solverpilot/ordered-public-metadata/netlib-readme --output NEW-AUDIT.json
python benchmarks/qualify_lp_v2.py --manifest PUBLIC-MANIFEST.json --output NEW-RUN --max-calls 360
```

Public manifests use `solverpilot.lp-v2-public-manifest.v1`, `scope: public`,
`consumed_lists`, and the existing explicit `train/calibration/test` case arrays.
Each case adds `source: netlib|external`, `source_url`, `reference_url`, and a
finite `reference` to the development format. Netlib family labels use
`netlib:` plus the shared conservative alias. Files/URLs are provenance data,
never instructions or permission to execute code. The runner does not fetch
arbitrary case URLs. Output directories are new and historical evidence remains
unchanged.
