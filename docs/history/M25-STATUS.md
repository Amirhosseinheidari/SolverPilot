# M25 Status — Public/OOD Selection Opportunity Audit

**Version:** 0.0.30  
**State:** COMPLETE / VERIFIED

M25 asks the question that must be answered before building another selector: does a sufficiently large, stable SBS→VBS opportunity exist on public/OOD structure **after paying the current OptiMind inspection cost**?

## Pre-registered gate

Before observing M25 outcomes, the opportunity gate was frozen:

- 48 deterministic MIPLIB-2017-derived continuous LP relaxations,
- two rounds per backend,
- DS vs IPM under the same 1.0s solver cutoff,
- official MIPLIB corpus integrity required,
- terminal rate >= 90% for both routes,
- zero objective mismatches,
- VBS relative gain >= 3%,
- oracle+inspection relative gain >= 2%,
- bootstrap 95% upper bound for oracle+inspection/SBS < 1,
- >=3 non-SBS wins and >=2 decisive non-SBS wins.

The final evidence was executed in four deterministic process-isolated shards. Clean `python -S` workers were required because normal Python startup in this environment includes timing-distorting site hooks. Sharding changed only execution scheduling, not cohort membership or acceptance criteria.

## Result

**The opportunity gate passed in two unchanged runs.**

Run 1:
- SBS: `scipy-highs-ipm`
- SBS mean: 0.116570 s
- VBS mean: 0.103279 s
- VBS relative gain: **11.40%**
- oracle+inspection mean: 0.104356 s
- oracle+inspection relative gain: **10.48%**
- oracle+inspection/SBS bootstrap 95%: **[0.8186, 0.9473]**
- DS wins 37, IPM wins 11, ties 0
- objective mismatches 0

Run 2:
- SBS: `scipy-highs-ipm`
- SBS mean: 0.117470 s
- VBS mean: 0.104308 s
- VBS relative gain: **11.20%**
- oracle+inspection mean: 0.105286 s
- oracle+inspection relative gain: **10.37%**
- oracle+inspection/SBS bootstrap 95%: **[0.8186, 0.9476]**
- DS wins 34, IPM wins 13, ties 1
- objective mismatches 0

The same 48-instance cohort and cohort signature were used in both runs. Five near-boundary winner labels changed between repeats, but the aggregate opportunity gate passed both times.

## Product consequence

M25 does **not** enable LP performance routing and does **not** switch the production baseline. It establishes only that the broader public/OOD regime contains enough value to justify building a selector.

Production evidence now has a separate `selection_opportunity_validated` gate. Future performance ranking requires both:

1. validated selection opportunity, and
2. a separately pre-registered, held-out selector that passes its own deployment promotion gate.

M25 satisfies (1), not (2). Therefore `supports_performance_ranking` remains `False` and the current production LP baseline remains `scipy-highs-ds`.

## Verification

- full regression: **234 / 234 passed**, 0 skipped, 0 failed, 0 errors
- two complete M25 opportunity runs: PASS
- process-isolated clean workers: PASS
- official MIPLIB integrity inherited/rechecked against M22 evidence: PASS
- wheel 0.0.30 offline build: PASS
- isolated wheel import: PASS
- real M25 evidence parsed from installed wheel: `selection_opportunity_validated=True`
- production ranking from M25-only evidence: correctly remains disabled
