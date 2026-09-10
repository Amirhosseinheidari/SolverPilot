# Benchmark Report — M25 Public/OOD Selection Opportunity Audit

## Question

M24 falsified the old `m/n >= 1.5` selector on 24 public/OOD LP relaxations. M25 deliberately does **not** try another rule. It first asks whether there is enough SBS→VBS headroom to make selector research worthwhile after feature/inspection cost.

This follows standard algorithm-selection accounting: SBS is the mandatory baseline, VBS defines the available oracle gap, and feature computation cost must be charged before claiming selector improvement.

## Frozen protocol

- source: official MIPLIB 2017 benchmark-v2 archive already integrity-verified in M22
- problem form: continuous LP relaxations of official MIPLIB models
- cohort: first 48 eligible instances by deterministic SHA-256(filename) order
- resource filter: compressed source <=512 KiB, n<=5000, m<=5000, nnz<=100000
- backends: SciPy/HiGHS dual simplex vs SciPy/HiGHS IPM
- solver cutoff: 1.0 second
- repetitions: 2 per backend
- inspection cost: median `inspect_problem` time per instance, added to oracle cost
- execution: four deterministic shards; each solve in clean `python -S` process with controller hard timeout
- bootstrap: 20,000 paired resamples

The opportunity gate was frozen before outcomes were examined.

## Run 1

- 48 instances
- SBS = IPM, mean 0.116570 s
- VBS mean 0.103279 s
- VBS relative gain = **11.40%**
- oracle + inspection mean 0.104356 s
- oracle + inspection relative gain = **10.48%**
- oracle+inspection / SBS = 0.89522
- bootstrap 95% = **[0.81858, 0.94729]**
- DS wins 37 / IPM wins 11 / ties 0
- decisive non-SBS wins 26
- DS terminal rate 93.75%; IPM terminal rate 100%
- objective mismatches 0
- every pre-registered opportunity gate passed

## Run 2

- identical cohort/signature
- SBS = IPM, mean 0.117470 s
- VBS mean 0.104308 s
- VBS relative gain = **11.20%**
- oracle + inspection mean 0.105286 s
- oracle + inspection relative gain = **10.37%**
- oracle+inspection / SBS = 0.89628
- bootstrap 95% = **[0.81865, 0.94759]**
- DS wins 34 / IPM wins 13 / ties 1
- decisive non-SBS wins 28
- DS terminal rate 93.75%; IPM terminal rate 100%
- objective mismatches 0
- every pre-registered opportunity gate passed

## Stability interpretation

Five winner labels changed between repeated timing runs. This does not invalidate the opportunity: both runs retain roughly 10% oracle-with-feature headroom and the bootstrap upper bound remains well below 1.

The label instability does affect the **next selector design**. A future selector should optimize cost/regret rather than raw winner-classification accuracy. Several large performance gaps matter much more than small timing flips around ties.

## Important boundary

M25 is an opportunity audit, not a trained selector benchmark. It does not authorize DS/IPM auto-routing, does not switch the baseline, and makes no claim that the 10% oracle headroom is achievable by a learned policy.
