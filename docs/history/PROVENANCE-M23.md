# Provenance — M23

## Code baseline

The executable runtime source available in this session is the verified M13 source artifact (`optimind_m13.zip`, package 0.0.14). The raw source trees for M14–M22 were not persisted into this runtime. M23 therefore adds the production evidence-aware planning layer to that stable runtime core and consumes the separately preserved official M22 evidence bundle.

This release does **not** claim to reconstruct missing M14–M22 source modules. M22 official benchmark evidence remains independent and is bundled with the final benchmark archive.

## M22 evidence consumed

- `M22-FINALIZATION-GATE.json`
- `M22-TRANSFER-INTEGRITY.json`
- `M22-MIPLIB-PHASE.json`
- `M22-QPLIB-PHASE.json`
- `M22-PACE-PHASE.json`
- `M22-REFERENCE-CROSSCHECK.json`

## New M23 code

- `src/optimind/plan/production.py`
- `solve_production` in runtime auto layer
- M22 evidence parser
- M23 production planner tests
- `benchmarks/m23_production_planner_campaign.py`

## Version

0.0.28
