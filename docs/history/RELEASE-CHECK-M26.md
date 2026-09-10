# Release Check — M26 0.0.31

**Classification:** COMPLETE / VERIFIED NEGATIVE-RESULT MILESTONE — selector rejected, test preserved.

## Scientific gates

- exact M25 48-instance cohort: PASS
- deterministic group-safe 20/12/16 split: PASS
- train/validation artifact contains zero test costs/names: PASS
- pre-registered low-capacity ridge family / 9-candidate grid: PASS
- feature/inspection cost included: PASS
- validation evaluation complete: PASS
- validation promotion gate: **FAIL** (`policy/SBS = 1.1886926884`, required <= 0.98)
- all 9 candidate configurations beat SBS: **NO; all 9 are worse than SBS**
- final held-out test opened: **NO (correct fail-closed behavior)**
- production performance ranking remains disabled: PASS

## Software release gates

- full pytest: **243 collected / 240 passed / 3 skipped / 0 failed / 0 errors**
- compile/import source: PASS
- offline wheel build 0.0.31: PASS
- wheel ZIP integrity / metadata version: PASS
- isolated wheel import outside source tree: PASS
- installed-wheel M26 evidence parse: PASS
- installed-wheel M26 performance authority: correctly FALSE
- installed-wheel manual IPM override: correctly REJECTED
- installed-wheel conservative LP baseline: `scipy-highs-ds`
