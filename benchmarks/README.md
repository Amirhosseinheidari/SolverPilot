# Benchmarks

The [repeated-LP guide](../docs/REPEATED-LP-QUALIFICATION.md) covers opt-in
HiGHS reuse, split assurance costs, `qualify_highs_reuse.py` analytic transitions
and the separate preparation/execution stages of `qualify_public_reuse.py`.
`compare_lp_workflows.py` v2 can include `solverpilot-reuse` and presolve on/off;
its original four-method default remains available for historical comparisons.

The current [audit evaluation guide](../docs/AUDIT-REMEDIATION.md) documents
`challenge_production_evidence.py` and `compare_lp_workflows.py`, including
reproduction commands, matched native settings, retained warm states, explicit
failures and limits on interpreting synthetic observations. The separate
participant protocol requires real observed feedback; benchmark timings do not
measure human time saved.

M1 includes lightweight executable verification scripts that require only the base dependencies.

```bash
PYTHONPATH=src python benchmarks/m1_property_checks.py
PYTHONPATH=src python benchmarks/m1_smoke.py
```

They are **not** a replacement for the planned Benchopt + MIPLIB/QPLIB release protocol.

Raw outputs are written under `benchmarks/results/`.
