# Benchmarks

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
