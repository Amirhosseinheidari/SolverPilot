# Benchmarks

M1 includes lightweight executable verification scripts that require only the base dependencies.

```bash
PYTHONPATH=src python benchmarks/m1_property_checks.py
PYTHONPATH=src python benchmarks/m1_smoke.py
```

They are **not** a replacement for the planned Benchopt + MIPLIB/QPLIB release protocol.

Raw outputs are written under `benchmarks/results/`.
