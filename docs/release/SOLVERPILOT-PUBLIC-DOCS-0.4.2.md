# SolverPilot 0.4.2

This patch release includes PR #29's installed-artifact CI repair and lower LP
certificate preparation cost. It preserves the complete 0.4.1 public signatures,
backend IDs, validation tolerances and proof categories. The 0.4.1 API/backend
snapshots and historical measurement artifacts remain unchanged.

## Install

```bash
python -m pip install "solverpilot==0.4.2"
python -m pip install "solverpilot[highs,osqp]==0.4.2"
python -m pip check
python -m solverpilot.cli.doctor --compact
```

Python 3.12-3.14 is supported. Dependencies and optional integration requirements
are unchanged from [0.4.1](SOLVERPILOT-PUBLIC-DOCS-0.4.1.md).

## Changes and evidence

Installed-wheel CI previously allowed native subprocess tests to insert the
checkout's implementation. The nine regression cells and release cells now
check installed package origins and file hashes against the tested wheel, with
receipts for parent and same-environment child processes, including isolated
Python. See [the guard scope](../INSTALLED-ARTIFACT-TESTS.md).

LP propagation skips outward conversions only after an exact integer comparison
proves that the endpoint cannot tighten the current box. Eligible calls reuse
immutable matrix coefficients; current bounds and candidate checks remain fresh.
LP residual products no longer construct a sparse identity stack. QP certificate
formulas and cooperative proof budgets are unchanged. See
[the arithmetic argument and independent tests](../LP-CERTIFICATE-PREPARATION.md).

The development comparison retained all 256 frozen numerical proofs, including
the 68 earlier repairs. A separate rational vertex oracle checks tiny bounded
LPs, changed matrices/bounds, arbitrary multipliers and adversarial endpoints.
These tests complement the arithmetic argument; they are not a generic formal
proof or an exact rational primal-feasibility claim.

Two alternating verifier-only diagnostics on eight public model families showed
about 26% less verifier time without a workspace and 28-30% with one. Those
diagnostics loaded old/new verifier code with unchanged shared helpers; they
are separate from installed-package qualification. Individual cases fluctuate.
The primary installed-wheel end-to-end campaign was mixed, with considerable
host variation in unchanged direct-HiGHS controls. No overall solve speedup or
statistical significance is claimed. Related transformations and repetitions
are not independent problem families. Development artifacts reported version
0.4.1; their hashes distinguish them from the released 0.4.1 and new 0.4.2 files.

## Guarantee and publication scope

The [known limitations](../../KNOWN-LIMITATIONS.md) remain applicable. Learned
routing is experimental and disabled automatically. Native model/basis reuse
does not guarantee factorization reuse. Exact SCIP/VIPR and GPU execution require
separate executable/hardware qualification.

Publication uses the exact wheel/sdist pair from a successful attested
qualification of the final main commit. TestPyPI installation checks precede
PyPI publication. Fresh registry wheel/sdist installations check hashes,
import ownership, dependencies, independent examples and the 256-vector replay.
The GitHub release records the actual outcomes and artifact identities.
