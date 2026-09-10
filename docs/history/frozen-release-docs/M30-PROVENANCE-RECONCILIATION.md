# M30 provenance reconciliation

M30 consolidates the **verifiable source line actually present in the active artifacts**.

## Verifiable source lineage

The current source tree descends from the verified M13 runtime baseline and the M23–M29 planner/evidence hardening work. It contains the canonical LP/MILP/QP runtime, validation, diagnostics, session/reuse accounting, backend adapters, benchmark harness, production planner, and the research evidence guards added after M22.

## Historical M14–M22 implementation gap

Historical project records describe M14–M22 prototypes such as structural inspection extensions, exact specialized solvers/reductions, decomposition, and reoptimization-planner work. The corresponding source-tree bytes were not present when the later M23 source line was reconstructed in the active runtime. M22 **official benchmark evidence** is preserved, but benchmark evidence is not a substitute for missing implementation source.

Therefore M30 does not claim those historical prototype modules as shipped M30 functionality. Reintroducing any such feature requires recovering its original source or reimplementing it under fresh tests/provenance; it cannot be reconstructed from narrative summaries and silently called equivalent.

This is an explicit trust boundary, not a deletion claim about what may have existed in an earlier runtime.
