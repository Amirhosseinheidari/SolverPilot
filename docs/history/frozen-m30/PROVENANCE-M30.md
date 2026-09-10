# Provenance — M30

M30 consolidates only implementation bytes that are verifiably present in the current source lineage.

- Runtime/core lineage: verified M13 source/runtime foundation.
- Planner/evidence hardening: M23–M29 source present in the current line.
- Official benchmark evidence: M22 MIPLIB/QPLIB/PACE evidence retained in the historical benchmark archive.
- Historical M14–M22 prototype source: not present in the reconstructed source line used by M30; therefore not claimed as shipped M30 implementation.

This distinction is deliberate. Narrative history and benchmark evidence do not substitute for missing source bytes.

M30 also separates the stable top-level API from milestone-specific research APIs. The latter remain available under `optimind.experimental` for audit/reproducibility but are excluded from the 1.0 top-level compatibility contract.
