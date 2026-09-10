# PROVENANCE P9 — Real P8 Rebase

P9 initially existed as an M26-based recovery/delta build because the P8 executable archive was absent after a runtime reset. On 2026-09-07 the exact P8 verified archive was supplied by the user and verified:

- P8 archive: `optimind_track_p_p8_minlp_orchestration_0.0.39_verified.zip`
- SHA-256: `dd53c346d49c20b3a7b4b4884bdeb7b9ade4520be09ea61bbfe0090530c9822e`
- ZIP integrity: PASS

The locked P9 delta was taken from the previously produced baseline recovery bundle:

- P9 delta SHA-256: `8b8c2fd52a0831dc49dbb757884e72d9296f80d57f9b4653d231ce17c60beb05`
- M26 base SHA-256: `783c1075355ab3556e7c31b1db113096e956ed54705c7df0b44d6967c10d5391`

Three-way hash preflight reported 16 clean additions and 4 conflicts. The conflicts were manually resolved by retaining P8 as the executable parent and applying only the P9 intent necessary for version/CP integration. No P8 model, capability, bridge, session, conic, NLP or MINLP surface was replaced by the M26-based recovery versions.

This source tree is therefore a real executable successor of P8 with the P9 CP delta integrated.
