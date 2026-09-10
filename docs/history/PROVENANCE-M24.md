# Provenance — M24

- Baseline: verified M23 package 0.0.28.
- M24 package version: 0.0.29.
- Public/OOD source: reconstructed official MIPLIB 2017 `benchmark.zip` from the M22 Drive transfer.
- Archive SHA-256 used by M24: `c756eefd544d83b31809306b45d3549a1a5b9378e6aa78b68738b1a3b6a418fa`.
- M24 independently checks that this digest matches the M22 transfer-integrity record before marking `official_corpus_integrity_passed=true`.
- Membership order comes from `benchmark-v2.test` retained from the M22 official transfer.
- M24 does not modify, relabel, or claim ownership of MIPLIB instances.
- The benchmark transforms selected canonical MILPs into continuous LP relaxations by relaxing variable domains only; objective, matrix, row bounds, variable bounds, sense, and offset are preserved.
- The M4/M5 threshold rule was fixed before M24 and is not tuned on MIPLIB data.
