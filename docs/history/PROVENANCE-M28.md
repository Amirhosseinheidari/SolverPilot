# PROVENANCE M28

M28 starts from the verified M27 source tree (`0.0.32`) and produces source version `0.0.33`.

Primary external data:
- MIPLIB archive: `/mnt/data/benchmark.zip` during execution; SHA-256 `c756eefd544d83b31809306b45d3549a1a5b9378e6aa78b68738b1a3b6a418fa`.
- Official 240-name benchmark-v2 manifest SHA-256 `0ee9758dba64bb1689cc121c794ee547992a046bcabe1cce9ea79158375dabd5`.
- M22 transfer-integrity evidence is hash-locked inside each M28 shard.

Fresh-cohort controls:
- M25 prior cohort file SHA-256 `8ff4ea9d488978ecdfd11f1bc713385399959ade1739f98cfd0e7fb2d0e52c78`.
- Fresh cohort signature `800c50ea8bb68c87283902d34fb6e0677a730f2fec7ca423b56643ff11ab3327`.
- Overlap with M25 selected instances: 0.

Execution provenance embedded directly in raw shard files includes full environment payload, timing-environment fingerprint, fixed-host fingerprint, protocol SHA-256, cohort SHA-256, runner SHA-256, worker SHA-256, and M22 integrity-file SHA-256.

No current-public highspy binary was used for M28 timing. The primary solver core is CasADi 3.7.2's verified bundled HiGHS 1.10.0 library.
