# M4 Public Dataset Acquisition Status

Date: 2026-08-31

M4 does **not** bundle MIPLIB or QPLIB instances. The current runtime cannot download binary/compressed benchmark archives from the external file hosts, so no public-corpus solve is claimed.

## MIPLIB 2017

Target: Benchmark Set v2 (240 instances).

Official sources:
- https://miplib.zib.de/set_benchmark.html
- https://miplib.zib.de/download

Required future procedure:
1. download from the official MIPLIB source in a network-enabled environment;
2. retain the exact archive filename and checksum;
3. unpack outside the Python package;
4. generate an immutable instance manifest with SHA-256 per instance;
5. record any excluded instance and the exclusion reason;
6. never tune the production selector on the frozen final test partition.

## QPLIB

Target for v0.x: continuous convex QP subset only. Do not describe all QPLIB instances as supported.

Official sources:
- https://qplib.zib.de/
- https://qplib.zib.de/statistics.html

Required future procedure:
1. use QPLIB metadata to identify continuous convex instances;
2. record the metadata snapshot/version date;
3. preserve original instance checksums;
4. validate parser/objective semantics on a hand-checked subset before timing comparisons.

## Current M4 substitute

`m4_lp_selector_holdout.py` is a controlled synthetic structural benchmark. It exists to answer whether the DS/IPM development portfolio has enough complementarity to justify selector research. It is **not** a substitute for MIPLIB/Netlib/QPLIB and must never be cited as a public benchmark result.
