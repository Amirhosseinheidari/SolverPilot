# Release Check — M4

Date: 2026-08-31

This is a research milestone archive, **not a public release candidate**.

## Required checks performed

- [x] source reconstructed from the original full M3 ZIP, not the incomplete loose report directory;
- [x] package version advanced to 0.0.5;
- [x] editable offline install works with local build tooling;
- [x] full pytest suite passes except explicit missing-optional-package skips;
- [x] native wheel lock has official file-host URLs and SHA-256 digests;
- [x] direct wheel retrieval failure is recorded as environment limitation, not package absence;
- [x] M4 selector benchmark executed twice unchanged;
- [x] raw results from both runs retained;
- [x] M1/M2 property checks re-run;
- [x] M3 native reuse benchmark re-run;
- [x] production planner remains capability-only;
- [x] no MIPLIB/QPLIB run is claimed;
- [x] cache/build byproducts removed from final archive.

## Public-release blockers

- public package name unresolved;
- exact public highspy/OSQP/PySCIPOpt integration tests still skipped;
- MIPLIB/QPLIB not executed;
- cross-solver selector not validated;
- no frozen public train/test benchmark manifest;
- no CI matrix across supported Python/OS combinations;
- no package publication or security review.
