# Release Check — M27 0.0.32

**Classification:** COMPLETE / VERIFIED NEGATIVE HELD-OUT RESULT — selector rejected; test consumed.

## Scientific gates

- public/OOD opportunity already established by M25: PASS
- M26 test remained sealed before M27 freeze: PASS
- M27 protocol frozen before test: PASS
- M27 model frozen before test: PASS
- development evaluation uses only 32 development rows: PASS
- development gate: PASS (`policy/SBS = 0.9801884`)
- one-time exact 16-instance held-out execution: PASS
- exact 64 DS/IPM outcomes: PASS
- worker errors: 0
- hard timeouts: 0
- solver-level error outcomes: 2, conservatively cutoff-charged
- objective mismatches: 0
- held-out policy/SBS <= 0.97: **FAIL** (`1.0077011`)
- held-out bootstrap 95% upper < 1: **FAIL** (`1.02718`)
- tail safety p90 <= 1.25: PASS
- harmful switch <= 0.10 s: PASS
- selector production promotion: **NO**
- post-test retuning: **NO**

## Software gates

- version: 0.0.32
- full regression: **252 / 249 passed / 3 skipped / 0 failures / 0 errors**
- offline wheel build: PASS
- wheel ZIP integrity: PASS (`testzip() = None`)
- wheel metadata version 0.0.32: PASS
- final wheel SHA-256: `92f03259c5ba21c35a57278cee2bd7c6f4832dd6c67654a9440df31de5cb0505`
- isolated import outside source tree: PASS
- installed-wheel M27 evidence parse: PASS
- installed-wheel performance authority: correctly FALSE
- explicit fixed-environment authority from M27 payload: correctly FALSE (test-time fingerprint was not embedded)
- installed-wheel manual speed override: correctly REJECTED
- installed-wheel LP validation: PASS

## Claims boundary

M27 verifies a **negative** selector result. It does not establish that learned LP routing improves production performance. The M27 test is permanently consumed and cannot be used as the held-out gate for another selector.
