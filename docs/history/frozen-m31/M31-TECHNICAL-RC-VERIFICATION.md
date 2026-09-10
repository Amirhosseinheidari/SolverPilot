# M31 Technical RC Verification

**Classification:** VERIFIED TECHNICAL RC CHECKPOINT — NOT PUBLIC RC / NOT 1.0

- version: `0.0.36rc1`
- full regression: 291 collected / 288 passed / 3 skipped / 0 failed / 0 errors
- Linux / CPython 3.13 wheel smoke: PASS
- Linux / CPython 3.13 NLopt 2.11 optional backend: PASS
- Linux / CPython 3.13 CasADi 3.7.2 verification bridge: PASS
- Python 3.10 grammar audit: PASS (syntax-only)
- sdist -> wheel -> isolated install: PASS
- wheel metadata/content audit: PASS
- Windows-style clean-worker path bug: FIXED + regression test
- compatibility cells not executed in this runtime remain pending, not pass
- public 1.0 compatibility gate: CLOSED

Wheel SHA-256: `015fbc33ff1a9a6619248d32ec5c2c51c858ac9aac145a6384c96fea768dec09`

sdist SHA-256: `1ac682581592178995f001bfccd27d12db206b18971cad08aaf2ea9652139a39`
