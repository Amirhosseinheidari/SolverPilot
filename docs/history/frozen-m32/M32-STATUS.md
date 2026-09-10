# M32 Status — External Compatibility & Publication Hardening

**Version:** `0.0.36rc2`  
**Classification:** **VERIFIED SUPPLY-CHAIN / EXTERNAL-COMPATIBILITY HARDENING CHECKPOINT — PUBLIC RC FAIL-CLOSED**

M32 hardens the release process around the M31 frozen API. It does not add solver-selection claims and it does not convert unexecuted OS/Python cells into compatibility passes.

## What M32 verified locally

- M31→M32 public API: unchanged, 78 frozen top-level symbols.
- Backend IDs/policy: unchanged; learned LP performance routing remains off.
- Full source regression: **303 collected / 299 passed / 4 skipped / 0 failures / 0 errors**.
- Exact `0.0.36rc2` wheel installed outside the source tree on Linux / CPython 3.13.5.
- LP: validated optimal.
- MILP: validated optimal.
- QP: independently valid feasible result on the non-certifying SLSQP route.
- Production LP planner: `scipy-highs-ds`, performance ranking off.
- NLopt 2.11.0 optional backend: healthy on the exact wheel.
- CasADi 3.7.2 verification bridge: healthy on the exact wheel.
- sdist → rebuilt wheel → isolated install → solve smoke: pass.
- Rebuilt sdist wheel has identical member hashes to the original wheel except `RECORD`; byte-for-byte reproducibility is not claimed.
- Dist manifest/rehash/ZIP integrity: pass.

## Important local non-pass

The host-level `pip check` fails because the container already contains an unrelated `moviepy 2.2.1` / `Pillow 12.3.0` conflict. OptiMind exact-wheel solve/validation succeeds, but the pre-registered M32 core-cell gate requires dependency consistency in a clean environment. Therefore Linux/Python 3.13 is **not promoted to `executed_pass` in the external compatibility matrix** from this contaminated host.

`twine` is not installed in the local runtime, so local `twine check` is recorded as **not executed**, not pass. The GitHub Actions build job requires `twine check` before the matrix can run.

## External compatibility matrix

The intended core matrix remains 12 cells:

- Linux × Python 3.10–3.13
- macOS × Python 3.10–3.13
- Windows × Python 3.10–3.13

All 12 remain `pending_external_execution` until the exact built wheel is exercised in clean CI. Upstream wheel availability and Python-3.10 syntax parsing do not satisfy this gate.

## Supply-chain hardening added in M32

- build wheel/sdist once, then test the exact downloaded artifacts in every matrix job;
- SHA-256 distribution manifest + rehash before install;
- resolver version report for Python/pip/NumPy/SciPy/optional solver bindings;
- mandatory `pip check` in clean CI;
- full JUnit regression in every core cell;
- all external GitHub Actions pinned to full 40-character commit SHAs;
- build provenance attestation prepared;
- no active PyPI publishing workflow or long-lived PyPI token;
- publication remains blocked until owner metadata and Trusted Publisher identity are defined.

## Public release decision

`public_rc_authorized = false`  
`public_1_0_authorized = false`

M32 itself is complete as a **fail-closed hardening milestone**. A public RC requires actual external workflow execution plus owner publication metadata.
