# M33 Status — External CI Execution & Publication Decision

**Version:** `0.0.36rc3`  
**Milestone classification:** **VERIFIED ENGINEERING CLOSURE — PUBLIC RC FAIL-CLOSED**

M33 freezes a forward-looking public-1.0 support target of CPython 3.12/3.13/3.14 on Linux/macOS/Windows, raises the scientific runtime floors to NumPy >=2.2 and SciPy >=1.15, and hardens exact-artifact CI/security/release governance. It does not reopen learned LP routing.

## Verified locally

- full regression: **316 collected / 312 passed / 4 skipped / 0 failures / 0 errors**;
- M30–M33 release-contract suite: **49/49 passed** before final full regression;
- exact rc3 wheel LP/MILP/QP/production smoke outside the source tree: PASS;
- NLopt 2.11 and CasADi 3.7.2 optional-backend smoke on Linux/Python 3.13: PASS;
- sdist -> wheel -> isolated-target smoke: PASS;
- distribution metadata/integrity checks: PASS;
- static workflow supply-chain design: PASS (all third-party Actions full-SHA pinned; build-once/test-many; attestation generate+verify; Dependabot; disabled OIDC publish template).

## Not promoted to compatibility pass

The 9 public core cells (3 OS x Python 3.12/3.13/3.14) remain `pending_external_execution`. The local Linux/Python 3.13 smoke is deliberately non-promoting because the available host-backed dependency environment is not clean: `pip check` reports an unrelated `moviepy`/`Pillow` conflict. Python 3.12/3.14 runtime downloads, `pip-audit`, and `twine` could not be materialized locally because DNS/network access failed.

The minimum-dependency job (Python 3.12 + NumPy 2.2.0 + SciPy 1.15.0), runtime vulnerability/CycloneDX gate, GitHub-hosted attestation issuance/verification, repository security settings, and owner publication metadata therefore remain external gates.

## Publication decision

- `public_rc_authorized = false`
- `public_1_0_authorized = false`
- publish workflow remains intentionally disabled

M33 may close as a verified engineering milestone because its purpose is to define and harden the external release gate; it does **not** convert missing external evidence into a pass.
