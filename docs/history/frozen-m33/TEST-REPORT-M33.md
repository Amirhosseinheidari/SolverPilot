# Test Report M33

Final frozen-tree regression:

- collected: **316**
- passed: **312**
- skipped: **4**
- failures: **0**
- errors: **0**

The skips remain explicit optional/external-evidence gaps; they are not relabeled as passes.

Additional local release evidence:

- exact wheel LP/MILP/QP/production smoke: PASS
- NLopt optional backend: PASS on Linux/Python 3.13
- CasADi verification bridge: PASS on Linux/Python 3.13
- sdist rebuild/install/smoke: PASS
- local resolver report: runtime versions recorded; `pip check` not clean because the host contains an unrelated `moviepy 2.2.1` / `Pillow 12.3.0` conflict
- Python 3.12/3.14 runtime materialization: blocked by DNS
- local pip-audit/Twine materialization: blocked by DNS

Compatibility promotion therefore remains external-CI-only under the frozen M33 protocol.
