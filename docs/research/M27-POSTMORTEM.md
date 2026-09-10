# M27 Postmortem — Selective Absolute-Saving Tree

## Final result

M27 is a valid negative selector milestone. The frozen selector passed development OOF gates but failed the one-time held-out promotion gate:

- held-out policy/SBS = **1.0077011**
- paired bootstrap 95% = **[0.99643, 1.02718]**
- required policy/SBS <= 0.97: FAIL
- required bootstrap upper < 1: FAIL

Correctness and accounting remained clean: 64/64 outcomes, zero worker errors, zero objective mismatches. Two solver-level `error` outcomes occurred for DS on `cvs16r128-89`; they were conservatively charged at the cutoff and are not hidden as worker failures.

## Why the selector failed

The model switched only twice:

- `neos-4954672-berkel`: useful DS switch, solver saving about **+5.325 ms** before inspection.
- `uct-subprob`: harmful DS switch, solver regret about **15.464 ms** before inspection.

Net solver effect of the selected switches was negative before paying feature cost.

The support guard was useful but blunt. Four additional test points reached the positive-saving leaf but were rejected as extrapolation. Among them were both profitable and dangerous DS cases. So the guard reduced tail risk but could not distinguish safe from unsafe switch regions.

The five structural features were also insufficient globally. Several instances with substantial DS advantage landed in a development leaf whose average target strongly favored IPM, including `app1-1`, `neos-1171737`, and `roll3000`. This indicates representation/generalization failure rather than a simple threshold miss.

## Provenance hardening found during final audit

The consumed M27 result did not embed a full test-time environment fingerprint or a cryptographic runner hash. Post-hoc filesystem chronology and frozen artifact hashes are consistent with the recorded sequence, and a post-hoc environment capture using the M25 fingerprint schema matches M25. However, this cannot retroactively prove the exact test-time environment.

Therefore the production evidence translator is intentionally conservative:

- `fixed_environment = False` unless explicitly carried by the test payload;
- `corpus_integrity_passed = False` unless explicitly carried by the test payload;
- model self-hash is recomputed before held-out integrity can be true.

These downgrades do not change the scientific negative result. They make accidental future promotion even less likely.

## Irreversible boundary

The original 16 M26/M27 test instances are consumed. No future selector may use them for promotion. A new selector requires a new untouched public/OOD test cohort.
