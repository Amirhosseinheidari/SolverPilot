# INTERNET / LITERATURE AUDIT M28

Date: 2026-09-06

## Current HiGHS semantics

Official HiGHS development documentation generated 2026-09-02 lists `solver="choose"` as the default option and states in the solver overview that, for LP, `choose` selects the default solver, currently simplex. The same documentation exposes forced `simplex`, `ipm`, `ipx`, `hipo`, and other routes.

Sources:
- https://ergo-code.github.io/HiGHS/dev/solvers/
- https://ergo-code.github.io/HiGHS/dev/options/definitions/

This is **current HiGHS documentation**, not evidence about the old bundled 1.10.0 binary. M28 measured the 1.10.0 binary directly.

## HiGHS 1.10.0 historical context

The official GitHub release notes for v1.10.0 state that when a valid basis exists and solver option is `simplex` or `choose`, simplex is chosen. This is consistent with `choose` being strongly simplex-oriented in that release, but M28 does not infer full runtime behavior from the note; iteration diagnostics are the primary evidence.

Source:
- https://github.com/ERGO-Code/HiGHS/releases/tag/v1.10.0

## SciPy `method="highs"`

SciPy's public documentation historically describes `method="highs"` as choosing between `highs-ds` and `highs-ipm`. However, the SciPy source path used locally by M28 (1.17.0) maps `highs` to `solver=None`, and `_linprog_highs` documents that `None` uses simplex. M28 records source hashes in `m28-scipy-highs-semantics.json` and does not treat this route as an independent selector.

Current source reference:
- https://github.com/scipy/scipy/blob/main/scipy/optimize/_linprog.py

Local M28 probe:
- SciPy version: 1.17.0
- `linprog` source SHA-256: 341f316be763e183f9277409da609d6bdc88bed35b209a8e44f647a9b1f7fe7a
- `_linprog_highs` source SHA-256: 6bc682fdf9baef6f12260130d94900e8777dc6935fc89b92f3723e8c9031cfe4

## Research implication

M28's result is a negative baseline audit, not a criticism of HiGHS. A solver default is allowed to optimize robustness/general use rather than mimic a per-instance VBS oracle. The result simply means OptiMind cannot outsource its M25/M27 DS-vs-IPM opportunity to this `choose` route in the verified environment.
