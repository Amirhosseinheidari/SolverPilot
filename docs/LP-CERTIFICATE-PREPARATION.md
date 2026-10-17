# LP certificate preparation after 0.4.1

This change reduces validation work without changing feasibility/objective
tolerances, recovery caps, the QP certificate formula, or proof categories.
It is a follow-up to the [0.4.1 certificate repair](LP-CERTIFICATE-PERFORMANCE.md).

## Exact endpoint comparison

Implied-box propagation previously converted every candidate endpoint to a
`Fraction` and then rounded it outward, even when it could not tighten the
current box. A profile over one frozen candidate from each of eight public
models identified 104,768 calls to this conversion.

Write a candidate as `N/D`, normalize `D > 0`, and represent the current finite
binary64 endpoint exactly as `B / 2**e`. The sign of

```
(N << e) - B*D
```

is the sign of the difference between the candidate and the current endpoint.
For an upper endpoint, a nonnegative difference cannot improve the box after
upward rounding. For a lower endpoint, a nonpositive difference cannot improve
it after downward rounding. Such candidates can be skipped exactly. Equality,
negative values, and subnormal values use the same integer comparison.
Potential improvements and infinite current endpoints still use the original
outward conversion, including its conservative overflow behavior. Work counters
and row/visit ordering are unchanged. The same diagnostic profile then performed
8,040 conversions; profiler timings are not used as performance evidence.

## Preparation reuse

When an explicit `VerificationWorkspace` is available, propagation reuses its
content-hashed, immutable matrix coefficients. Current row endpoints, variable
endpoints, and all derived bounds remain call-local. Changed matrix values have
a different cache key; changed bounds are read afresh even for the same matrix.
Limited calls retain lazy preparation, preserving small work/time budgets and
avoiding eager conversion of unvisited coefficients.

For LP numerical stationarity and complementarity, products involving `[A; I]`
are evaluated directly from `A` and the bound multipliers. This removes repeated
sparse identity, stacking, and transpose preparation. Every call still checks
the current primal and dual vectors. No accepted proof is reused.

## Independent correctness checks

`tests/test_lp_certificate_exact_oracle.py` enumerates vertices of tiny bounded
polytopes using a separate Fraction Gaussian-elimination implementation. It
does not call a numerical solver or the production certificate arithmetic to
obtain reference optima. Reported lower bounds are compared exactly with those
optima (upper bounds for maximization), including fractional optima, dependent
and nearly dependent equations, changed endpoints, objective offsets, and
matrix/row/column transformations.

Arbitrary row multipliers on changed bounded polygons test conservative bounds
even when the candidate is not certified. Separate contradictory equalities
and an explicit unbounded direction test rejection after workspace reuse.
Corrupted candidates, nonfinite duals, small budgets, eviction, and signed/extreme
binary64 endpoints are also covered. These finite tests complement the exact
endpoint argument; they are not a generic formal proof or an exact feasible
rational-primal claim for numerical solutions.

The earlier independent Fraction propagation reference and the frozen 256-vector
public replay remain in use. Repeated/transformed observations from eight public
models are not 256 independent problem families.
