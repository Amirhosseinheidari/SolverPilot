# P7 STATUS — Nonlinear Optimization + Automatic Differentiation

**Package:** `optimind-core-codename 0.0.38`  
**Track:** P — Platform V2  
**Classification:** VERIFIED RELEASE  
**Date:** 2026-09-06

## Scope delivered

P7 extends the P0–P6 platform with a smooth continuous nonlinear layer:

- nonlinear semantic DAG atoms: `sin`, `cos`, `exp`, `log`, `sqrt`, `tanh`, nonnegative integer powers, symbolic division
- `NLPProblem` / `NLPConstraintBlock` execution IR
- continuous-variable nonlinear objective and nonlinear scalar-set constraints
- automatic differentiation through CasADi 3.7.2:
  - objective gradient
  - sparse constraint Jacobian
  - exact Lagrangian Hessian
  - Jacobian-vector and transposed-Jacobian-vector products
  - Jacobian/Hessian structural sparsity
- independent NumPy-space nonlinear primal/objective validation
- P3 capability-v2 keys for NLP and derivative surfaces
- verification-only CasADi 3.7.2 -> bundled Ipopt execution bridge
- recomputed KKT stationarity gate before a candidate is labeled `local_optimal_candidate`

## Guarantee boundary

P7 **does not claim global optimality for generic NLPs**.

A successful Ipopt termination is promoted only to a local-optimal candidate after:

1. a finite primal vector is returned,
2. bounds and nonlinear constraints pass independent validation,
3. the objective is independently recomputed and consistent,
4. KKT stationarity is recomputed from OptiMind's AD gradient/Jacobian plus solver multipliers and is within tolerance.

`globally_proven` remains `false` for the generic P7 backend.

## Correctness evidence

- final regression: **410 collected / 407 passed / 3 skipped / 0 failed / 0 errors**
- derivative campaign: **500/500**, including **200 Hessian** comparisons
  - max gradient abs error: `1.8627233089318906e-09`
  - max Jacobian abs error: `6.153939580144652e-10`
  - max Hessian abs error: `5.930544944021676e-11`
- analytic NLP campaign:
  - unconstrained: **120/120**, max x error `5.676425995915224e-11`
  - nonlinear constrained: **80/80**, max x error `3.4579382690580474e-09`
  - global claims emitted: **0**
- public-formulation smoke from official MINLPLib pages:
  - `rbrock`: reaches public bound 0 from the published start
  - `trigx`: reaches public bound near `0.09563139` from one start but a worse local point near `0.99214` from another
  - `mathopt4`: reaches bound 0 from `[0,0]` but also valid local points near `22.56` and `77.05` from other starts

The public smoke is explicitly classified as **official-formulation reproduction, not official archive execution**.

## Performance evidence

AD overhead characterization (not a solver-speed claim):

- n=10: derivative graph build median `0.00363 s`; eval grad/Jac/Hess median `0.000193 s`
- n=50: build `0.0223 s`; eval `0.000447 s`
- n=100: build `0.0831 s`; eval `0.00107 s`

## Wheel evidence

- wheel: `optimind_core_codename-0.0.38-py3-none-any.whl`
- wheel SHA-256: `4052cfd5fe17ee8b18738ae9ebabd0bba37a65bd4385dd12f226d18975faa10d`
- wheel members: `113`
- wheel `testzip()`: `None`
- NLP schema packaged: yes
- installed-distribution version: `0.0.38`
- out-of-source wheel smoke: PASS

## Deliberate boundaries

P7 does not implement:

- mixed-integer nonlinear optimization (P8)
- nonsmooth atoms such as generic `abs/max/min`
- global NLP proof
- dedicated `cyipopt` binding
- public CUTEst/MINLPLib archive parser/execution
- persistent NLP parameter patching; P7 performs full NLP relowering when data changes

## Result

P7 correctness, derivative, backend, guarantee, wheel-isolation and release-integrity gates are complete. P7 is a **VERIFIED RELEASE**.
