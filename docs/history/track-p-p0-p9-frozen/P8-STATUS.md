# P8 STATUS — MINLP Orchestration

**Package:** `optimind-core-codename 0.0.39`  
**Track:** P — Platform V2  
**Classification:** VERIFIED RELEASE  
**Date:** 2026-09-06

## Scope delivered

P8 adds a proof-aware convex binary MINLP orchestration layer on top of P3 MILP capabilities and P7 NLP/AD:

- `MINLPProblem` execution IR and packaged schema
- binary/integrality-aware original-space validator
- structural convexity certificate for the P8 v1 supported atom set
- exact binary-assignment enumeration oracle for certified convex fixed-binary NLP subproblems
- Outer Approximation (OA) with continuous-relaxation initialization, exact gradient cuts, binary no-good cuts, explicit lower/upper-bound accounting, cycling protection and fail-closed unresolved NLP handling
- P3 capability key `problem.minlp` and MINLP requirements classification
- explicit rejection of general integer MINLP, nonconvex/bilinear models, nonlinear equalities/two-sided nonlinear constraints and indicator+MINLP composition in P8 v1

## Guarantee boundary

Global proof is emitted only for the certified-convex, binary-discrete P8 v1 scope and only after original-space MINLP validation plus bound closure. A successful Ipopt fixed-NLP status by itself never implies global MINLP optimality. If an NLP assignment is unresolved, P8 does not add an infeasibility no-good cut and downgrades the overall result.

## Final correctness evidence

- full regression: **430 collected / 427 passed / 3 skipped / 0 failed / 0 errors**
- focused P8 tests: **20/20**
- random convex binary MINLP campaign: **100/100 OA vs enumeration**, max objective difference `6.3916273662645295e-09`
- fractional-relaxation campaign: **60/60**, mean OA iterations `4.0333`, max `8`
- scaling campaign (4–8 binaries): **10/10**, max OA iterations `156`
- official MINLPLib `batch` page-formulation reconstruction: objective `285506.5042651033`, published primal bound `285506.5082`, absolute difference `0.003934896667`, globally proven in `3` OA iterations

The MINLPLib result is explicitly an **official-page formulation reconstruction**, not raw archive execution.

## Performance evidence

Fresh release-evidence rerun retained the negative result:

- fractional small median per-case OA/enumeration ratio: `2.1008`
- 4–8 binary scaling median per-case OA/enumeration ratio: `2.5762`

No OA speed claim is made for P8 on these small workloads.

## Release evidence

- wheel: `optimind_core_codename-0.0.39-py3-none-any.whl`
- wheel SHA-256: `9717942fd465ace6f8b3de4c6f0c168eb89b231f05ce4a7aac8d234d38af013e`
- wheel member integrity: PASS
- isolated out-of-source wheel smoke: PASS
- source regression: PASS
- manifest / internal checksums / outer ZIP integrity: PASS

## Deferred from P8 v1

- general integer MINLP
- nonconvex/global MINLP (GOA/McCormick global relaxations)
- ECP, feasibility pump and LP/NLP branch-and-bound production implementations
- nonlinear equality handling
- indicator+MINLP bridge composition
- official MINLPLib raw archive/parser execution
