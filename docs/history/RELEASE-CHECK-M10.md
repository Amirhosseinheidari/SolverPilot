# Release Check M10

- Version: 0.0.11
- Source pytest: 166 passed / 3 skipped / 0 failed / 0 errors
- compileall: PASS
- wheel build: PASS
- isolated wheel install: PASS
- isolated wheel OSQP QP solve: PASS
- isolated wheel OSQP second-solve reuse evidence: PASS (`casadi_osqp_primal_warm_start`)
- isolated wheel CBC MILP solve/validation: PASS
- CasADi verification bridges absent from default_registry: PASS
- public highspy/osqp/PySCIPOpt packages: still unavailable in this runtime
