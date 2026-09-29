# Exact LP/MILP certificates (development)

`solverpilot.exact.solve_exact` is an explicit optional API. It runs a caller-selected
exact SCIP executable, binds the emitted VIPR 1.0 model to the original `LinearProblem`,
and invokes a separate caller-selected VIPR checker. It does not change production
routing. The ordinary PySCIPOpt wheel is not a substitute for the exact executable.

```python
from solverpilot import LinearProblem
from solverpilot.exact import solve_exact, verify_exact_certificate

problem = LinearProblem(
    A=[[2, 2]], c=[1, 1], variable_lower=[0, 0], variable_upper=[2, 2],
    constraint_lower=[3], constraint_upper=[float("inf")],
    domains=["integer", "integer"],
)
result = solve_exact(
    problem,
    scip_executable="/absolute/path/to/scip",
    checker_executable="/absolute/path/to/viprchk",
    time_limit=60,
    evidence_directory="exact-evidence",
)
print(result.status, result.objective, result.absolute_gap, result.reason)
```

## What is proved

- `optimal`: an exactly feasible rational solution for the original model and a
  separately checked bound have identical objective values.
- `infeasible`: VIPR proved infeasibility from assumptions valid for the original model.
- `bound_verified`: a finite bound was checked, but exact optimality was not proved.
  `independently_verified` refers to this bound, not to optimality.
- `unverified`: no usable proof was established. This includes interruption, checker
  failure, missing certificates, unsupported transformations and unboundedness.

`x`, `objective`, `bound` and `absolute_gap` use `fractions.Fraction`; floating-point
conversion is the caller's choice. For maximization, `bound` is an upper bound;
for minimization it is a lower bound. The original objective offset is restored.

Exactness refers to the **binary64 coefficients stored in LinearProblem**. For example,
`0.1` is verified as its exact binary floating-point fraction, not as the decimal 1/10.
This release does not introduce a rational-text modeling interface.

## Model binding and trust boundary

The checker alone verifies the problem written inside a certificate. SolverPilot also
checks that every certificate assumption is an original constraint/bound, an equivalent
positive scaling, or a valid rounded integer-variable bound. The objective and variable
mapping must match; all certificate solutions are checked against the original rows,
bounds and integrality using rational arithmetic. A certificate may describe a relaxation
with fewer constraints: its lower bound remains valid, while primal feasibility is still
checked on the complete original model. Unsupported presolve transformations fail closed.

SCIP runs with exact mode enabled **before** reading the problem, no objective scaling,
no presolve rounds/restarts, and separation disabled. This conservative first path avoids
incomplete cut certificates requiring `viprcomp`. It can be slower than standard SCIP.

SCIP's informational `global` suffix is accepted after derivations. When its proof
stops at variable bounds, SolverPilot can append a rational linear combination that
closes the objective bound. VIPR checks this added step and all its dependencies;
the `global` annotation is never treated as an axiom. Both input and checked certificate
hashes are returned, and the original solver proof is retained in the evidence directory.

Executables are trusted native programs explicitly selected by the caller. Their SHA256
hashes and the certificate hash are returned for reproducibility; hashes are provenance,
not a replacement for mathematical checking. This is not a formally verified entire
software stack. Saved result flags are never accepted as proof: use
`verify_exact_certificate(problem, certificate_path, checker_executable=...)` to recheck.

Certificates are size-bounded, fully parsed, and copied into a private execution directory.
Malformed, truncated, trailing, duplicate-index, unsupported-version and incomplete proofs
are rejected. A successful checker exit must also contain its explicit success marker.
The checked bytes and executable hashes are checked again before accepting the result.
Preparation and verification share the solve wall budget. Process termination/cleanup and
individual Python parsing operations may overrun that wall budget; such runs cannot verify.

## Building and qualification

`tools/build_exact_runtime.sh /absolute/build/root` builds pinned SCIP 10.0.3, SoPlex
8.0.3 and VIPR sources. Linux build packages: CMake, Ninja, C++ compiler, GMP, MPFR,
Boost, TBB and zlib development libraries. The script does not install OS packages.
The GitHub `Exact MILP qualification` workflow installs these packages on its disposable
Ubuntu runner and exercises real generation, replay, and false-proof rejection.

Native qualification is Linux-specific. Windows parser and process cleanup unit tests
do not establish that an exact Windows SCIP build is supported. Native GPU execution,
general nonlinear proofs, exact MIQP/MINLP and learned routing are separate work.

Initial resource scope: at most 10,000 variables, 10,000 rows, 100,000 matrix nonzeros,
32 MiB per certificate/log, and 200,000 derivations. These are implementation budgets,
not mathematical limits. A supplied checker is not sandboxed against malicious behavior.

Sources: [SCIP exact mode](https://www.scipopt.org/doc-10.0.3/html/EXACT.php),
[VIPR format and checker](https://github.com/scipopt/vipr),
[certificate format 1.0](https://github.com/scipopt/vipr/blob/master/cert_spec_v1_0.md).
