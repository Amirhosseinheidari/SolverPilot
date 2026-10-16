# SolverPilot 0.4.1 documentation contract

- Version metadata, import version and current API/backend contracts are aligned.
- Current API signatures include additive proof controls and trace fields; old
  snapshots remain historical compatibility checks.
- Release scope includes all accumulated post-0.4 changes, not only PR #27.
- LP tolerances remain unchanged; recovery has explicit arithmetic/work bounds.
- Numerical verification is distinguished from exact primal/global certificates.
- Individual timing regressions and the historical measured wheel identities are retained.
- Native model/basis reuse does not imply native factorization reuse.
- Exact executables and GPU hardware are separately qualified; learned routing
  stays experimental and disabled in production.
- Qualification uses the final main SHA and exact wheel/sdist hashes, followed by
  TestPyPI installation verification and publication of those same bytes to PyPI.
- Fresh published wheel and source-distribution installations are checked for
  version/import ownership, dependencies, smoke examples and all 256 frozen LP checks.
