# SolverPilot 0.3 release gates

- Keep the 78 historical top-level names/signatures compatible.
- Pass the six 0.2 audit regressions plus scaling, KKT derivative, atom and scenario oracles.
- Check cold/repeated compilation and process/sequential batch costs independently.
- Run Ruff correctness rules and mypy common-API contracts.
- Require at least 80% combined line/branch coverage in full optional-integration CI.
- Qualify the exact wheel and sdist on the supported Python/OS/optional-backend matrix.
- Check package metadata, `py.typed`, artifact hashes, SBOM and dependency audit.
- Preserve 0.1 and 0.2 release artifacts and historical evidence.
- Publish only the exact artifacts that passed external qualification.
