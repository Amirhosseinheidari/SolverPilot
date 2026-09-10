# SolverPilot public documentation — 0.0.40rc2

`0.0.40rc2` documents the integrated SolverPilot Core + Track P P0–P9 prerelease.

## Public documentation surface

The README covers:

- what SolverPilot is and its trust boundary;
- Python/runtime requirements and installation paths;
- LP, MILP, and continuous convex QP examples;
- backend selection and `SolveResult` semantics;
- infeasibility diagnostics;
- repeated solve/reoptimization;
- semantic modeling;
- conic modeling boundaries;
- local NLP with CasADi/Ipopt;
- certified convex binary MINLP scope;
- CP reference and optional CP-SAT execution;
- PersistentSession usage;
- optional dependencies;
- explicit unsupported/uncertified capabilities;
- API/provenance/security/contribution links.

The repository contains **14 executable public examples** under `examples/`. All 14 execute successfully against the installed diagnostic wheel in the verified environment. Optional examples fail closed when their declared dependency is unavailable.

## Important claim boundaries

Documentation does not claim:

- universal or learned performance superiority;
- a production learned LP selector;
- global optimality for local NLP candidates;
- general nonconvex MINLP support;
- universal conic solver coverage;
- in-process CP-SAT ABI compatibility with arbitrary native solver load orders.

The CP-SAT backend is intentionally subprocess-isolated because integration testing found a real native HiGHS ABI collision when SciPy/HiGHS is loaded before OR-Tools CP-SAT on Linux. Returned CP assignments are independently revalidated in the parent process.
