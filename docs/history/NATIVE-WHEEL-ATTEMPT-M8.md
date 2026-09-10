# Native Wheel Network Attempt — M8

M8 retried all three exact official wheel URLs locked in `benchmarks/native-wheel-lock.json`. The runtime could not resolve `files.pythonhosted.org`; each curl request returned code 6 before download.

No binary was accepted without successful transfer and SHA-256 verification. Consequently:

- `highspy` remains unexecuted as a public package;
- `osqp` remains unexecuted as a public package;
- `pyscipopt` remains unexecuted as a public package.

Machine-readable evidence: `benchmarks/results/m8-native-wheel-network-attempt.json`.
