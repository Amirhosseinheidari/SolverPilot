# Benchopt handoff — M11

Benchopt 1.9.1 is the recommended optional orchestration layer once the dedicated public solver environment exists. M11 does not make Benchopt a runtime dependency because the raw runner must remain usable in minimal/HPC environments.

Benchopt 1.9.1 provides the pieces we do not want to reimplement as a competing framework: caching, repetitions/seeds, parallel and SLURM execution, fresh-environment benchmark testing, result merging, visualization and publishing.

Recommended integration sequence on the network-enabled benchmark machine:

```bash
pip install 'benchopt>=1.9.1,<2'
benchopt --check-env
```

Before publishing any Benchopt wrapper around OptiMind, require:

```bash
benchopt test <benchmark-repository>
```

The M11 raw JSONL protocol should remain the audit source of truth for solver status, canonical validation, reference checking, instance hashes and planner overhead. A future Benchopt adapter should consume the same frozen dataset manifests and backend list rather than inventing a separate experiment definition.

Benchopt was **not installed or executed in the M11 runtime**, so this document is a handoff contract, not a completed Benchopt benchmark.
