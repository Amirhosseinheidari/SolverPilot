# BENCHMARK REPORT M13

## Direct OSQP C API vs CasADi OSQP bridge

- Instances: **96** constructed-KKT convex QPs
- Direct/bridge solver calls: **192**
- Direct failures: **0**
- Bridge failures: **0**
- Max direct objective error: `1.7763568394002505e-15`
- Max direct-vs-bridge objective difference: `0.0`
- Max direct x infinity error: `8.881784197001252e-16`
- Verified OSQP core: **0.6.3**
- Library SHA-256: `6e69e9bd09cd9db73d410e61af1f1eb409c161d892a61911b2b4fd1480a257a3`
- Header SHA-256: `fd7243e3907e26618f8965954a9cba3c29d16a3622ca96696bb8503972b3cc6b`

Scope: ABI/binding correctness verification only; this is not a benchmark of the public OSQP Python package.

## Direct HiGHS C API exact MILP cross-check

- Binary MILP instances: **100**
- Solver calls: **300**
- Direct HiGHS failures: **0**
- SciPy/HiGHS failures: **0**
- CBC failures: **0**
- Max direct HiGHS objective error vs brute force: `1.2789769243681803e-13`
- CBC max objective error: `0.0`
- Verified HiGHS core: **1.10.0**
- Library SHA-256: `bce96689405f83a849500fa3cf680d61b3eeb2ab179d358726e86526a047df4f`
- Header SHA-256: `b5384c3f1169c8cb09d21f9bc4e14c55bb3b62654c03ec87cd8ae47848365593`

## Direct C-API reoptimization

### HiGHS LP — persistent model + numerical update + explicit basis

| Size | Median stateful/cold | Approx speedup | Median iteration ratio |
|---|---:|---:|---:|
| small | 0.901 | 1.11x | 0.112 |
| medium | 0.916 | 1.09x | 0.169 |
| large | 0.757 | 1.32x | 0.169 |

All LP sequence objectives agreed and every expected reuse step was observed.

### OSQP QP — persistent workspace + documented updates + automatic warm-start

| n | Median stateful/cold | Approx speedup |
|---:|---:|---:|
| 20 | 0.961 | 1.04x |
| 60 | 0.942 | 1.06x |
| 120 | 0.920 | 1.09x |

All QP sequence objectives agreed. Every sequence observed the expected reuse; matrix-value update steps were present. **No factorization-reuse claim is made.**

## Worker startup audit

On this platform only:

- normal Python startup: **0.930s**
- `python -S` no-site worker: **0.438s**

The difference is platform startup-hook overhead, not solver speed. Worker mode is therefore part of the benchmark protocol identity.

## Fresh deployable `@auto` trust smoke

- SBS: **scipy-highs-ds**
- policy/SBS point ratio: **1.128**
- paired bootstrap 95% CI: **[1.022, 1.307]**
- fixed-SBS external policy reconstruction: **1.0**
- mixed-environment merge rejected: **True**

The entire CI is above 1 in this fresh short-LP smoke. This is **negative evidence** for the current deployable auto-selection overhead on short problems, not a general solver-selection conclusion. No production performance rule was added.
