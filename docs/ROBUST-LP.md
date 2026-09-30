# Conservative LP abstention and numerical certificate recovery

The [stage4 public challenge](evidence/public-lp-local/README.md) exposed two
separate weaknesses: a synthetic-trained selector fell back to PDLP on every
unfamiliar instance, and some primal-valid LP solutions lacked independently
verified dual bounds. This development change addresses both without treating
solver optimality status as a proof. No new PyPI release is implied.

## Explicit routing, conservative abstention

`solverpilot.experimental.robust_lp.solve_robust_lp` is an opt-in experimental
wrapper. In-support candidates still come from the unchanged cost-sensitive stump.
An environment mismatch, out-of-support feature vector, unavailable recommended
candidate or absent fallback invokes ordinary `solve_production()` instead of the
training-only best solver. Neither default API installs this selector automatically.

The caller supplies a loaded `LPSelector`, measured environment identity and a map
from frozen candidate IDs to backend objects. Model/environment discovery outside
the wrapper is the caller's responsibility; the qualification worker includes it
in its API timing. Availability checks, decision and solve share one remaining
budget. A failed solve is not retried with a fresh budget. Setup exhaustion raises
`TimeoutError`. The existing native time limit does not impose a hard deadline on
validation; qualification additionally enforces a process-group deadline.

The returned raw statistics include `experimental_lp_route`, its reason, original
selector reason, model digest and measured call time. An abstention is visible,
not relabeled as a learned success. The historical v1 model, decision function and
synthetic/public evidence remain unchanged.

For an existing continuous `LinearProblem` named `problem`, a development checkout
can explicitly use a captured model as follows. Environment mismatch safely routes
to the production planner; copying a model does not qualify it for a new machine.

```python
from pathlib import Path
from solverpilot.backends import HighspyNativeBackend, PDLPBackend
from solverpilot.benchmark.environment import capture_environment
from solverpilot.experimental.learned_lp import LPSelector
from solverpilot.experimental.lp_environment import bind_lp_environment
from solverpilot.experimental.robust_lp import solve_robust_lp

folder = Path("docs/evidence/robust-lp-local")
model = LPSelector.load(folder / "model.json")
binding = bind_lp_environment(model, folder / "protocol.json",
    capture_environment(packages=("numpy", "scipy", "highspy", "ortools")))
result = solve_robust_lp(problem, model,
    environment_id=binding["decision_environment_id"], time_limit_s=2,
    backends={
        "highs-simplex": HighspyNativeBackend(solver="simplex", threads=1),
        "highs-ipm": HighspyNativeBackend(solver="ipm", threads=1),
        "ortools-pdlp": PDLPBackend(threads=1),
    })
print(result.optimality_evidence.independently_verified_optimal)
print(result.raw_statistics["experimental_lp_route"])
```

## Certificate repair preserves independent checks

HiGHS reports lower/upper-bound dual values with opposite signs; SolverPilot's
canonical convention is positive for upper bounds and negative for lower bounds.
See the official [HiGHS terminology](https://ergo-code.github.io/HiGHS/stable/terminology/).

For continuous LPs only, an illegal multiplier toward an infinite original bound
can be projected to zero. This is merely a new candidate: the runtime preserves
`canonical_dual` and `original_optimality_check`, records
`prepared_canonical_dual`, then repeats the independent stationarity,
complementarity, primal-validity and corrected-gap checks. Projection alone never
establishes optimality and does not repair arbitrary dual errors.

For minimization, the checker bounds the Lagrangian residual `r = c + B.T*y`.
If `min r.T*x` over the original variable box is infinite, a small residual is
insufficient. Two bounded procedures can establish a finite lower bound using
the original linear constraints:

1. **Implied box:** for `a*x + rest <= b`, a valid lower bound on `rest` implies
   an upper bound on `a*x`; the lower-row case is symmetric. Arithmetic is exact
   on represented binary64 values and conversion rounds outward. At most three
   passes and 50,000 nonzero visits are used. Contradictory boxes are discarded.
2. **Residual substitution:** use the exact identity
   `r.T*x = (r_j/a_ij)*(A_i*x) + remaining.T*x`. A finite row endpoint bounds
   the first term from below. Eliminate that coefficient exactly, without
   reintroducing an already eliminated variable. The remaining form is bounded
   over the enclosing box. Work is capped at 50,000 row-nonzero visits and
   coefficient growth is guarded. Coupled free variables need not have finite
   individual bounds for this to succeed.

Every step is a consequence of the original model. The lower bound may be loose
or unavailable; then independent optimality remains false. No residual is simply
rounded to zero, no feasibility tolerance is loosened, and no generic global NLP
or mixed-integer guarantee is added. `optimality_check.domain_refined` records
successful residual-bound refinement. These remain tolerance-qualified numerical
optimality checks on the supplied binary64 model, not external exact certificates.
For background on bound propagation, see [SCIP propagators](https://scipopt.org/doc/html/group__PROPAGATORS.php).

## Public retraining and fresh evaluation

`benchmarks/qualify_robust_lp.py` freezes source hashes and a protocol before any
training solve. Development/training uses the 24 consumed stage4 public models:
12 Netlib LPs and 12 MIPLIB continuous relaxations. This is broader than the earlier
synthetic-only training, but is still a small, single-host public sample.

A fresh cohort is prepared by `prepare_public_lp.py`, excluding the audited prior
names and related families, including all stage4 instances. Both readers must
agree on each model. The runner additionally rejects training/test name, group or
canonical-data-hash overlap. No difficult timed solve can cause replacement.

The training run measures three candidates, two repeats each: 144 calls. Its
target is full fresh-process wall time for independently verified results within
2 seconds of API work and 12 seconds of process wall; every other repeat costs
120 seconds. All failures and late repeats count. Stump settings are fixed:
minimum four groups per leaf and 0.005-second switch margin. The baseline is
selected from this public training data only. No support-range expansion based
on the held-out set is allowed.

The [recorded outcome](evidence/robust-lp-local/README.md) passed comparisons
against `solve_production()` but failed the combined promotion rule against both
default APIs. Automatic learned routing remains disabled.

The model digest is saved before any held-out solve. Evaluation performs 288 calls
on 24 fresh public instances: the explicit robust route, ordinary default and
production APIs, HiGHS simplex/IPM and CPU PDLP, with two order-balanced repeats.
There are also 24 separate stress calls. Calls are serial, in fresh processes,
under one logical CPU affinity and one-thread BLAS settings on this laptop's WSL.
Environment binding tolerates only the existing bounded guest-RAM reporting drift.

The preregistered stage4 gate is reused against both default APIs: at least 20
groups, actual in-support learned switches, no verified objective disagreement,
at least 3% lower PAR10 cost, group-bootstrap upper ratio below 1, p90 slowdown at
most 25%, and no lost within-budget verified repetitions on any instance. A
constant baseline or widespread fallback cannot establish learned-selection value.
No result alone authorizes automatic production routing. PAR10 is a penalized
completion cost, not average solve latency.

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 \
python benchmarks/qualify_robust_lp.py \
  --training consumed-stage4-corpus --heldout fresh-public-corpus \
  --output new-output-directory
```

The output directory must not already exist. Training and test outcomes, all
admission exclusions, source/model hashes and the full gate decision are retained.
After evaluation, this fresh cohort is consumed and cannot be reused as new
held-out evidence for a revised policy.
