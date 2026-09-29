# Certificate recovery and measured experimental routing

Development after published 0.4. This change does not publish a new version or
enable automatic learned routing. Historical experiments remain frozen.

## Original-model certificate recovery

The equality elimination previously searched original sparse column adjacency.
Eliminating a pivot can introduce a variable into another row; the old search
missed that row and could stop after one pivot. Recovery now maintains transformed
row incidence and reports bounded, no-equality-pivot, time, work, pivot and bit
limit outcomes. Every substitution uses exact rationals of the represented
binary64 inputs. No small residual is silently rounded to zero.

Equality rows alone do not finish pilot4. The next stage introduces nonnegative
slacks for original inequalities **and variable bounds**, retaining the existing
dual-weighted slack penalties. This distinction matters: discarding a tiny
negative residual on an unbounded variable would be invalid. A two-sided row may
be relaxed to one side, never strengthened to a numerically active equality.

The HiGHS adapter supplies `raw_statistics['lp_basis']` as an elimination hint.
Its indices and model hash are checked. The basis never establishes correctness:
the final primal validation, stationarity, complementarity and exact corrected
gap must all pass. A structurally valid but poor hint may fail to recover a bound.

```python
from solverpilot import solve
from solverpilot.backends import HighspyNativeBackend
from solverpilot.validate.lp_dual import recover_lp_optimality

result = solve(problem, backend=HighspyNativeBackend(solver="simplex", threads=1))
check = recover_lp_optimality(
    problem, result.x, result.raw_statistics["canonical_dual"],
    basis=result.raw_statistics.get("lp_basis"), time_limit_s=30,
)
print(check.verified, check.gap, check.recovery_diagnostics)
```

Windows and Linux/WSL replays of the consumed pilot4 model both pass the numerical
certificate check with corrected gap about 4.242e-7. The ordinary solve still
reports its original certificate status; this optional, more expensive recovery
does not overwrite it. This is a tolerance-qualified numerical certificate, not
an exact feasible rational primal or a zero-gap proof. unitcal_7 remains verified.
The JSON observations in `evidence/certificate-routing/` retain the original
model hashes, proof diagnostics and measured recovery times. These are correctness
replays, not an unseen routing benchmark or a controlled speed comparison.

`solverpilot.validate.escalation.recover_lp_with_fallback` optionally escalates to
the existing exact SCIP/VIPR path. Supply both trusted executable paths explicitly.
Numerical recovery and exact execution share the caller's remaining budget; late
results do not qualify. An exact bound without an optimal solution is reported as
`verified_bound`, never optimal. An exact fallback solution is separate from the
original numerical candidate. No solver is downloaded by this API.

The real pinned native fallback also solved pilot4 on this machine: SCIP produced
a certificate and VIPR independently verified an exact feasible rational optimum
with zero gap in approximately 7.4 seconds including model conversion/checking.
This is a separate experiment from the numerical certificate above. Its summary
retains solver/checker/certificate hashes. Native execution exposed two additional
defects now covered by regression: a temporary proof fragment can disappear during
size monitoring, and a legitimate rational token can exceed 4096 characters while
each integer component is smaller. The parser now bounds components separately
(4096 decimal digits each) and retains the 32 MiB certificate size cap.

Numerical time checks are cooperative, including preparation and final checking;
they are not a hard latency guarantee. Work caps apply separately to each algebra
stage. Use an owned isolated process where a hard stopping boundary is needed.
The public recovery wrapper deducts dual preparation and basis validation before
starting verification, and rejects a positive result returned after its total
budget. Deterministic clock tests cover expiry in either preparation stage and at
the final return boundary.
Greedy basis recovery remains incomplete; resource limits or an unsuitable basis
can return unverified without implying infeasibility or nonoptimality.

## Validated repeated-routing sessions

`LPRoutingSession.create` hashes the implementation and binds immutable model,
guard, environment, cutoff and backend configurations once. Each warm decision
checks the same artifact objects and source-file metadata. Ordinary edits, file
replacement, unavailable candidates, changed settings or a changed environment
cause conservative fallback. Source checks, availability, feature extraction,
routing, solving and verification consume the call budget.

```python
from solverpilot.experimental.lp_session import LPRoutingSession
from solverpilot.experimental.robust_lp import solve_robust_lp

session = LPRoutingSession.create(
    model, guard, environment_id=measured_environment_id,
    cutoff_s=2., backends=backends,
)
result = solve_robust_lp(
    problem, model, gain_guard=guard, session=session, backends=backends,
    environment_id=measured_environment_id, time_limit_s=2.,
)
```

Session preparation is explicitly reported outside an individual repeated call.
Cold benchmarks must charge all preparation; amortized benchmarks must declare
their reuse count in advance. A decision-only session may omit backends, but
cannot authorize a solve using unbound backend objects. Bound backends require
serializable dataclass settings. Recreate after code reload or dependency/config
changes. Metadata checks are not protection against deliberate same-metadata file
rewrites, and checksums do not authenticate a benchmark or its author.

A decision-only replay on an installed wheel in the Linux filesystem measured
eight eligible recommendations: median stateless cost 0.656 ms, session cost
0.110 ms, with approximately 0.745 ms one-time preparation. All eight were
permitted in both modes. This reuses consumed historical observations and assumes
their environment binding; it is not new solve-speed or promotion evidence.
On the Windows-mounted checkout under WSL, metadata checks remained expensive
enough to reject eligible switches at the 5 ms cap. The session deliberately
keeps that conservative behavior; use an ordinary native-filesystem installation
for that workload rather than weakening integrity or hiding preparation cost.

The legacy evaluator now charges decision overhead to **each repeat before**
applying the deadline and PAR10 penalty. A solve that crosses its deadline after
routing is no longer counted as successful. Historical evidence is not rewritten;
its unaffected numerical metrics continue to replay with the corrected label.

## Versioned small-tree research

`solverpilot.experimental.learned_lp_v2` leaves the v1 schema and loaders intact.
It adds five sparse features: mean row/column occupancy, free/fixed variable
fractions and log coefficient range. The thirteen features require no dense
matrix, solver probe or learned runtime dependency.

The deterministic cost-sensitive tree has at most two split levels. Each family
has equal total training weight, so many variants cannot dominate a small family.
Only training rows fit the tree. Different calibration families qualify individual
leaves, with minimum family support, gain after overhead and no lost verified
repeats. Test families and hashes must be disjoint from both sets. Family labels
must be supplied and audited honestly; no generic schema proves semantic novelty.

Evaluation reports cold and fixed-amortized costs, complete repeated outcomes,
training-frozen best-single-solver and production comparisons, group bootstrap,
tail slowdown and verified-success losses. The manifest-driven development runner
also measures actual routed calls against default and production. Synthetic
development runs and consumed public replays cannot promote industrial routing.
Automatic production selection remains disabled until a fresh public qualification
meets its preregistered gates on the relevant hardware and budget.

The initial v2 development run contains eight distinct structural templates and
52 actual solver calls, including four guarded test calls and both baselines.
All four guarded test calls verified within budget. No leaf qualified on the
separate calibration families, so the guard correctly used production fallback.
It did not establish a learned speed advantage: cold mean test cost was about
17.60 ms and the fixed 20-call amortized scenario about 13.32 ms, versus 12.95 ms
for production. These tiny synthetic examples validate execution/accounting only;
they do not support a public-family performance claim.

Final review found that the initial scheduler's repeat-dependent rotation could
cancel its odd-repeat reversal for two candidates. The runner now rotates by case
only and reverses alternate repeats. The original measurements above are retained
unchanged; `development-balanced-run/` captures a second 52-call replay on the same
already-consumed development templates, with exactly reversed strategy order for
every repeat pair. All four guarded test calls verified, and again no leaf
qualified. Cold mean was 17.89 ms; fixed-20 amortized mean was 14.33 ms versus
15.61 ms for production. The guard used production fallback throughout, so the
small timing difference is not evidence of a learned speed advantage. Neither run
is an independent industrial qualification or authorizes automatic activation.
