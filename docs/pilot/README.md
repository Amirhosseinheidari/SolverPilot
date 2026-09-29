# Production workflow technical trial

This folder provides an attributed public educational example, a separate synthetic diagnostic case, and blank feedback templates. It supports a reproducible technical trial of the continuous-production workflow. It contains no real participant feedback, customer operating data, measured human time savings or evidence of adoption. A real human pilot requires a participant to review the workflow with their own stated task and provide feedback.

## Inputs and reference results

The baseline in [nd-production.json](nd-production.json) adapts the numerical model from Jeffrey C. Kantor's [ND Pyomo Cookbook, section 2.1.4](https://jckantor.github.io/ND-Pyomo-Cookbook/notebooks/02.01-Production-Models-with-Linear-Constraints.html#production-plan-mixed-product-strategy). The [project repository](https://github.com/jckantor/ND-Pyomo-Cookbook) identifies the collection as instructional material developed at Notre Dame. The capacity and price variations below were prepared locally for this trial.

| Case | X | Y | Profit, USD/week | Interpretation |
| --- | ---: | ---: | ---: | --- |
| baseline | 20 | 60 | 2600 | Published source optimum |
| more-labor | 10 | 80 | 2800 | Locally derived scenario; capacity [90,100] |
| tighter-capacity | 30 | 40 | 2400 | Locally derived scenario; capacity [70,100] |
| changed-prices | 0 | 80 | 3200 | Locally derived scenario; profit [30,40] |
| reset | 20 | 60 | 2600 | Omitted parameters return to baseline |

An independent arithmetic reference for the baseline is

`40X + 30Y = 20(X + Y) + 10(2X + Y) <= 20*80 + 10*100 = 2600`.

The feasible point `(20,60)` attains that upper bound. For the two capacity variations, replacing the first capacity by 90 or 70 gives bounds 2800 and 2400, attained by the listed points. For changed prices, `30X+40Y = 40(X+Y)-10X <= 3200`, attained by `(0,80)`. These small exact arithmetic references are independently written checks of these inputs. They do not upgrade the runtime's numerical evidence to an exact solver certificate.

The [synthetic-minimum-commitments.json](synthetic-minimum-commitments.json) fixture adds invented commitments `X>=10` and `Y>=10`. Its baseline and reset still give `(20,60)` and 2600. Its shortage scenario has Labor A capacity 5, incompatible with `X+Y>=20`; it should report the scoped capacity contradiction and no accepted objective or objective delta. These commitments and shortage are not supplied by the source or by a real business.

## Run and replay

The portable kit contains the exact candidate wheel, `runtime-lock.txt`, this directory and the benchmark runner. Extract the kit and open a terminal in its root. This candidate still carries package version 0.4; it is an unreleased development artifact identified by its wheel SHA-256, not the published PyPI 0.4 wheel. Install the enclosed wheel explicitly. These PowerShell commands use Python 3.12 without activating or modifying an existing environment:

```powershell
py -3.12 -m venv .pilot-origin
.\.pilot-origin\Scripts\python.exe -I -m pip install -r runtime-lock.txt .\solverpilot-0.4-py3-none-any.whl
.\.pilot-origin\Scripts\python.exe -I benchmarks/qualify_production_workflow.py run --input docs/pilot/nd-production.json --output public-trial --repeats 6 --include-model

py -3.12 -m venv .pilot-replay
.\.pilot-replay\Scripts\python.exe -I -m pip install -r runtime-lock.txt .\solverpilot-0.4-py3-none-any.whl
.\.pilot-replay\Scripts\python.exe -I benchmarks/qualify_production_workflow.py replay --study public-trial/study.json --output independent-replay.json
```

Use fresh output paths for later runs. On POSIX, use `python3.12 -m venv` and the corresponding `.pilot-origin/bin/python` and `.pilot-replay/bin/python` paths. `SHA256SUMS.json` lists kit file digests; these detect corruption but are not signatures. Keep the wheel, dependency pins and emitted provenance together. Do not substitute an editable source install or a different package build.

Maintainers can reproduce the complete installation check from a repository checkout with:

```powershell
python benchmarks/qualify_production_install.py --wheel PATH-TO-CANDIDATE.whl --output NEW-INSTALL-QUALIFICATION --repeats 6 --include-model
```

The helper creates two fresh environments, checks isolated import provenance, runs both fixtures in the first, and replays them in separate processes in the second. Its `qualification.json` reports the clean-installation gate and packages a portable kit. The ordinary workflow runner's `technical_gate_passed` covers only reference agreement and same-environment replay; it does not independently certify clean installation.

Choose a fresh output directory for each run:

```powershell
python benchmarks/qualify_production_workflow.py run --input docs/pilot/nd-production.json --output NEWDIR --repeats 6 --include-model
python benchmarks/qualify_production_workflow.py replay --study NEWDIR/study.json --output NEWJSON
```

For the synthetic diagnostic fixture, use another output directory:

```powershell
python benchmarks/qualify_production_workflow.py run --input docs/pilot/synthetic-minimum-commitments.json --output NEWDIR-SYNTHETIC --repeats 6 --include-model
```

`NEWDIR`, `NEWJSON` and `NEWDIR-SYNTHETIC` are placeholders for new local paths. `--include-model` explicitly includes the full contract and canonical model needed for replay. Inspect exported inputs, labels and decisions before sharing. No participant feedback is generated by these commands.

Use the same wheel and exact dependency versions for replay. Source/version checks and a fresh execution identity are part of replay; a replayed result is a new solve linked to the earlier run. See [the workflow documentation](../PRODUCTION-EVIDENCE-WORKFLOW.md) for the export policy, evidence meanings and replay limits.

Repeated machine timings describe only the measured execution path on that environment. Six repetitions do not establish speedup over another tool, human time savings, user comprehension or industrial scalability. Keep setup, compilation, solve, validation and reporting costs distinct when the output exposes them; do not relabel total workflow time as solver time.

Input reading/validation, aggregate workflow, direct-reference construction/solve/validation, report rendering, export and replay are timed. Compilation and contract audits are included in aggregate workflow time, not measured separately. Native import/startup and human data preparation are excluded. Warmups are recorded separately, and measured trial order alternates. Origin `kind`, `url` and `attribution` text are exported even without full-model opt-in; arbitrary nested origin metadata is rejected.

## Scope and interpretation

The contract maximizes linear per-unit profit for continuous production quantities with nonnegative resource consumption, nonnegative resource capacities, and fixed lower/upper quantity bounds. Product X has maximum 40; the JSON string `"inf"` denotes no finite upper bound for Y. The workflow does not add integer batches, setup costs, inventory, lead times or stochastic recourse. Units are supplied by the caller.

Formulation agreement, independent contract candidate checks and numerical optimality evidence answer different questions. Acceptance does not prove that all real business requirements were specified. Changed profit coefficients intentionally suppress an objective delta against the baseline because the objective definition changed. Invalid inputs, infeasible scenarios and missing candidates must not appear as zero-profit solutions.

## Human feedback, when available

Copy [feedback-template.md](feedback-template.md) or [feedback-template.json](feedback-template.json) into a separate local result file. Both are blank templates; `null` means no response or measurement has been supplied. Keep unknown fields null. A participant can review the generated report, explain a decision and a capacity shortage in their own words, and identify requirements missing from the contract. Only record responses they actually provide.

The form distinguishes observed elapsed time from estimates. Record a manual baseline only when a person actually performs or measures that baseline under a described task. Do not convert machine timings into manual effort saved. Use a participant alias where needed; sharing feedback remains an explicit participant decision.

## Source and licensing

The numerical baseline was adapted from the source's code formulation. Attribute it to Jeffrey C. Kantor, ND Pyomo Cookbook, section 2.1.4, and retain [LICENSE-ND-PYOMO-CODE.txt](LICENSE-ND-PYOMO-CODE.txt), copied from the project's [official code license](https://raw.githubusercontent.com/jckantor/ND-Pyomo-Cookbook/main/LICENSE-CODE.txt): MIT, copyright 2019 Jeffrey Kantor. Source checked on 2026-09-29.

The project separately identifies its notebook text as [CC BY-NC-ND 4.0](https://raw.githubusercontent.com/jckantor/ND-Pyomo-Cookbook/main/LICENSE-TEXT.txt). This folder does not reproduce its prose or figures. The explanations and scenario variations here were written for SolverPilot, and the synthetic variation is explicitly labeled.
