# Independent workflow comparison protocol

This is a prospective protocol, not a completed human study. No participant,
feedback, task time or productivity gain is implied by the technical benchmark.
Use [comparison-feedback-template.json](comparison-feedback-template.json) for
observed sessions. Keep missing observations null.

## Freeze before observing participants

Record the exact candidate wheel hash, dependency lock, task files, reference
answers, comparison wrapper source and this protocol's hash. Describe the
participant's prior Python and optimization experience. Choose one comparison
tool and backend before running the session. Use the same native solver version,
threads, tolerances and time limit in both workflows.

The comparator must include its documented status/feasibility checks, persistent
or parameterized update support, and a clearly identified application wrapper
for any contract checking or export being compared. Do not compare a complete
SolverPilot workflow to an intentionally unchecked one-line solve. Record the
time to build that wrapper separately from participant task time.

## Matched tasks and order

Prepare two equivalent task variants, with different coefficients and product
names, from the public production fixture. An observer checks reference answers
beforehand using an independently assembled LP and an analytic or exact bound
where available. Keep those answers separate from participant instructions.

Assign the first participant SolverPilot then the comparator, the next the
reverse order. Alternate which numerical variant is paired with each tool.
Explain both tools equally and retain the training time separately. A small
initial pilot measures usability issues, not population-wide superiority.

For each workflow, observe these tasks:

1. Load the supplied production data, solve and explain a feasible decision.
2. Change one resource capacity, solve again and explain the objective change.
3. Diagnose a scenario whose fixed minimum commitments exceed capacity.
4. Review an intentionally omitted capacity row and identify the mismatch with
   the supplied contract. Do not supply a wrong contract to only one tool.
5. Explain what evidence supports optimality, and what remains a solver claim.
6. Save the result, rerun it in a fresh installation, and identify the new run and
   the original run it came from, using the comparator's equivalent wrapper.

Also ask the participant to identify a business requirement missing from the
given contract. Neither tool receives credit for proving completeness of an
unstated specification. Unit labels alone do not prove dimensional correctness.

## Record observations, not estimates

The observer records actual start/end timestamps and pauses for each task,
success/failure, interventions and the participant's explanation verbatim where
provided. Machine solve time, setup time, human task time and report reading time
are separate measurements. A timed-out or abandoned task remains a failure with
its observed duration; do not omit it from a successful-only median.

Score against the frozen reference: candidate feasibility and objective,
correct capacity comparison, recognition of infeasibility, detection of the
omitted row, correct interpretation of evidence and correct replay lineage.
Record disagreement with the reference and investigate it before counting a
tool as wrong. Do not require identical decision vectors when optima are nonunique.

Primary outcomes are correct completed tasks, consequential modeling mistakes,
incorrect confidence in an unsupported claim, and observed time to a correct
decision. Report participant-level paired differences, task order, assistance
and failures. Do not pool repeated tasks as independent people. Publish a
productivity claim only after actual observations support it; a technical
installation gate cannot substitute for this study.

## Current status

The repository contains public educational data, a portable installation kit,
technical checks and blank feedback templates. Real operating data and
independent participant observations are still pending. No message to a
participant or professor is sent by these scripts.
