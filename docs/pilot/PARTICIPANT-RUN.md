# Run the first observed production-workflow session

This is an execution checklist for the prospective
[comparison protocol](COMPARISON-PROTOCOL.md). It records real observations when
a participant and permitted operating data are available. Neither recruitment
nor a completed participant session is implied by this document.

Computational benchmarks do not supply participant observations or confirm the
business rules of an operating production process. Feedback files remain blank
until a real session is observed.

1. Identify one willing participant who regularly makes the relevant production
   decision. Record an alias, role, prior optimization/Python experience and
   permission for the intended use of observations. Arrange the session outside
   these scripts; no invitation is sent automatically.
2. Obtain one permitted, deidentified planning period from that participant:
   products, resource coefficients and capacities, contribution margins, demand
   bounds and commitments. Record units, source period and the owner of each
   input. Ask the participant to identify missing business rules before freezing
   the task; distinguish their stated rules from analyst assumptions.
3. Independently assemble and check the mathematical model and reference answers
   before the session. Retain the original table and the model mapping. Include
   a capacity change, inconsistent commitments and an intentionally omitted row
   in matched variants, as specified in the protocol. Keep the answer key apart
   from participant instructions. If operating data are unavailable, run only
   an explicitly labelled educational usability rehearsal.
4. Freeze hashes of the wheel, dependencies, source table, task variants,
   references, protocol and comparator wrapper. Use the same native HiGHS
   version, options and documented checks. Include the comparator's persistent
   updates and export support where the workflow needs them. Record wrapper
   construction time separately from participant task time.
5. Copy `comparison-feedback-template.json` into a new session directory. Set
   `template_only` to false only after recording actual observations. Leave
   unavailable answers null. Record the assigned tool/variant order and training
   time before beginning. Alternate order across participants when more than
   one participates; tasks within one person are not independent participants.
6. Observe all six protocol tasks. Record start/end times, pauses, interventions,
   correct completion, abandoned tasks, reference disagreements, unsupported
   claims accepted and the participant's own explanation. Retain every failure.
   Save solve evidence separately from the human feedback; a valid evidence
   bundle cannot establish that the person understood the result.
7. Ask whether the participant would use the workflow again and why. Have the
   observer review scoring against the frozen answers. Report paired results
   at participant level, assistance and missing observations. One initial
   session establishes usability findings, not an industrial productivity or
   population-wide speed advantage.

The first production-pilot acceptance gate is a complete observed session with
permitted operating inputs, traceable model mapping, frozen references and
honest failure records. If any element is missing, name it explicitly; passing
the machine benchmark cannot fill it in.
