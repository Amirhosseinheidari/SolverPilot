"""Evidence-preserving explanations across the supported result families."""
from solverpilot.runtime.unified import summarize
from solverpilot.runtime.manifest import json_value


def explain_any(result):
    summary=summarize(result)
    descriptions={
        'independent_numerical_bound':'Original-coordinate numerical primal/dual evidence was checked.',
        'independent_exact_certificate':'The exact certificate establishes the stated zero-gap optimum.',
        'solver_reported':'Optimality is reported by the solver or scoped orchestrator; independent optimality is not established.',
        'local_candidate':'A local continuous candidate passed validation and local stationarity checks; no global proof is claimed.',
        'not_established':'No optimality evidence is established.',
    }
    raw=getattr(result,'raw_statistics',None) or {}
    return json_value({'schema':'solverpilot.common-explanation.v1','summary':summary,
                      'interpretation':descriptions[summary.optimality],
                      'proof_scope':getattr(result,'proof_scope','none'),
                      'budget':raw.get('call_budget',{}),
                      'proof_policy':raw.get('proof_policy'),
                      'verification_reason':summary.optimality_reason})
