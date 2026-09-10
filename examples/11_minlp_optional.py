"""Optional certified convex binary MINLP example."""
from solverpilot.model import Model
from solverpilot.nlp import CasadiIpoptBackend

if not CasadiIpoptBackend().is_available():
    print("MINLP example skipped: install solverpilot[minlp] and ensure the verified NLP path is available.")
else:
    m = Model("binary-minlp")
    x = m.variable(lower=0.0, upper=3.0, name="x")
    z = m.binary(name="z")
    m.minimize((x - 2.0) ** 2 + 0.2 * (1 - z))

    result = m.compile().solve()
    print("algorithm:", result.algorithm)
    print("status:", result.status)
    print("objective:", result.objective)
    print("globally_proven:", result.globally_proven)
    print("proof_scope:", result.proof_scope)
    print("independently_verified_global:", result.independently_verified_global)

    assert result.globally_proven
    assert result.proof_scope == "solver_certified_under_convexity_assumptions"
    assert not result.independently_verified_global
    assert abs(result.objective) < 1e-5
