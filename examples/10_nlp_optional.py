"""Optional smooth NLP example. Install with: pip install 'solverpilot[nlp]'"""
from solverpilot.model import Model
from solverpilot.nlp import CasadiIpoptBackend

backend = CasadiIpoptBackend()
if not backend.is_available():
    print("NLP example skipped: install solverpilot[nlp] and ensure the verified Ipopt path is available.")
else:
    m = Model("smooth-nlp")
    x = m.variable(lower=-4.0, upper=4.0, name="x")
    m.minimize((x - 1.25) ** 4 + 0.2 * (x - 1.25) ** 2)
    result = m.solve(x0=[0.0])

    print("backend_status:", result.backend_status)
    print("x:", result.x)
    print("objective:", result.objective_reported)
    print("local_optimal_candidate:", result.local_optimal_candidate)
    print("globally_proven:", result.globally_proven)

    assert result.validation.valid
    assert result.local_optimal_candidate
    assert not result.globally_proven
