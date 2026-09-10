import pytest
pytest.importorskip("casadi", reason="optional CasADi extra not installed in core CI")

from solverpilot.nlp import conform_casadi_ipopt_backend

def test_p7_ipopt_conformance_report_passes():
    r=conform_casadi_ipopt_backend(); assert r.available; assert r.passed; assert all(c.passed for c in r.checks)
