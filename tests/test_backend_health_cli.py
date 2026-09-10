from __future__ import annotations

import json

from solverpilot.cli.backend_health import build_report, main


def test_health_report_is_json_serializable_and_complete():
    report = build_report()
    encoded = json.dumps(report)
    assert encoded
    assert report["schema_version"] == "1.0"
    assert report["solverpilot_version"] == "0.1.0rc2"
    entries = report["backends"]
    names = {entry["backend"] for entry in entries}
    assert names == {
        "highspy-native",
        "osqp-native",
        "pyscipopt-native",
        "scipy-highs-ds",
        "scipy-highs-ipm",
        "scipy-highs-bridge",
        "scipy-slsqp-qp-bridge",
        "nlopt-slsqp-native",
        "casadi-osqp-bridge",
        "casadi-highs-bridge",
        "casadi-cbc-bridge",
        "bundled-osqp-capi",
        "bundled-highs-capi",
    }
    for entry in entries:
        assert entry["status"] in {
            "healthy",
            "unhealthy",
            "unavailable",
            "no_smoke_capability",
        }


def test_health_cli_default_exit_code_is_zero_when_no_backend_is_unhealthy(capsys):
    code = main(["--compact"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["solverpilot_version"] == "0.1.0rc2"
    assert code == 0, [(e["backend"], [(c.get("capability"), c.get("error")) for c in e.get("checks", [])], e.get("metadata", {}).get("cleanup_error")) for e in payload["backends"] if e["status"] == "unhealthy"]
