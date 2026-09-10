import json

from solverpilot.cli.capabilities import main


def test_capabilities_cli_projection(capsys):
    assert main(["--backend", "scipy-highs-ds"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["backend"] == "scipy-highs-ds"
    assert payload[0]["manifest"]["schema_version"] == "2.0"


def test_capabilities_cli_verified_p0_projection(capsys):
    assert main(["--verify", "--p0-schema", "--backend", "scipy-highs-ds"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["conformance_passed"] is True
    assert payload[0]["manifest"]["problem_classes"]["lp"] == "supported"
