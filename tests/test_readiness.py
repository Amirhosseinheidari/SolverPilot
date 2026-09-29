import json
from solverpilot.runtime.readiness import readiness_report
from solverpilot.cli.doctor import main


def test_passive_identity_never_claims_execution_qualification(monkeypatch):
    import solverpilot
    monkeypatch.setattr(solverpilot,'solve',lambda *a,**k: (_ for _ in ()).throw(AssertionError('must not solve')))
    report=readiness_report()
    assert len(report['source_sha256'])==64
    assert not report['execution_qualified'] and not report['automatic_learned_routing_enabled']
    assert not report['automatic_gpu_routing_enabled']
    capabilities={row['name']:row for row in report['capabilities']}
    assert capabilities['core_lp_milp_convex_qp']['included_in_published_0_4']
    assert not capabilities['guarded_learned_lp']['included_in_published_0_4']
    assert capabilities['guarded_learned_lp']['maturity']=='experimental'


def test_doctor_cli(capsys):
    assert main(['--compact'])==0
    report=json.loads(capsys.readouterr().out)
    assert report['code_lineage']=='development after published 0.4'
