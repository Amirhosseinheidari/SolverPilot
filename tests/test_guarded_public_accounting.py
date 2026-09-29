"""The public gate must count late failures and objective contradictions."""
import importlib
from pathlib import Path
import pytest


@pytest.mark.parametrize('fault', [None, 'late', 'objective', 'missing'])
def test_public_gate_complete_accounting(monkeypatch, fault):
    pytest.importorskip('highspy')
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'benchmarks'))
    module=importlib.import_module('qualify_guarded_public')
    cases=[{'file':str(i),'group':str(i),'reference':1.} for i in range(24)]
    rows=[{'instance':c['file'],'strategy':s,'repeat':repeat,'verified':True,'objective':1.,
           'wall_s':1.,'api_wall_s':.2 if s=='guarded' else .3,'route':{'candidate':'highs-simplex'}}
          for c in cases for s in module.STRATEGIES for repeat in range(2)]
    if fault=='missing':
        with pytest.raises(ValueError,match='incomplete'):module.summarize(rows[:-1],cases)
        return
    if fault=='late':rows[0]['api_wall_s']=2.01
    if fault=='objective':rows[0]['objective']=2.
    result=module.summarize(rows,cases)
    assert result['public_gate_passed'] is (fault is None)
    assert not result['automatic_production_routing_enabled']
    if fault=='late':
        assert result['verified_within_budget']['guarded']==47
        assert not result['gates']['production_successes']
