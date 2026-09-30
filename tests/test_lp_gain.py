from dataclasses import replace
import pytest

from solverpilot.experimental.learned_lp import LPObservation, lp_features
from solverpilot.experimental.lp_gain import calibrate_gain_guard
from solverpilot.experimental.robust_lp import decide_robust_lp
from test_robust_lp import case


def calibration():
    p, model = case()
    rows = [LPObservation(str(i), 'group'+str(i), f'{i:064x}', 'train', 'env', lp_features(p),
             {'a': ((.1, True), (.1, True)), 'b': ((.2, True), (.2, True)),
              'production': ((.3, True), (.3, True))}) for i in range(4)]
    return p, model, rows


def test_positive_gain_requires_all_binding_checks():
    p, model, rows = calibration()
    guard = calibrate_gain_guard(model, rows, overhead_limit_s=.1)
    assert guard.candidate_gains[0][0] == 'a'
    def decision(**kwargs):
        return decide_robust_lp(p, model, environment_id='env', available=('a','b'), gain_guard=guard, **kwargs)
    assert decision(cutoff_s=2.).candidate == 'a'
    assert decision(cutoff_s=1.).candidate == 'production'
    assert decision(cutoff_s=2., setup_elapsed_s=.2).candidate == 'production'
    for wrong in (replace(guard, model_sha256='0'*64), replace(guard, implementation_sha256='0'*64)):
        assert not wrong.permits(model, 'a', elapsed_s=0., cutoff_s=2.)


@pytest.mark.parametrize('kind', ['late', 'failure', 'slower', 'too_few'])
def test_no_gain_or_lost_verified_repeat_blocks_switch(kind):
    _, model, rows = calibration()
    if kind == 'too_few': rows=rows[:3]
    else:
        samples=dict(rows[0].samples)
        samples['a']={'late': ((2.,True),)*2, 'failure': ((.01,False),)*2,
                      'slower': ((.4,True),)*2}[kind]
        rows[0]=replace(rows[0], samples=samples)
    assert not calibrate_gain_guard(model,rows).candidate_gains


def test_heldout_or_mixed_environment_calibration_rejected():
    _, model, rows=calibration()
    for options in ({'split':'test'}, {'environment_id':'other'}):
        changed=[replace(r,**options) for r in rows]
        with pytest.raises(ValueError): calibrate_gain_guard(model,changed)


def test_saved_guard_roundtrip_and_tamper_rejection(tmp_path):
    import json
    from solverpilot.experimental.lp_gain import LPGainGuard
    _,model,rows=calibration()
    guard=calibrate_gain_guard(model,rows)
    path=tmp_path/'guard.json'
    payload=guard.payload();path.write_text(json.dumps(payload))
    assert LPGainGuard.load(path)==guard
    payload['cutoff_s']=20.;path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='digest'): LPGainGuard.load(path)
