from dataclasses import replace

import pytest

from solverpilot.experimental._repeat_quality import paired_repeat_losses
from solverpilot.experimental.learned_lp_v2 import calibrate_lp_guard_v2
from test_learned_lp_v2 import observations, fit, evaluate, decisions


@pytest.mark.parametrize('selected,baseline,losses', [
    ([False, True], [True, False], 1),
    ([True, True], [True, False], 0),
    ([[True, False], [False, True]], [[True, True], [True, False]], 2),
])
def test_losses_are_paired_not_compensated_by_another_repeat(selected, baseline, losses):
    assert paired_repeat_losses(selected, baseline) == losses


@pytest.mark.parametrize('selected,baseline', [([], []), ([True], [True, False]),
                                             ([1], [True]), ([None], [False]), (True, True)])
def test_invalid_pairing_fails_closed(selected, baseline):
    with pytest.raises(ValueError):
        paired_repeat_losses(selected, baseline)


def swapped_rows(split):
    return [replace(r, samples={**r.samples,
        'production': ((2.7, True), (2.7, False)),
        'a': ((.1, False), (.1, True)), 'b': ((.1, False), (.1, True))})
        for r in observations(split, production=True)]


def test_calibration_cannot_certify_a_leaf_with_equal_counts_but_lost_repeats():
    guard = calibrate_lp_guard_v2(fit(), swapped_rows('calibration'), min_families=2)
    assert not guard.certified_leaves


def test_v1_guard_also_rejects_swapped_repeat_losses():
    from test_lp_gain import calibration
    from solverpilot.experimental.lp_gain import calibrate_gain_guard
    _, model, rows = calibration()
    rows = [replace(r, samples={**r.samples, 'a': ((.01, False), (.01, True)),
                               'production': ((.3, True), (.3, False))}) for r in rows]
    assert not calibrate_gain_guard(model, rows).candidate_gains


def test_development_gate_records_each_paired_loss():
    report = evaluate(fit(), swapped_rows('test'))
    for scope in report['scopes'].values():
        production = scope['comparisons']['production']
        assert production['lost_verified_repeats'] == 12
        assert not production['gates']['no_lost_verified_solves']
        assert not scope['research_gate_passed']


def test_setup_charge_affects_loss_at_exact_cutoff():
    rows = [replace(r, samples={**r.samples, 'a': ((1., True),)*2,
        'b': ((1., True),)*2, 'production': ((2.9, True),)*2})
        for r in observations('test', production=True)]
    report = evaluate(fit(), rows, session_setup_s=2., reuse_count=10,
                      decisions=decisions(fit(), rows, overhead=0.))
    assert report['scopes']['cold']['comparisons']['production']['lost_verified_repeats'] == 0
    report = evaluate(fit(), rows, session_setup_s=2.0001, reuse_count=10,
                      decisions=decisions(fit(), rows, overhead=0.))
    assert report['scopes']['cold']['comparisons']['production']['lost_verified_repeats'] == 24
    assert report['scopes']['amortized']['comparisons']['production']['lost_verified_repeats'] == 0
