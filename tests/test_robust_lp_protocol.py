import importlib
from pathlib import Path

import pytest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'benchmarks'))
    # Import remains side-effect free: native backends are probed only on execution.
    return importlib.import_module('qualify_robust_lp')


def cases(prefix):
    return [dict(file=f'{prefix}{i}.mps', name=f'{prefix}{i}', group=f'{prefix}g{i}',
                 data_hash=f'{i:064x}' if prefix == 'train' else f'{100+i:064x}', reference=0.)
            for i in range(24)]


@pytest.mark.parametrize('key', ['name', 'group', 'data_hash'])
def test_public_split_rejects_identity_family_and_hash_leakage(runner, key):
    training, heldout = cases('train'), cases('test')
    runner.verify_split(training, heldout)
    heldout[0][key] = training[0][key]
    with pytest.raises(ValueError, match='overlap'):
        runner.verify_split(training, heldout)


def test_training_penalizes_failed_and_late_repetitions_and_rejects_missing(runner):
    cohort = cases('train')
    rows = [dict(instance=c['file'], strategy=s, repeat=r, features=[0.]*8,
                 verified=True, wall_s=1., api_wall_s=.1, objective=0.)
            for c in cohort for s in runner.CANDIDATES for r in range(2)]
    rows[0]['api_wall_s'] = 3.
    rows[1]['verified'] = False
    fitted = runner.training_rows(rows, cohort, 'env')
    assert fitted[0].samples[runner.CANDIDATES[0]] == ((1., False), (1., False))
    with pytest.raises(ValueError, match='incomplete'):
        runner.training_rows(rows[:-1], cohort, 'env')
    rows[2]['objective'] = 1.
    with pytest.raises(ValueError, match='objective mismatch'):
        runner.training_rows(rows, cohort, 'env')
