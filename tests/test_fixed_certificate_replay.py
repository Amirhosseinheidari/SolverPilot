import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from solverpilot import LinearProblem

DIRECTORY = Path(__file__).resolve().parents[1]/'benchmarks'
sys.path.insert(0, str(DIRECTORY))
try:
    SPEC = importlib.util.spec_from_file_location('replay_lp_certificates', DIRECTORY/'replay_lp_certificates.py')
    bench = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(bench)
finally:
    sys.path.pop(0)


def record():
    return {'name': 'sample', 'repeat': 0, 'presolve': False, 'mode': 'cold', 'step': 'baseline',
        'candidate': [1.], 'objective': 1., 'statistics': {'canonical_dual': [-1., 0.]},
        'evidence': {'primal_validated': True, 'dual_verified': True, 'gap_verified': True,
                     'certificate_verified': False}}


def problem():
    return LinearProblem.from_data(A=[[1.]], c=[1.], variable_lower=[0.],
        variable_upper=[2.], constraint_lower=[1.], constraint_upper=[np.inf])


@pytest.mark.parametrize('missing', ['candidate', 'objective', 'statistics'])
def test_missing_historical_data_is_retained_as_a_failed_observation(missing):
    p = problem(); old = record(); del old[missing]
    failed = bench.replay_one(old, {'reference': 1.}, p, p)
    assert failed['previously_verified'] and not failed['verified']
    assert 'error' in failed and len(failed['key']) == 5
    # A failed observation cannot prevent the next intended candidate's replay.
    assert bench.replay_one(record(), {'reference': 1.}, p, p)['verified']


def test_original_data_check_remains_a_required_gate():
    p = problem()
    row = bench.replay_one(record(), {'reference': 99.}, p, p)
    assert row['check']['verified']
    assert not row['verified'] and not row['original_candidate_check']['passed']


def test_backend_optimality_or_primal_flag_alone_does_not_count_as_old_proof():
    assert not bench.independently_verified({'backend_reported_optimal': True, 'primal_validated': True})
