from dataclasses import dataclass, replace

import pytest

from solverpilot import LinearProblem, execute, solve
from solverpilot.backends.base import BackendSolveResult
from solverpilot.capabilities import BackendManifest, Capability, SupportLevel
from solverpilot.runtime.evidence import evidence_bundle, verify_evidence_bundle
from solverpilot.runtime.manifest import _seal
from solverpilot.runtime.unified import summarize


@dataclass
class WitnessBackend:
    claim: object
    status: str = 'infeasible'
    x: object = None

    @property
    def manifest(self):
        return BackendManifest(name='witness-fixture', version=None,
                               capabilities={Capability.LP: SupportLevel.NATIVE})

    def is_available(self):
        return True

    def solve(self, problem):
        return BackendSolveResult(self.status, self.x,
            raw_statistics={'termination_certificate': self.claim})


def problem(upper=2.):
    return LinearProblem.from_data(A=[[1.]], c=[1.], variable_lower=[0.],
        variable_upper=[upper], constraint_lower=[3.], constraint_upper=[float('inf')])


@pytest.mark.parametrize('dual', [[0., 0.], [float('nan'), 0.], [10**400, 0.]])
def test_backend_verified_flag_cannot_promote_invalid_witness(dual):
    p = problem(4.)  # x=3 is feasible
    r = execute(p, WitnessBackend({'problem_data_hash': p.data_hash, 'kind': 'infeasible',
                                  'verified': True, 'dual': dual}))
    assert r.termination_certificate is None
    assert summarize(r).termination_evidence == 'not_established'
    assert verify_evidence_bundle(evidence_bundle(p, r))['summary']['termination_evidence'] == 'not_established'


@pytest.mark.parametrize('claim', [None, [], {'kind': []}, {'kind': 'infeasible', 'problem_data_hash': [1, 2]}])
def test_malformed_backend_certificate_metadata_does_not_crash_execution(claim):
    assert execute(problem(), WitnessBackend(claim)).termination_certificate is None


def test_valid_backend_witness_is_rechecked_without_trusting_its_flag():
    p = problem()
    r = execute(p, WitnessBackend({'problem_data_hash': p.data_hash, 'kind': 'infeasible',
                                  'verified': False, 'dual': [-1., 1.]}))
    assert r.termination_certificate.verified
    assert summarize(r).termination_evidence == 'independent_numerical_termination_certificate'
    payload = verify_evidence_bundle(evidence_bundle(p, r))
    assert payload['run']['termination_check']['problem_data_hash'] == p.data_hash
    assert summarize(replace(r, raw_statistics={})).termination_evidence != 'not_established'


def test_owned_recovery_and_unbounded_witness_remain_exportable():
    p = problem()
    recovered = solve(p, backend='scipy-highs-ds', certificate_recovery=2.)
    assert recovered.termination_certificate.verified
    verify_evidence_bundle(evidence_bundle(p, recovered))
    ray_problem = LinearProblem.from_data(A=[[1.]], c=[-1.], variable_lower=[0.],
        variable_upper=[float('inf')], constraint_lower=[0.], constraint_upper=[float('inf')])
    ray = execute(ray_problem, WitnessBackend({'problem_data_hash': ray_problem.data_hash,
        'kind': 'unbounded', 'verified': False, 'point': [0.], 'direction': [1.]}, 'unbounded'))
    assert ray.termination_certificate.verified
    verify_evidence_bundle(evidence_bundle(ray_problem, ray))
    origin = execute(ray_problem, WitnessBackend({'problem_data_hash': ray_problem.data_hash,
        'kind': 'unbounded', 'point': [0.], 'direction': [1.]}, 'unbounded', [0.]))
    assert origin.validation.valid and origin.termination_certificate.verified
    assert summarize(origin).feasible
    verify_evidence_bundle(evidence_bundle(ray_problem, origin))


def test_feasible_candidate_contradicts_backend_infeasibility():
    p = problem(4.)
    r = execute(p, WitnessBackend({}, x=[3.]))
    assert r.status.value == 'error' and r.validation.valid
    assert r.termination_certificate is None
    assert 'contradict' in r.trace.warnings[-1]
    verify_evidence_bundle(evidence_bundle(p, r))


@pytest.mark.parametrize('mutation', ['hash', 'kind', 'shape'])
def test_witness_must_match_model_status_and_shape(mutation):
    p = problem()
    claim = {'problem_data_hash': p.data_hash, 'kind': 'infeasible', 'verified': True, 'dual': [-1., 1.]}
    claim[{'hash': 'problem_data_hash', 'kind': 'kind', 'shape': 'dual'}[mutation]] = {
        'hash': 'a'*64, 'kind': 'unbounded', 'shape': [-1.] }[mutation]
    assert execute(p, WitnessBackend(claim)).termination_certificate is None


@pytest.mark.parametrize('label', ['independent_exact_infeasibility',
                                  'independent_numerical_termination_certificate'])
def test_resealed_record_cannot_add_unbacked_termination_claim(label):
    p = problem(4.)
    payload = evidence_bundle(p, solve(p, backend='scipy-highs-ds'))
    payload['summary']['termination_evidence'] = label
    payload['run']['summary']['termination_evidence'] = label
    payload['run'] = _seal(payload['run'])
    with pytest.raises(ValueError):
        verify_evidence_bundle(_seal(payload))
