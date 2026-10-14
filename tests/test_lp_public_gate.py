"""Public evidence must measure routes, retain failures and preserve fresh data."""
import copy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

BENCH = Path(__file__).resolve().parents[1]/'benchmarks'
sys.path.insert(0, str(BENCH))
gate = importlib.import_module('lp_public_gate')
runner = importlib.import_module('qualify_lp_v2')
sys.path.pop(0)


@pytest.fixture(autouse=True)
def benchmark_import_path(monkeypatch):
    # Sibling benchmark imports are intentionally deferred by the CLI modules.
    monkeypatch.syspath_prepend(str(BENCH))


def measured():
    cases = [dict(name=f'n{i}', family=f'f{i}', data_hash=f'{i:064x}', reference=1.) for i in range(20)]
    rows = [dict(**case, strategy=strategy, repeat=repeat, split='test', verified=True,
                 objective=1., objective_consistent=True, api_wall_s=.2 if strategy == 'v2' else 1.,
                 preparation_s=0., route=dict(candidate='highs-simplex', reason='in_support',
                 leaf_id=0, learned_reason='learned_tree_v2', model_sha256='m', gain_guard_sha256='g'))
            for case in cases for repeat in range(2) for strategy in runner.strategies('test')]
    return cases, rows


def evaluate(cases, rows):
    return gate.evaluate_actual_routes(rows, cases, dict(repeats=2, cutoff_s=2., reuse_count=20),
        baseline='highs-ipm', model_sha256='m', guard_sha256='g')


def test_complete_actual_routes_pass_only_research_gates():
    cases, rows = measured()
    report = evaluate(cases, rows)
    assert all(s['qualified'] for s in report['scopes'].values())
    assert set(report['scopes']['cold']['comparisons']) == {'default', 'production', 'highs-ipm'}
    assert report['actual_learned_switches'] == 40
    assert not report['production_authorized']


@pytest.mark.parametrize('defect', ['fallback', 'single_solver', 'binding', 'objective', 'late', 'family', 'missing', 'duplicate'])
def test_false_success_cannot_qualify(defect):
    cases, rows = measured()
    if defect == 'fallback':
        for row in rows:
            row['route']['candidate'] = 'production'
    elif defect == 'single_solver':
        for row in rows:
            row['route'].update(candidate='highs-ipm', learned_reason='training_baseline')
    elif defect == 'binding':
        rows[0]['route']['model_sha256'] = 'wrong'
    elif defect == 'objective':
        rows[0]['objective'] = 99.
    elif defect == 'late':
        rows[0]['api_wall_s'] = 2.001
    elif defect == 'family':
        rows[0]['family'] = 'other'
    elif defect == 'missing':
        rows.pop()
    else:
        rows.append(copy.deepcopy(rows[0]))
    if defect in {'family', 'missing', 'duplicate'}:
        with pytest.raises(ValueError):
            evaluate(cases, rows)
    else:
        assert not evaluate(cases, rows)['scopes']['cold']['qualified']


def test_setup_charged_before_deadline_separately_for_cold_and_amortized():
    cases, rows = measured()
    for row in rows:
        if row['strategy'] == 'v2':
            row['preparation_s'] = 2.
    scopes = evaluate(cases, rows)['scopes']
    assert scopes['cold']['verified_repeats'] == 0
    assert not scopes['cold']['qualified']
    assert scopes['amortized']['qualified']


def test_unequal_family_sizes_have_equal_weight():
    cases, rows = measured()
    # Three instances share one family; its slower cost is still one group.
    for case in cases[:3]:
        case['family'] = 'shared'
    for row in rows:
        if row['name'] in {'n0', 'n1', 'n2'}:
            row['family'] = 'shared'
            if row['strategy'] == 'v2':
                row['api_wall_s'] = 1.
    report = evaluate(cases, rows)
    assert report['scopes']['cold']['comparisons']['production']['family_weighted_ratio'] == pytest.approx((1+17*.2)/18)
    assert not report['scopes']['cold']['gates']['enough_test_families']


def test_loaded_source_mismatch_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='executed'):
        gate.check_loaded_source(SimpleNamespace(__path__=[str(tmp_path/'stale')]), tmp_path/'expected')


def test_mixed_module_origin_is_rejected(tmp_path, monkeypatch):
    import solverpilot
    monkeypatch.setitem(sys.modules, 'solverpilot.fake_external',
        SimpleNamespace(__file__=str(tmp_path/'foreign.py')))
    with pytest.raises(ValueError, match='mixed'):
        gate.check_loaded_source(solverpilot, Path(solverpilot.__file__).parent)


def test_failed_learned_switches_do_not_count_as_qualified_coverage():
    cases, rows = measured()
    for row in rows:
        if row['strategy'] == 'v2':
            row['verified'] = False
    report = evaluate(cases, rows)
    assert report['actual_learned_switches'] == 40
    assert report['scopes']['cold']['successful_switch_families'] == 0
    assert not report['scopes']['cold']['gates']['actual_learned_switches']


def test_public_minimum_family_capacity_before_parsing():
    def case(i):
        return dict(name=f'n{i}', family=f'f{i}', format='mps', reference=1., source='external',
                    source_url='https://example.org/model', reference_url='https://example.org/reference')
    cases = dict(train=[case(i) for i in range(4)], calibration=[case(i) for i in range(4,8)],
                 test=[case(i) for i in range(8,27)])
    with pytest.raises(ValueError, match='test needs 20.*19'):
        gate.public_preflight({'consumed_lists': ['history.json']}, cases, dict(repeats=2))


def public_demo(tmp_path):
    path = runner.generate_development_demo(tmp_path/'corpus')
    manifest = json.loads(path.read_text())
    manifest.update(schema=runner.PUBLIC_SCHEMA, scope='public', consumed_lists=['consumed.json'])
    (path.parent/'consumed.json').write_text('{}')
    return path, manifest


def test_too_few_public_families_rejected_before_parsing(tmp_path):
    path, manifest = public_demo(tmp_path)
    def forbidden(*args):
        pytest.fail('preflight must preserve all unsolved public inputs')
    with pytest.raises(ValueError, match='materialized|families'):
        runner.validate_manifest(manifest, path.parent, loader=forbidden)


@pytest.mark.parametrize('old,new', [('perold','pilot4'), ('bp1','25fv47'), ('powell','fffff800'),
    ('gub','czprob'), ('wood1p','woodw'), ('sierra','gfrd-pnc'), ('scagr7','scfxm1'), ('maros','maros-r7')])
def test_consumed_netlib_alias_rejected_before_parsing(tmp_path, old, new):
    path, manifest = public_demo(tmp_path)
    (path.parent/'consumed.json').write_text(json.dumps({'names': [old]}))
    case = manifest['splits']['calibration'][0]
    case.update(name='netlib-'+new, source='netlib')
    with pytest.raises(ValueError, match='consumed.*alias'):
        runner.validate_manifest(manifest, path.parent, loader=lambda *a: pytest.fail('must reject before parsing'))


@pytest.mark.parametrize('failure', ['no_leaves', 'baseline_only', 'preparation'])
def test_public_calibration_failures_never_consume_test(tmp_path, monkeypatch, failure):
    # Isolate lifecycle from schema validation: those checks are tested above.
    path = runner.generate_development_demo(tmp_path/'corpus')
    original = json.loads(path.read_text())
    config, cases, problems, consumed, calls = runner.validate_manifest(original, path.parent)
    original.update(schema=runner.PUBLIC_SCHEMA, scope='public')
    path.write_text(json.dumps(original))
    monkeypatch.setattr(runner, 'validate_manifest', lambda *a, **k: (config, cases, problems, consumed, calls))
    monkeypatch.setattr(runner, 'source_hashes', lambda: {'fixed': 'hash'})
    class Artifact:
        baseline = 'highs-ipm'
        certified_leaves = () if failure == 'no_leaves' else (SimpleNamespace(
            candidate='highs-ipm' if failure == 'baseline_only' else 'highs-simplex'),)
        def save(self, where):
            where.write_text('{}')
    class Runtime:
        environment = {}
        environment_id = 'fake'
        def features(self, problem): return (0.,)*13
        def fit(self, *a): return Artifact()
        def calibrate(self, *a): return Artifact()
        def prepare(self, *a): raise RuntimeError('broken session')
        def execute(self, problem, strategy, *a):
            assert all(problem is not problems[c['name']] for c in cases['test'])
            return dict(verified=False)
    output = tmp_path/'run'
    summary = runner.run_protocol(path, output, runtime=Runtime())
    assert summary['test_evaluations'] == 0 and not summary['test_consumed']
    assert summary['total_api_calls'] == 32
    assert not (output/'test-outcomes.json').exists()
