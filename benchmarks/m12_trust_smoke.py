from __future__ import annotations
import json
from pathlib import Path
import tempfile

from solverpilot.benchmark import RunSpec, run_benchmark, summarize_jsonl
from solverpilot.benchmark.registry import benchmark_registry
from solverpilot.benchmark.policy import evaluate_policy_map

OUT = Path(__file__).parent / 'results' / 'm12-trust-smoke.json'


def mps(name: str, demand: int, cap: int) -> str:
    return f'''NAME          {name}\nROWS\n N  COST\n G  DEMAND\n L  CAP\nCOLUMNS\n    X1        COST       1        DEMAND     1\n    X1        CAP        1\n    X2        COST       2        DEMAND     1\n    X2        CAP        2\nRHS\n    RHS1      DEMAND     {demand}        CAP        {cap}\nBOUNDS\n UP BND1      X1         5\n UP BND1      X2         8\nENDATA\n'''


def main() -> None:
    with tempfile.TemporaryDirectory(prefix='optimind-m12-trust-') as td:
        root = Path(td)
        data = root / 'data'; data.mkdir()
        names=[]
        # Small deterministic LPs. The smoke validates benchmark accounting/trust,
        # not general solver performance.
        for i, (d, c) in enumerate([(1,4),(2,5),(3,6),(4,7),(5,9),(6,11),(7,13),(8,15)]):
            name=f'trust-{i}.mps'; names.append(name)
            (data/name).write_text(mps(f'TRUST{i}', d, c), encoding='utf-8')
        manifest=root/'trust.test'; manifest.write_text('\n'.join(names)+'\n', encoding='utf-8')
        output=root/'rows.jsonl'
        backends=('scipy-highs-ds','scipy-highs-ipm','@auto')
        # In-process on purpose: this smoke is about accounting/trust mechanics.
        # Hard process isolation is covered separately by unit/integration tests.
        spec=RunSpec(dataset_dir=data, manifest=manifest, reference=None, backends=backends,
                     output_jsonl=output, repetitions=2, hard_timeout_s=None,
                     thread_env_limit=None, solver_threads=None, seed=12013)
        run=run_benchmark(spec, registry=benchmark_registry(backends))
        summary=summarize_jsonl([output], cutoff_s=1.0, cost_field='wall_s', bootstrap_draws=2000, bootstrap_seed=12013)
        rows=[json.loads(x) for x in output.read_text(encoding='utf-8').splitlines() if x.strip()]

        # Mixed-environment timing must be rejected by default.
        mixed=[dict(r) for r in rows]
        mixed[0]['environment_id']='deliberately-different-environment'
        mixed_rejected=False
        try:
            from solverpilot.benchmark.summary import summarize_rows
            summarize_rows(mixed, cutoff_s=1.0, cost_field='wall_s', bootstrap_draws=100)
        except ValueError as exc:
            mixed_rejected='multiple benchmark environments' in str(exc)

        # Fixed-SBS external policy must reconstruct SBS when overhead is zero.
        sbs=summary['sbs_solver']
        policy={inst:{'backend':sbs,'overhead_s':0.0} for inst in sorted({r['instance'] for r in rows})}
        policy_eval=evaluate_policy_map(rows, policy, cutoff_s=1.0, cost_field='wall_s', bootstrap_draws=500, bootstrap_seed=12013)

        payload={
            'benchmark':'M12 benchmark trust/accounting smoke',
            'scope_note':'Synthetic short LP smoke for harness trust mechanics only; not solver-performance evidence.',
            'run':run,
            'rows':len(rows),
            'expected_rows':len(names)*len(backends)*2,
            'summary':summary,
            'mixed_environment_rejected':mixed_rejected,
            'fixed_sbs_policy_ratio':policy_eval['policy_metrics']['policy_to_sbs_ratio_bootstrap_95pct']['point'],
            'fixed_sbs_policy_ci':policy_eval['policy_metrics']['policy_to_sbs_ratio_bootstrap_95pct'],
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
        print(json.dumps({
            'rows':payload['rows'], 'expected_rows':payload['expected_rows'],
            'sbs_solver':summary['sbs_solver'], 'sbs_cost':summary['sbs_cost'], 'vbs_cost':summary['vbs_cost'],
            'auto':summary['auto_policy_metrics'], 'mixed_environment_rejected':mixed_rejected,
            'fixed_sbs_policy_ratio':payload['fixed_sbs_policy_ratio'],
        }, indent=2))


if __name__ == '__main__':
    main()
