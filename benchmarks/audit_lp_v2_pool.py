"""Admission-only Netlib inventory. Never downloads models or calls a solver.

Failing capacity preserves the remaining models for a preregistered study with
enough independent calibration and test families. Source notes inform aliases;
family independence still needs human/domain audit.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json

from lp_public_gate import POLICY
from qualify_public_reuse import CAPS, collect_prior, discover_prior, metadata_cases, sha


def audit_pool(workspace, metadata, output):
    workspace, metadata, output = map(Path, (workspace, metadata, output))
    if output.exists():
        raise FileExistsError('refusing to overwrite a prior admission audit')
    paths, directories = discover_prior(workspace)
    prior = collect_prior(paths, directories)
    raw = metadata.read_bytes()
    rows = []
    for case in metadata_cases(raw.decode('utf-8')):
        reason = ('consumed_name' if case['name'] in prior['names'] else
                  'consumed_family' if case['family'] in prior['families'] else
                  'structural_cap' if any(case[k] > v for k, v in CAPS.items()) else None)
        rows.append({**case, 'excluded_reason': reason})
    remaining = [r for r in rows if r['excluded_reason'] is None]
    families = sorted({r['family'] for r in remaining})
    before_caps = sorted({r['family'] for r in rows if r['excluded_reason'] in {None, 'structural_cap'}})
    required = POLICY['min_calibration_families']+POLICY['min_test_families']
    report = dict(schema='solverpilot.lp-v2-pool-audit.v1', captured_at=datetime.now(timezone.utc).isoformat(),
        source_url='https://www.netlib.org/lp/data/readme', metadata_sha256=sha(raw),
        source_hashes={name:sha((Path(__file__).parent/name).read_bytes()) for name in
            ('audit_lp_v2_pool.py', 'lp_public_gate.py', 'public_lp_families.py', 'qualify_public_reuse.py')},
        policy=dict(POLICY), structural_caps=dict(CAPS), prior=prior, candidates=rows,
        fresh_families_within_caps=families, fresh_families_before_caps=before_caps,
        required_fresh_calibration_and_test_families=required,
        capacity_sufficient=len(families) >= required,
        missing_families=max(0, required-len(families)), solves_performed=0,
        test_consumed=False, production_authorized=False,
        scope='Availability within the listed consumed registers and conservative aliases; not proof of exhaustive semantic independence',
        wood_family='Conservative inference from shared contributor/model name, not a Netlib assertion of equivalence')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('capacity_sufficient', 'fresh_families_within_caps',
        'fresh_families_before_caps', 'missing_families', 'solves_performed', 'test_consumed')}))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--metadata', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    audit_pool(args.workspace, args.metadata, args.output)
