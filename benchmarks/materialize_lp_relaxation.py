"""Prepare a continuous public model; audit its family/provenance separately."""
import argparse
import json
from pathlib import Path
from solverpilot.benchmark.relaxations import materialize_lp_relaxation
from solverpilot.runtime.manifest import json_value


def main():
    p=argparse.ArgumentParser(); p.add_argument('source',type=Path); p.add_argument('destination',type=Path)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--time-limit',type=float,default=30.)
    args=p.parse_args()
    if args.output.exists(): raise SystemExit('refusing to overwrite prior evidence')
    report=materialize_lp_relaxation(args.source,args.destination,reference_time_s=args.time_limit)
    args.output.write_text(json.dumps(json_value(report),indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'reference_verified':report['reference_verified'],'data_hash':report['data_hash']}))


if __name__=='__main__': main()
