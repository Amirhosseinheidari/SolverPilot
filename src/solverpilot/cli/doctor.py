"""Passive package identity and feature readiness report."""
import argparse
import json
from solverpilot.runtime.readiness import readiness_report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compact',action='store_true')
    args=parser.parse_args(argv)
    print(json.dumps(readiness_report(),indent=None if args.compact else 2,sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
