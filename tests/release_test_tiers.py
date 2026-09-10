from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'tests'/'test-tiers.json').read_text(encoding='utf-8'))

def _all_test_files(): return [str(p.relative_to(ROOT)) for p in sorted((ROOT/'tests').rglob('test_*.py'))]
def tier_targets(tier):
    native=list(CONFIG['native_files']); evidence=list(CONFIG['evidence_nodes'])
    if tier=='native': return native,[]
    if tier=='evidence': return evidence,[]
    core=[p for p in _all_test_files() if p not in set(native)]
    return core,['-m','not evidence']
def _summary(path):
    root=ET.parse(path).getroot(); suites=root.findall('testsuite') if root.tag=='testsuites' else [root]
    out={'tests':0,'failures':0,'errors':0,'skipped':0}
    for suite in suites:
        for key in out: out[key]+=int(suite.attrib.get(key,0))
    out['passed']=out['tests']-out['failures']-out['errors']-out['skipped']; return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('tier',choices=('core','native','evidence')); ap.add_argument('--junitxml',type=Path,required=True); ap.add_argument('--json',type=Path); ap.add_argument('--shard-index',type=int,default=0); ap.add_argument('--shard-count',type=int,default=1); args=ap.parse_args()
    if args.shard_count<1 or not 0<=args.shard_index<args.shard_count: ap.error('shard-index must be in [0, shard-count)')
    targets,extra=tier_targets(args.tier); targets=[t for i,t in enumerate(targets) if i%args.shard_count==args.shard_index]
    cmd=[sys.executable,'-m','pytest','-q',*targets,*extra,f'--junitxml={args.junitxml}']; env=os.environ.copy(); env['PYTHONPATH']=str(ROOT/'src')+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else ''); env.setdefault('TERM','xterm')
    args.junitxml.parent.mkdir(parents=True,exist_ok=True); cp=subprocess.run(cmd,cwd=ROOT,env=env)
    payload={'tier':args.tier,'returncode':cp.returncode,'targets':targets,'shard':{'index':args.shard_index,'count':args.shard_count},'summary':_summary(args.junitxml) if args.junitxml.exists() else None}
    if args.json: args.json.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps(payload,indent=2,sort_keys=True)); return cp.returncode
if __name__=='__main__': raise SystemExit(main())
