from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
from time import perf_counter

OUT=Path(__file__).parent/'results'/'m13-worker-startup-audit.json'
SRC=str(Path(__file__).resolve().parents[1]/'src')
CODE="from solverpilot.backends import BundledOSQPCAPIBackend,BundledHighsCAPIBackend; print(BundledOSQPCAPIBackend().is_available(),BundledHighsCAPIBackend().is_available())"

def run(mode,timeout_s=12):
    cmd=[sys.executable]
    env=os.environ.copy()
    if mode=='no_site':
        cmd.append('-S'); env['PYTHONPATH']=SRC
    cmd.extend(['-c',CODE]); t=perf_counter()
    try:
        p=subprocess.run(cmd,text=True,capture_output=True,timeout=timeout_s,env=env)
        return {'mode':mode,'state':'completed','returncode':p.returncode,'wall_s':perf_counter()-t,'stdout':p.stdout.strip(),'stderr_tail':p.stderr[-1000:],'success':p.returncode==0}
    except subprocess.TimeoutExpired as e:
        return {'mode':mode,'state':'hard_timeout','wall_s':perf_counter()-t,'timeout_s':timeout_s,'stdout':(e.stdout or ''),'stderr':(e.stderr or ''),'success':False}

def main():
    payload={'benchmark':'M13 Python worker startup audit','note':'platform-specific; demonstrates why benchmark worker startup mode is protocol-versioned','runs':[run('normal'),run('no_site')]}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print(json.dumps(payload,indent=2))
    if not all(r.get('success',False) for r in payload['runs']):
        raise SystemExit(1)
if __name__=='__main__':main()
