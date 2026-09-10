from __future__ import annotations
import argparse, hashlib, json, statistics
from collections import defaultdict
from pathlib import Path

TARGETS={'train':20,'validation':12,'test':16}

def group_id(name:str)->str:
    b=name.removesuffix('.mps.gz')
    if b.startswith('csched'): return 'csched-family'
    if b.startswith('markshare'): return 'markshare-family'
    if b.startswith('ic97_') or b.startswith('icir97_'): return 'ic97-family'
    return b

def h(s:str)->str: return hashlib.sha256(s.encode()).hexdigest()
def file_sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run1',type=Path,required=True); ap.add_argument('--run2',type=Path,required=True); ap.add_argument('--split-out',type=Path,required=True); ap.add_argument('--trainval-out',type=Path,required=True); args=ap.parse_args()
    a=json.loads(args.run1.read_text()); b=json.loads(args.run2.read_text())
    if a['cohort_signature']!=b['cohort_signature']: raise SystemExit('cohort signature mismatch')
    A={r['instance']:r for r in a['rows']}; B={r['instance']:r for r in b['rows']}
    if set(A)!=set(B) or len(A)!=48: raise SystemExit('expected identical 48-instance rows')
    groups=defaultdict(list)
    for name in sorted(A): groups[group_id(name)].append(name)
    items=sorted(groups.items(),key=lambda kv:h(kv[0]))
    remaining=TARGETS.copy(); alloc={k:[] for k in TARGETS}
    priority={'train':2,'validation':1,'test':0}
    for gid,names in items:
        size=len(names); candidates=[k for k in TARGETS if remaining[k]>=size]
        if not candidates: raise SystemExit(f'cannot place group {gid}: {remaining}')
        split=max(candidates,key=lambda k:(remaining[k]/TARGETS[k],remaining[k],priority[k]))
        alloc[split].append((gid,names)); remaining[split]-=size
    if any(remaining.values()): raise SystemExit(f'incomplete allocation {remaining}')
    rows=[]
    for split,gs in alloc.items():
        for gid,names in gs:
            for name in names: rows.append({'instance':name,'split':split,'group':gid})
    rows.sort(key=lambda r:r['instance'])
    split_payload={'schema':'optimind.m26.frozen_split.v1','cohort_signature':a['cohort_signature'],'source_run_sha256':[file_sha(args.run1),file_sha(args.run2)],'targets':TARGETS,'allocation_method':'sha256(group-id) order + deterministic largest-relative-deficit placement','records':rows}
    raw=json.dumps(split_payload,sort_keys=True,separators=(',',':')).encode(); split_payload['split_sha256']=hashlib.sha256(raw).hexdigest()
    args.split_out.parent.mkdir(parents=True,exist_ok=True); args.split_out.write_text(json.dumps(split_payload,indent=2,sort_keys=True))
    split_map={r['instance']:r['split'] for r in rows}
    # Intentionally omit ALL test timing/winner fields from this artifact.
    tv=[]
    for name in sorted(A):
        if split_map[name]=='test': continue
        ra,rb=A[name],B[name]
        for key in ('n','m','nnz','density','m_over_n'):
            if ra[key]!=rb[key]: raise SystemExit(f'feature drift {name} {key}')
        tv.append({'instance':name,'split':split_map[name],'n':ra['n'],'m':ra['m'],'nnz':ra['nnz'],'density':ra['density'],'m_over_n':ra['m_over_n'],
                   'ds_cost_s':float(statistics.median([ra['ds_median_cost_s'],rb['ds_median_cost_s']])),
                   'ipm_cost_s':float(statistics.median([ra['ipm_median_cost_s'],rb['ipm_median_cost_s']])),
                   'inspect_cost_s':float(statistics.median([ra['inspect_median_s'],rb['inspect_median_s']])),
                   'objective_mismatch': bool((ra.get('objective_abs_diff') or 0)>1e-7 or (rb.get('objective_abs_diff') or 0)>1e-7)})
    tv_payload={'schema':'optimind.m26.trainval_only.v1','split_sha256':split_payload['split_sha256'],'cohort_signature':a['cohort_signature'],'test_costs_present':False,'rows':tv}
    args.trainval_out.write_text(json.dumps(tv_payload,indent=2,sort_keys=True))
    print(json.dumps({'split_sha256':split_payload['split_sha256'],'counts':{k:sum(r['split']==k for r in rows) for k in TARGETS},'groups':len(groups),'trainval_rows':len(tv),'test_costs_present':False},indent=2))
if __name__=='__main__': main()
