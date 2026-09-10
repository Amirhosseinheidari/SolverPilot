from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np

LAMBDAS=(0.1,1.0,10.0); THRESHOLDS=(-0.05,0.0,0.05)
FEATURES=('log1p_n','log1p_m','log1p_nnz','log_density','log1p_m_over_n')

def feat(r):
    return np.array([math.log1p(r['n']),math.log1p(r['m']),math.log1p(r['nnz']),math.log(max(r['density'],1e-12)),math.log1p(r['m_over_n'])],dtype=float)
def ridge_fit(X,y,lam):
    mu=X.mean(axis=0); sd=X.std(axis=0); sd=np.where(sd<1e-12,1.0,sd); Z=(X-mu)/sd; D=np.column_stack([np.ones(len(Z)),Z]); P=np.eye(D.shape[1]); P[0,0]=0.0
    w=np.linalg.solve(D.T@D+lam*P,D.T@y); return mu,sd,w
def predict(r,mu,sd,w):
    z=(feat(r)-mu)/sd; return float(np.r_[1.0,z]@w)
def eval_policy(rows,mu,sd,w,thr):
    choices=[]; costs=[]
    ds=[r['ds_cost_s'] for r in rows]; ip=[r['ipm_cost_s'] for r in rows]; sbs='highs-ds' if np.mean(ds)<=np.mean(ip) else 'highs-ipm'; sbs_costs=ds if sbs=='highs-ds' else ip
    for r in rows:
        pred=predict(r,mu,sd,w); choice='highs-ipm' if pred<thr else 'highs-ds'; route=r['ipm_cost_s'] if choice=='highs-ipm' else r['ds_cost_s']; costs.append(route+r['inspect_cost_s']); choices.append(choice)
    return {'mean_policy_s':float(np.mean(costs)),'sbs_backend':sbs,'sbs_mean_s':float(np.mean(sbs_costs)),'policy_to_sbs':float(np.mean(costs)/np.mean(sbs_costs)),'ds_choices':choices.count('highs-ds'),'ipm_choices':choices.count('highs-ipm')}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--trainval',type=Path,required=True); ap.add_argument('--split',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    d=json.loads(args.trainval.read_text()); s=json.loads(args.split.read_text())
    if d.get('test_costs_present') is not False: raise SystemExit('training artifact must explicitly contain no test costs')
    if d['split_sha256']!=s['split_sha256']: raise SystemExit('split hash mismatch')
    rows=d['rows']; tr=[r for r in rows if r['split']=='train']; va=[r for r in rows if r['split']=='validation'];
    if len(tr)!=20 or len(va)!=12 or any(r['split']=='test' for r in rows): raise SystemExit('train/validation isolation failure')
    if any(r['objective_mismatch'] for r in rows): raise SystemExit('objective mismatch in train/validation source evidence')
    X=np.vstack([feat(r) for r in tr]); y=np.array([math.log(max(r['ipm_cost_s'],1e-12)/max(r['ds_cost_s'],1e-12)) for r in tr])
    candidates=[]
    for lam in LAMBDAS:
        mu,sd,w=ridge_fit(X,y,lam)
        for thr in THRESHOLDS:
            ev=eval_policy(va,mu,sd,w,thr); candidates.append({'lambda':lam,'threshold':thr,'validation':ev,'mu':mu.tolist(),'sd':sd.tolist(),'weights':w.tolist()})
    # Pre-registered selection: min validation policy/SBS, then stronger regularization, then threshold closest to zero.
    best=min(candidates,key=lambda c:(c['validation']['policy_to_sbs'],-c['lambda'],abs(c['threshold']),c['threshold']))
    vg=best['validation']; validation_gate=(vg['policy_to_sbs']<=0.98 and vg['ds_choices']>=2 and vg['ipm_choices']>=2)
    model={'schema':'optimind.m26.ridge_selector.v1','model_family':'ridge_log_cost_ratio','feature_names':FEATURES,'lambda':best['lambda'],'threshold':best['threshold'],'mu':best['mu'],'sd':best['sd'],'weights':best['weights'],'split_sha256':s['split_sha256'],'trainval_sha256':sha(args.trainval),'train_instances':20,'validation_instances':12,'test_outcomes_seen_during_fit':False,'hyperparameter_grid':{'lambda':LAMBDAS,'threshold':THRESHOLDS},'validation':vg,'validation_gate_passed':validation_gate,'candidate_validation_summary':[{'lambda':c['lambda'],'threshold':c['threshold'],**c['validation']} for c in candidates]}
    raw=json.dumps(model,sort_keys=True,separators=(',',':')).encode(); model['model_sha256']=hashlib.sha256(raw).hexdigest(); args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(model,indent=2,sort_keys=True))
    print(json.dumps({'model_sha256':model['model_sha256'],'lambda':model['lambda'],'threshold':model['threshold'],'validation':vg,'validation_gate_passed':validation_gate},indent=2))
if __name__=='__main__': main()
