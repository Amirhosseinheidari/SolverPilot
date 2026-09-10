from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import RidgeCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED=20260906
PERMUTATIONS=2000
ALPHAS=(0.1,1.0,10.0,100.0)
FAMILY_ORDER=('A','B','C','D')
CUMULATIVE={'A':('A',),'AB':('A','B'),'ABC':('A','B','C'),'ABCD':('A','B','C','D')}


def sha256_file(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()


def robust_targets(m25r1:dict,m25r2:dict,m28:dict)->dict[str,dict[str,dict[str,float]]]:
 a1={r['instance']:r for r in m25r1['rows']}; a2={r['instance']:r for r in m25r2['rows']}
 m25={}
 for i in sorted(set(a1)&set(a2)):
  ds=float(np.median([a1[i]['ds_median_cost_s'],a2[i]['ds_median_cost_s']]))
  ipm=float(np.median([a1[i]['ipm_median_cost_s'],a2[i]['ipm_median_cost_s']]))
  m25[i]={'simplex':ds,'ipm':ipm}
 m28t={r['instance']:{'simplex':float(r['cost_medians_s']['simplex']),'ipm':float(r['cost_medians_s']['ipm'])} for r in m28['rows']}
 return {'m25':m25,'m28':m28t}


def target_values(costs:dict[str,float])->dict[str,float|str]:
 ds,ipm=costs['simplex'],costs['ipm']; den=max((ds+ipm)/2.0,1e-12); delta=ipm-ds
 return {'delta_s':delta,'delta_relative':delta/den,'absolute_opportunity_s':abs(delta),'vbs_s':min(ds,ipm),'winner':'simplex' if ds<ipm else 'ipm' if ipm<ds else 'tie'}


def flatten_features(row:dict)->dict[str,float]:
 out={}
 for fam in FAMILY_ORDER:
  for k,v in row['features'][fam].items(): out[f'{fam}:{k}']=float(v)
 return out


def family_feature_names(example:dict,famkey:str)->list[str]:
 allowed=set(CUMULATIVE[famkey]); return sorted(k for k in example if k.split(':',1)[0] in allowed)


def cumulative_cost(row:dict,famkey:str)->float:
 return float(sum(float(row['cost_s'][f]) for f in CUMULATIVE[famkey]))


def ranks_matrix(X:np.ndarray)->np.ndarray:
 return np.column_stack([rankdata(X[:,j],method='average') for j in range(X.shape[1])])


def standardized(A:np.ndarray)->np.ndarray:
 A=np.asarray(A,dtype=float); mu=A.mean(axis=0); sd=A.std(axis=0,ddof=1); sd=np.where(sd>0,sd,1.0); return (A-mu)/sd


def permutation_spearman(X:np.ndarray,y:np.ndarray,nperm:int,seed:int)->tuple[np.ndarray,np.ndarray]:
 Xr=ranks_matrix(X); yr=rankdata(y,method='average'); Xz=standardized(Xr); yz=standardized(yr.reshape(-1,1)).ravel(); n=len(y)
 obs=(Xz.T@yz)/(n-1)
 rng=np.random.default_rng(seed); counts=np.zeros(X.shape[1],dtype=int)
 for _ in range(nperm):
  yp=yz[rng.permutation(n)]; cor=(Xz.T@yp)/(n-1); counts += (np.abs(cor) >= np.abs(obs)-1e-15)
 p=(counts+1)/(nperm+1)
 return obs,p


def bh_qvalues(p:np.ndarray)->np.ndarray:
 p=np.asarray(p,float); m=len(p); order=np.argsort(p); q=np.empty(m,float); prev=1.0
 for rank,idx in reversed(list(enumerate(order,start=1))):
  val=min(prev,p[idx]*m/rank); q[idx]=val; prev=val
 return q


def transfer_probe(source:str,target:str,famkey:str,features:dict,targets:dict,feature_rows:dict)->dict:
 snames=sorted(set(features[source])&set(targets[source])); tnames=sorted(set(features[target])&set(targets[target]))
 fns=family_feature_names(next(iter(features[source].values())),famkey)
 Xs=np.array([[features[source][i][f] for f in fns] for i in snames],float)
 ys=np.array([targets[source][i]['delta_relative'] for i in snames],float)
 Xt=np.array([[features[target][i][f] for f in fns] for i in tnames],float)
 yt=np.array([targets[target][i]['delta_relative'] for i in tnames],float)
 cv=KFold(n_splits=5,shuffle=True,random_state=SEED)
 model=make_pipeline(StandardScaler(),RidgeCV(alphas=np.array(ALPHAS),cv=cv,scoring='neg_mean_absolute_error'))
 model.fit(Xs,ys); pred=model.predict(Xt)
 # Spearman via ranks; avoid scipy warnings for constants.
 pr=rankdata(pred); yr=rankdata(yt); rho=float(np.corrcoef(pr,yr)[0,1]) if np.std(pr)>0 and np.std(yr)>0 else 0.0
 decisive=np.abs(yt)>=0.05
 if decisive.any() and len(np.unique((yt[decisive]>0).astype(int)))>1:
  bacc=float(balanced_accuracy_score((yt[decisive]>0).astype(int),(pred[decisive]>0).astype(int)))
 else:bacc=0.0
 # Cost-adjusted policy against target cohort SBS.
 simple=np.array([targets[target][i]['simplex'] for i in tnames],float); ipm=np.array([targets[target][i]['ipm'] for i in tnames],float)
 sbs='simplex' if simple.mean()<=ipm.mean() else 'ipm'; sbs_cost=simple if sbs=='simplex' else ipm
 chosen=np.where(pred>0,simple,ipm)
 fcost=np.array([cumulative_cost(feature_rows[target][i],famkey) for i in tnames],float)
 total=chosen+fcost; vbs=np.minimum(simple,ipm)
 policy_ratio=float(total.mean()/sbs_cost.mean())
 raw_policy_ratio=float(chosen.mean()/sbs_cost.mean())
 choices=np.where(pred>0,'simplex','ipm')
 harmful=int(np.sum((choices!=sbs)&(chosen>sbs_cost+1e-15)))
 p90=float(np.quantile(np.maximum(total-vbs,0.0),0.90))
 alpha=float(model.named_steps['ridgecv'].alpha_)
 return {'source':source,'target':target,'family':famkey,'features':len(fns),'alpha':alpha,'target_spearman':rho,'decisive_count':int(decisive.sum()),'decisive_balanced_accuracy':bacc,'target_sbs':sbs,'policy_to_sbs':policy_ratio,'policy_to_sbs_without_feature_cost':raw_policy_ratio,'harmful_switches':harmful,'p90_absolute_regret_s':p90,'switches_to_simplex':int(np.sum(pred>0)),'switches_to_ipm':int(np.sum(pred<=0))}


def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--features',type=Path,required=True); ap.add_argument('--m25-run1',type=Path,required=True); ap.add_argument('--m25-run2',type=Path,required=True); ap.add_argument('--m28',type=Path,required=True); ap.add_argument('--protocol',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
 feat=json.loads(args.features.read_text()); r1=json.loads(args.m25_run1.read_text()); r2=json.loads(args.m25_run2.read_text()); r28=json.loads(args.m28.read_text())
 # Fail-closed source/protocol integrity. M29 is discovery-only, but even discovery
 # conclusions should not silently survive stale or mismatched evidence.
 protocol_sha=sha256_file(args.protocol)
 feature_protocol_match=str(feat.get('protocol_sha256'))==protocol_sha
 m25_names1={r['instance'] for r in r1.get('rows',[])}; m25_names2={r['instance'] for r in r2.get('rows',[])}
 m28_names={r['instance'] for r in r28.get('rows',[])}
 m25_integrity=bool(
  r1.get('official_corpus_integrity_passed') and r2.get('official_corpus_integrity_passed')
  and r1.get('selection_opportunity_validated') and r2.get('selection_opportunity_validated')
  and len(m25_names1)==48 and m25_names1==m25_names2
  and int((r1.get('metrics') or {}).get('objective_mismatches',1) or 0)==0
  and int((r2.get('metrics') or {}).get('objective_mismatches',1) or 0)==0
 )
 m28_metrics=r28.get('metrics') if isinstance(r28.get('metrics'),dict) else {}
 m28_gates=r28.get('pre_registered_gates') if isinstance(r28.get('pre_registered_gates'),dict) else {}
 m28_integrity=bool(
  len(m28_names)==48
  and m28_gates.get('official_corpus_integrity')
  and m28_gates.get('complete_three_round_accounting')
  and m28_gates.get('single_fixed_host_and_runner_provenance')
  and m28_gates.get('zero_material_objective_mismatch')
  and float(m28_metrics.get('forced_sbs_to_vbs_relative_gap',0.0) or 0.0)>=0.03
 )
 feature_counts_ok=bool(
  feat.get('counts',{}).get('m25',{}).get('ok')==48
  and feat.get('counts',{}).get('m28',{}).get('ok')==48
  and feat.get('counts',{}).get('m25',{}).get('probe_ok')==48
  and feat.get('counts',{}).get('m28',{}).get('probe_ok')==48
 )
 source_integrity={
  'feature_protocol_hash_match':feature_protocol_match,
  'feature_extraction_complete_48_48':feature_counts_ok,
  'm25_source_integrity':m25_integrity,
  'm28_source_integrity':m28_integrity,
  'm25_m28_overlap_zero':len(m25_names1 & m28_names)==0,
 }
 representation_integrity=all(source_integrity.values())
 selection_opportunity_context_validated=bool(m25_integrity and m28_integrity)
 costs=robust_targets(r1,r2,r28); targets={c:{i:{**v,**target_values(v)} for i,v in rows.items()} for c,rows in costs.items()}
 feature_rows={c:{} for c in costs}; features={c:{} for c in costs}
 for row in feat['rows']:
  if not row.get('ok'): continue
  c=row['cohort']; i=row['instance']; feature_rows[c][i]=row; features[c][i]=flatten_features(row)
 # Coverage and feature cost.
 cost_audit={}
 for c in ('m25','m28'):
  names=sorted(set(features[c])&set(targets[c])); sbs_simple=np.array([targets[c][i]['simplex'] for i in names]); sbs_ipm=np.array([targets[c][i]['ipm'] for i in names]); sbs=np.minimum(sbs_simple.mean(),sbs_ipm.mean())
  median_sbs_inst=float(np.median(sbs_simple if sbs_simple.mean()<=sbs_ipm.mean() else sbs_ipm))
  median_opp=float(np.median([targets[c][i]['absolute_opportunity_s'] for i in names]))
  cost_audit[c]={}
  for fam in CUMULATIVE:
   xs=np.array([cumulative_cost(feature_rows[c][i],fam) for i in names])
   cost_audit[c][fam]={'coverage':len(names)/48,'median_s':float(np.median(xs)),'p95_s':float(np.quantile(xs,.95)),'median_fraction_of_median_sbs':float(np.median(xs)/max(median_sbs_inst,1e-12)),'median_fraction_of_median_absolute_opportunity':float(np.median(xs)/max(median_opp,1e-12)),'cohort_sbs_mean_s':float(sbs),'cohort_median_sbs_instance_s':median_sbs_inst,'cohort_median_absolute_opportunity_s':median_opp}
 # Stable univariate info using all features present in both cohorts.
 allf=sorted(set.intersection(*(set(next(iter(features[c].values())).keys()) for c in ('m25','m28'))))
 uni={}
 for idx,c in enumerate(('m25','m28')):
  names=sorted(set(features[c])&set(targets[c])); X=np.array([[features[c][i][f] for f in allf] for i in names],float); y=np.array([targets[c][i]['delta_s'] for i in names],float)
  rho,pv=permutation_spearman(X,y,PERMUTATIONS,SEED+idx); qv=bh_qvalues(pv)
  uni[c]={f:{'rho':float(rho[j]),'p_perm':float(pv[j]),'q_bh':float(qv[j])} for j,f in enumerate(allf)}
 stable=[]
 for f in allf:
  a,b=uni['m25'][f],uni['m28'][f]; same=(a['rho']*b['rho']>0); cond=same and abs(a['rho'])>=.25 and abs(b['rho'])>=.25 and ((a['q_bh']<=.10 and b['p_perm']<=.10) or (b['q_bh']<=.10 and a['p_perm']<=.10))
  if cond: stable.append({'feature':f,'m25':a,'m28':b,'min_abs_rho':min(abs(a['rho']),abs(b['rho']))})
 stable.sort(key=lambda z:z['min_abs_rho'],reverse=True)
 # Cross cohort probes.
 probes=[]
 for fam in CUMULATIVE:
  probes.append(transfer_probe('m25','m28',fam,features,targets,feature_rows)); probes.append(transfer_probe('m28','m25',fam,features,targets,feature_rows))
 # Gate per family.
 family_gate={}
 for fam in CUMULATIVE:
  allowed=set(CUMULATIVE[fam]); stable_n=sum(1 for s in stable if s['feature'].split(':',1)[0] in allowed)
  p12=next(p for p in probes if p['family']==fam and p['source']=='m25'); p21=next(p for p in probes if p['family']==fam and p['source']=='m28')
  gates={
   'coverage_ge_0_95_both':cost_audit['m25'][fam]['coverage']>=.95 and cost_audit['m28'][fam]['coverage']>=.95,
   'median_feature_cost_le_0_10_median_sbs_both':cost_audit['m25'][fam]['median_fraction_of_median_sbs']<=.10 and cost_audit['m28'][fam]['median_fraction_of_median_sbs']<=.10,
   'stable_features_ge_2':stable_n>=2,
   'cross_cohort_spearman_ge_0_30_both':p12['target_spearman']>=.30 and p21['target_spearman']>=.30,
   'decisive_bacc_ge_0_60_both':p12['decisive_balanced_accuracy']>=.60 and p21['decisive_balanced_accuracy']>=.60,
   'cost_adjusted_policy_gate':(p12['policy_to_sbs']<=.99 and p21['policy_to_sbs']<=1.01) or (p21['policy_to_sbs']<=.99 and p12['policy_to_sbs']<=1.01),
   'no_candidate_solver_trajectory_features':True,
  }
  family_gate[fam]={'stable_feature_count':stable_n,'gates':gates,'passes_all':all(gates.values())}
 authorized=[f for f,v in family_gate.items() if v['passes_all']]
 final_authorized=authorized if representation_integrity and selection_opportunity_context_validated else []
 payload={
  'schema':'optimind.m29.value_of_information_audit.v1','purpose':'post-outcome feature representation audit only; never deployment promotion evidence','claim_boundary':['M25/M28 outcomes pre-existed before M29; this is discovery evidence','M29 may authorize only a fresh-corpus M30 experiment','M29 cannot enable production performance routing'],
  'protocol_sha256':protocol_sha,'feature_extraction_sha256':sha256_file(args.features),'source_hashes':{'m25_run1':sha256_file(args.m25_run1),'m25_run2':sha256_file(args.m25_run2),'m28':sha256_file(args.m28)},
  'source_integrity':source_integrity,'representation_audit_integrity_passed':representation_integrity,'selection_opportunity_context_validated':selection_opportunity_context_validated,
  'normalization_contract':{'constraint_rows':'divide row matrix coefficients and finite row sides by row max-abs coefficient','objective':'divide objective coefficients by objective max-abs coefficient','signed_bounds':'siglog summaries; no arbitrary variable-column rescaling'},
  'cohorts':{'m25':len(targets['m25']),'m28':len(targets['m28']),'overlap':len(set(targets['m25'])&set(targets['m28']))},
  'feature_counts':{'A':sum(k.startswith('A:') for k in allf),'B':sum(k.startswith('B:') for k in allf),'C':sum(k.startswith('C:') for k in allf),'D':sum(k.startswith('D:') for k in allf),'total':len(allf)},
  'feature_cost':cost_audit,'stable_univariate_features':stable,'stable_univariate_count':len(stable),'univariate_all':uni,'cross_cohort_probes':probes,'family_authorization':family_gate,'m30_authorized_families':final_authorized,'m30_authorized':bool(final_authorized),'production_performance_routing_authorized':False,
 }
 args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding='utf-8')
 print(json.dumps({'stable_features':len(stable),'authorized_families':authorized,'probes':probes,'family_gate':family_gate},indent=2))

if __name__=='__main__':main()
