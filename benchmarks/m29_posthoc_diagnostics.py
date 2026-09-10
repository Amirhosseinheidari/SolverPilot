from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, ks_2samp
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

SEED=20260906
ALPHAS=np.array([0.1,1.0,10.0,100.0])
CUM={'A':('A',),'AB':('A','B'),'ABC':('A','B','C'),'ABCD':('A','B','C','D')}

def flat(row):
 return {f'{fam}:{k}':float(v) for fam in 'ABCD' for k,v in row['features'][fam].items()}
def cost(row,fam):return sum(float(row['cost_s'][x]) for x in CUM[fam])
def targ(costs):
 ds,ipm=costs; d=ipm-ds; den=max((ipm+ds)/2,1e-12); return {'simplex':ds,'ipm':ipm,'delta_rel':d/den,'delta_s':d}
def spearman(a,b):
 ra,rb=rankdata(a),rankdata(b)
 return float(np.corrcoef(ra,rb)[0,1]) if np.std(ra)>0 and np.std(rb)>0 else 0.0

def build_targets(r1,r2,r28):
 a1={r['instance']:r for r in r1['rows']};a2={r['instance']:r for r in r2['rows']}
 m25={i:targ((float(np.median([a1[i]['ds_median_cost_s'],a2[i]['ds_median_cost_s']])),float(np.median([a1[i]['ipm_median_cost_s'],a2[i]['ipm_median_cost_s']])))) for i in sorted(a1)}
 m28={r['instance']:targ((float(r['cost_medians_s']['simplex']),float(r['cost_medians_s']['ipm']))) for r in r28['rows']}
 return {'m25':m25,'m28':m28}

def within_oof(c,fam,features,rows,targets):
 names=sorted(targets[c]); fns=sorted(k for k in features[c][names[0]] if k.split(':',1)[0] in CUM[fam]); X=np.array([[features[c][i][f] for f in fns] for i in names]); y=np.array([targets[c][i]['delta_rel'] for i in names]); pred=np.zeros(len(names));
 outer=KFold(n_splits=6,shuffle=True,random_state=SEED)
 for tr,te in outer.split(X):
  inner=KFold(n_splits=4,shuffle=True,random_state=SEED+1)
  model=make_pipeline(StandardScaler(),RidgeCV(alphas=ALPHAS,cv=inner,scoring='neg_mean_absolute_error')); model.fit(X[tr],y[tr]); pred[te]=model.predict(X[te])
 decisive=np.abs(y)>=.05; bacc=float(balanced_accuracy_score((y[decisive]>0).astype(int),(pred[decisive]>0).astype(int))) if decisive.any() and len(np.unique((y[decisive]>0).astype(int)))>1 else 0.0
 ds=np.array([targets[c][i]['simplex'] for i in names]);ipm=np.array([targets[c][i]['ipm'] for i in names]);sbs='simplex' if ds.mean()<=ipm.mean() else 'ipm'; sbsc=ds if sbs=='simplex' else ipm; chosen=np.where(pred>0,ds,ipm);fc=np.array([cost(rows[c][i],fam) for i in names]);
 return {'cohort':c,'family':fam,'features':len(fns),'oof_spearman':spearman(pred,y),'decisive_bacc':bacc,'sbs':sbs,'policy_to_sbs_no_feature_cost':float(chosen.mean()/sbsc.mean()),'policy_to_sbs_with_feature_cost':float((chosen+fc).mean()/sbsc.mean())}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--features',type=Path,required=True);ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--m25-run1',type=Path,required=True);ap.add_argument('--m25-run2',type=Path,required=True);ap.add_argument('--m28',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 fd=json.loads(a.features.read_text()); audit=json.loads(a.audit.read_text());r1=json.loads(a.m25_run1.read_text());r2=json.loads(a.m25_run2.read_text());r28=json.loads(a.m28.read_text());targets=build_targets(r1,r2,r28)
 rows={'m25':{},'m28':{}};features={'m25':{},'m28':{}}
 for r in fd['rows']:
  rows[r['cohort']][r['instance']]=r;features[r['cohort']][r['instance']]=flat(r)
 # Covariate shift on base A features.
 base=sorted(k for k in features['m25'][next(iter(features['m25']))] if k.startswith('A:'))
 shift=[]
 for f in base:
  x=np.array([features['m25'][i][f] for i in sorted(features['m25'])]);y=np.array([features['m28'][i][f] for i in sorted(features['m28'])]);pooled=np.sqrt((x.var(ddof=1)+y.var(ddof=1))/2);smd=float((y.mean()-x.mean())/pooled) if pooled>0 else 0.0;ks=ks_2samp(x,y,method='auto');shift.append({'feature':f,'standardized_mean_difference_m28_minus_m25':smd,'ks_statistic':float(ks.statistic),'ks_p':float(ks.pvalue)})
 # Within-cohort discovery CV, to distinguish no signal from non-portable signal.
 oof=[within_oof(c,f,features,rows,targets) for c in ('m25','m28') for f in CUM]
 # Residualize the single stable feature against base size features.
 residual=[]
 for sf in audit['stable_univariate_features']:
  f=sf['feature']
  for c in ('m25','m28'):
   names=sorted(targets[c]);X=np.array([[features[c][i][bf] for bf in base] for i in names]);z=np.array([features[c][i][f] for i in names]);y=np.array([targets[c][i]['delta_s'] for i in names]);pipe=make_pipeline(StandardScaler(),Ridge(alpha=10.0));zh=pipe.fit(X,z).predict(X);yh=make_pipeline(StandardScaler(),Ridge(alpha=10.0)).fit(X,y).predict(X);residual.append({'feature':f,'cohort':c,'raw_rho':spearman(z,y),'residual_rho_after_base_A':spearman(z-zh,y-yh)})
 payload={'schema':'optimind.m29.posthoc_diagnostics.v1','claim_boundary':'diagnostic only; not part of M30 authorization gate','covariate_shift_base_features':shift,'within_cohort_oof':oof,'stable_feature_residualization':residual,'zero_feature_cost_transfer':[{k:v for k,v in p.items() if k in ('source','target','family','target_spearman','decisive_balanced_accuracy','policy_to_sbs_without_feature_cost')} for p in audit['cross_cohort_probes']]}
 a.output.write_text(json.dumps(payload,indent=2,sort_keys=True));print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
