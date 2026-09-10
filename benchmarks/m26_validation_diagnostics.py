from __future__ import annotations
import argparse,json,math,statistics
from pathlib import Path
import numpy as np

def feat(r): return np.array([math.log1p(r['n']),math.log1p(r['m']),math.log1p(r['nnz']),math.log(max(r['density'],1e-12)),math.log1p(r['m_over_n'])],dtype=float)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--model',type=Path,required=True); ap.add_argument('--trainval',type=Path,required=True); ap.add_argument('--split',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
 m=json.loads(args.model.read_text()); tv=json.loads(args.trainval.read_text()); sp=json.loads(args.split.read_text())
 test_names={r['instance'] for r in sp['records'] if r['split']=='test'}; raw=args.trainval.read_text()
 leakage={'test_names_in_trainval':sorted(n for n in test_names if n in raw),'test_rows_in_trainval':sum(r.get('split')=='test' for r in tv['rows']),'model_test_outcomes_seen':bool(m.get('test_outcomes_seen_during_fit'))}
 mu=np.array(m['mu']); sd=np.array(m['sd']); w=np.array(m['weights']); vals=[r for r in tv['rows'] if r['split']=='validation']
 sbs=m['validation']['sbs_backend']; details=[]
 for r in vals:
  pred=float(np.r_[1.0,(feat(r)-mu)/sd]@w); choice='highs-ipm' if pred<m['threshold'] else 'highs-ds'; selected=r['ipm_cost_s'] if choice=='highs-ipm' else r['ds_cost_s']; sbs_cost=r['ipm_cost_s'] if sbs=='highs-ipm' else r['ds_cost_s']; policy=selected+r['inspect_cost_s'];
  details.append({'instance':r['instance'],'predicted_log_ipm_over_ds':pred,'actual_log_ipm_over_ds':math.log(max(r['ipm_cost_s'],1e-12)/max(r['ds_cost_s'],1e-12)),'choice':choice,'actual_faster':'highs-ipm' if r['ipm_cost_s']<r['ds_cost_s'] else 'highs-ds','policy_cost_s':policy,'sbs_route_cost_s':sbs_cost,'excess_vs_sbs_s':policy-sbs_cost})
 details.sort(key=lambda x:x['excess_vs_sbs_s'],reverse=True)
 positive=sum(max(0,d['excess_vs_sbs_s']) for d in details); negative=-sum(min(0,d['excess_vs_sbs_s']) for d in details)
 payload={'schema':'optimind.m26.validation_diagnostics.v1','leakage_audit':leakage,'validation_gate_passed':bool(m['validation_gate_passed']),'validation_policy_to_sbs':m['validation']['policy_to_sbs'],'all_grid_candidates_worse_than_sbs':all(c['policy_to_sbs']>1 for c in m['candidate_validation_summary']),'best_grid_policy_to_sbs':min(c['policy_to_sbs'] for c in m['candidate_validation_summary']),'worst_grid_policy_to_sbs':max(c['policy_to_sbs'] for c in m['candidate_validation_summary']),'total_positive_regret_s':positive,'total_savings_vs_sbs_s':negative,'largest_regret_instance':details[0],'largest_regret_share':details[0]['excess_vs_sbs_s']/positive if positive else 0.0,'validation_details':details,'interpretation':['The low-capacity relative-log-cost model failed before test evaluation.','Validation mean cost is dominated by a small number of expensive mistakes; winner accuracy is not the deployment objective.','The sealed 16-instance test set remains untouched and is reserved for a future pre-registered model family.']}
 args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)); print(json.dumps({k:payload[k] for k in ['validation_gate_passed','validation_policy_to_sbs','all_grid_candidates_worse_than_sbs','best_grid_policy_to_sbs','worst_grid_policy_to_sbs','largest_regret_share']},indent=2))
if __name__=='__main__': main()
