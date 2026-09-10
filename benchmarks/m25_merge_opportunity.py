from __future__ import annotations
import argparse, json, statistics
from pathlib import Path
import numpy as np

MIN_RELATIVE_VBS_GAIN=0.03
MIN_RELATIVE_ORACLE_FEATURE_GAIN=0.02
MIN_NON_SBS_WINS=3
MIN_DECISIVE_NON_SBS_WINS=2
DECISIVE_RATIO=0.90
ROUNDS=2
TARGET_INSTANCES=48


def bootstrap(a,b,draws=20000):
    rng=np.random.default_rng(250025); n=len(a); vals=np.empty(draws)
    for i in range(draws):
        idx=rng.integers(0,n,size=n); vals[i]=a[idx].mean()/b[idx].mean()
    q=np.quantile(vals,[.025,.5,.975])
    return {'point':float(a.mean()/b.mean()),'low':float(q[0]),'median':float(q[1]),'high':float(q[2]),'draws':draws}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--inputs',type=Path,nargs='+',required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    parts=[json.loads(p.read_text()) for p in args.inputs]
    sigs={p['cohort_signature'] for p in parts}; counts={p['shard_count'] for p in parts}; idxs=[p['shard_index'] for p in parts]
    if len(sigs)!=1 or len(counts)!=1: raise SystemExit('cohort/shard mismatch')
    shard_count=next(iter(counts))
    if sorted(idxs)!=list(range(shard_count)): raise SystemExit(f'incomplete shards: {idxs}')
    rows=[r for p in parts for r in p['rows']]
    names=[r['instance'] for r in rows]
    if len(rows)!=TARGET_INSTANCES or len(set(names))!=TARGET_INSTANCES: raise SystemExit(f'expected 48 unique rows, got {len(rows)}/{len(set(names))}')
    full=parts[0]['full_cohort_names']
    if set(names)!=set(full): raise SystemExit('merged rows do not equal frozen full cohort')
    ds=np.asarray([r['ds_median_cost_s'] for r in rows],float); ip=np.asarray([r['ipm_median_cost_s'] for r in rows],float); inspect=np.asarray([r['inspect_median_s'] for r in rows],float)
    means={'scipy-highs-ds':float(ds.mean()),'scipy-highs-ipm':float(ip.mean())}; sbs_name=min(means,key=means.get); sbs=ds if sbs_name.endswith('ds') else ip
    vbs=np.minimum(ds,ip); oracle=vbs+inspect
    vbs_gain=float((sbs.mean()-vbs.mean())/sbs.mean()); oracle_gain=float((sbs.mean()-oracle.mean())/sbs.mean()); boot=bootstrap(oracle,sbs)
    objective_mismatches=0
    for r in rows:
        if r['objective_abs_diff'] is None: continue
        ref=next((x['objective'] for x in r['samples']['highs-ds'] if x['objective'] is not None),0.0); tol=1e-6*max(1.0,abs(float(ref)))
        objective_mismatches += float(r['objective_abs_diff'])>tol
    total=len(rows)*ROUNDS; ds_terminal=sum(r['ds_terminal_rounds'] for r in rows); ip_terminal=sum(r['ipm_terminal_rounds'] for r in rows)
    non_sbs=sum(r['winner']=='highs-ipm' if sbs_name.endswith('ds') else r['winner']=='highs-ds' for r in rows)
    decisive=sum(float(r['ipm_over_ds'])<=DECISIVE_RATIO for r in rows) if sbs_name.endswith('ds') else sum(float(r['ipm_over_ds'])>=1/DECISIVE_RATIO for r in rows)
    gates={
      'official_corpus_integrity':all(p['official_corpus_integrity_passed'] for p in parts),
      'minimum_instances':len(rows)>=TARGET_INSTANCES,
      'ds_terminal_rate_ge_0_90':ds_terminal/max(total,1)>=.90,
      'ipm_terminal_rate_ge_0_90':ip_terminal/max(total,1)>=.90,
      'objective_mismatches_zero':objective_mismatches==0,
      'vbs_relative_gain_ge_0_03':vbs_gain>=MIN_RELATIVE_VBS_GAIN,
      'oracle_with_feature_relative_gain_ge_0_02':oracle_gain>=MIN_RELATIVE_ORACLE_FEATURE_GAIN,
      'oracle_bootstrap_upper_below_1':boot['high']<1.0,
      'non_sbs_wins_ge_3':non_sbs>=MIN_NON_SBS_WINS,
      'decisive_non_sbs_wins_ge_2':decisive>=MIN_DECISIVE_NON_SBS_WINS,
      'complete_deterministic_shards':True,
    }
    metrics={'sbs_backend':sbs_name,'sbs_mean_s':float(sbs.mean()),'vbs_mean_s':float(vbs.mean()),'oracle_with_inspection_mean_s':float(oracle.mean()),'vbs_relative_gain':vbs_gain,'oracle_with_inspection_relative_gain':oracle_gain,'oracle_with_inspection_to_sbs_bootstrap':boot,'ds_wins':sum(r['winner']=='highs-ds' for r in rows),'ipm_wins':sum(r['winner']=='highs-ipm' for r in rows),'ties':sum(r['winner']=='tie' for r in rows),'non_sbs_wins':non_sbs,'decisive_non_sbs_wins':decisive,'objective_mismatches':objective_mismatches,'ds_terminal_rate':ds_terminal/max(total,1),'ipm_terminal_rate':ip_terminal/max(total,1),'median_inspection_s':float(statistics.median(inspect))}
    payload={'schema':'optimind.m25.public_ood_lp_opportunity.v1','source':parts[0]['source'],'purpose':'pre-selector opportunity audit; no selector is trained or promoted in M25','miplib_zip_sha256':parts[0]['miplib_zip_sha256'],'manifest_sha256':parts[0]['manifest_sha256'],'official_corpus_integrity_passed':gates['official_corpus_integrity'],'cohort_signature':next(iter(sigs)),'instances':len(rows),'rounds':ROUNDS,'pre_registered_opportunity_gate':parts[0]['pre_registered_opportunity_gate'],'selection_protocol':parts[0]['selection_protocol'],'metrics':metrics,'opportunity_gates':gates,'selection_opportunity_validated':all(gates.values()),'shards':[{'index':p['shard_index'],'wall_s':p['wall_s'],'rows':len(p['rows'])} for p in parts],'environment':parts[0]['environment'],'claims_boundary':['M25 measures whether a nontrivial DS-vs-IPM selection opportunity exists after current inspection cost','M25 does not train, tune, or authorize a production selector','a failed opportunity gate means automatic LP performance routing remains disabled'],'rows':sorted(rows,key=lambda r:r['instance'])}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)); print(json.dumps({'metrics':metrics,'gates':gates,'validated':payload['selection_opportunity_validated']},indent=2))
if __name__=='__main__': main()
