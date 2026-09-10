from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import sklearn

FEATURES=('log1p_n','log1p_m','log1p_nnz','log_density','log1p_m_over_n')
MAX_DEPTH=2
MIN_SAMPLES_LEAF=8
RANDOM_STATE=270027
SWITCH_MARGIN_S=0.005
BOOTSTRAP_LOWER_Q=0.10
BOOTSTRAP_DRAWS=5000
DEV_POLICY_TO_SBS_MAX=0.99
DEV_MIN_SWITCHES=3
DEV_MAX_SWITCHES=12
DEV_P90_TO_DEFAULT_MAX=1.10


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def canonical_hash(payload: dict) -> str:
    raw=json.dumps(payload,sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(raw).hexdigest()

def feat(r):
    return np.array([
        math.log1p(r['n']), math.log1p(r['m']), math.log1p(r['nnz']),
        math.log(max(r['density'],1e-12)), math.log1p(r['m_over_n'])
    ],dtype=float)

def bootstrap_lower(vals, *, seed: int) -> float:
    vals=np.asarray(vals,dtype=float)
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(vals),size=(BOOTSTRAP_DRAWS,len(vals)))
    means=vals[idx].mean(axis=1)
    return float(np.quantile(means,BOOTSTRAP_LOWER_Q))

def serialize_tree(model: DecisionTreeRegressor) -> list[dict]:
    t=model.tree_; out=[]
    for i in range(t.node_count):
        out.append({
            'node':i,
            'left':int(t.children_left[i]),
            'right':int(t.children_right[i]),
            'feature':int(t.feature[i]),
            'threshold':float(t.threshold[i]),
            'value':float(t.value[i].reshape(-1)[0]),
            'samples':int(t.n_node_samples[i]),
            'is_leaf':bool(t.children_left[i] == t.children_right[i]),
        })
    return out

def leaf_stats(model, X, y):
    ids=model.apply(X); stats={}
    used_features=sorted({int(f) for f in model.tree_.feature if f>=0})
    for lid in sorted(set(ids.tolist())):
        mask=ids==lid; vals=y[mask]; leafX=X[mask]
        seed=int(hashlib.sha256(f'{RANDOM_STATE}:{lid}'.encode()).hexdigest()[:8],16)
        support={str(f):{'min':float(leafX[:,f].min()),'max':float(leafX[:,f].max())} for f in used_features}
        stats[str(int(lid))]={
            'samples':int(mask.sum()),
            'mean_saving_s':float(vals.mean()),
            'min_saving_s':float(vals.min()),
            'max_saving_s':float(vals.max()),
            'bootstrap_one_sided_90_lower_mean_s':bootstrap_lower(vals,seed=seed),
            'support_used_features':support,
        }
    return stats

def leaf_decision(model, Xtrain, ytrain, x):
    train_ids=model.apply(Xtrain); lid=int(model.apply(x.reshape(1,-1))[0]); vals=ytrain[train_ids==lid]
    seed=int(hashlib.sha256(f'oof:{lid}:{len(vals)}:{float(vals.sum()):.12g}'.encode()).hexdigest()[:8],16)
    lower=bootstrap_lower(vals,seed=seed); mean=float(vals.mean())
    used=sorted({int(f) for f in model.tree_.feature if f>=0})
    leafX=Xtrain[train_ids==lid]
    support=all(float(leafX[:,f].min())-1e-12 <= float(x[f]) <= float(leafX[:,f].max())+1e-12 for f in used)
    switch=bool(len(vals)>=MIN_SAMPLES_LEAF and mean>SWITCH_MARGIN_S and lower>SWITCH_MARGIN_S and support)
    return switch, {'leaf':lid,'samples':len(vals),'mean_saving_s':mean,'lower_saving_s':lower,'support_passed':support}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--trainval',type=Path,required=True)
    ap.add_argument('--split',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--model-output',type=Path,required=True)
    ap.add_argument('--dev-audit-output',type=Path,required=True)
    args=ap.parse_args()

    tv=json.loads(args.trainval.read_text()); split=json.loads(args.split.read_text())
    if tv.get('test_costs_present') is not False: raise SystemExit('test costs must be absent')
    if tv.get('split_sha256') != split.get('split_sha256'): raise SystemExit('split SHA mismatch')
    rows=list(tv['rows'])
    if len(rows)!=32 or any(r['split']=='test' for r in rows): raise SystemExit('development isolation failure')
    if any(bool(r.get('objective_mismatch')) for r in rows): raise SystemExit('objective mismatch in development rows')
    group={r['instance']:r['group'] for r in split['records']}
    test_names={r['instance'] for r in split['records'] if r['split']=='test'}
    raw=args.trainval.read_text()
    if any(n in raw for n in test_names): raise SystemExit('test instance leaked into training artifact')

    X=np.vstack([feat(r) for r in rows])
    ds=np.asarray([r['ds_cost_s'] for r in rows],float); ip=np.asarray([r['ipm_cost_s'] for r in rows],float)
    inspect=np.asarray([r['inspect_cost_s'] for r in rows],float)
    means={'scipy-highs-ds':float(ds.mean()),'scipy-highs-ipm':float(ip.mean())}
    default=min(means,key=means.get); alternate='scipy-highs-ipm' if default.endswith('ds') else 'scipy-highs-ds'
    default_cost=ds if default.endswith('ds') else ip; alternate_cost=ip if default.endswith('ds') else ds
    y=default_cost-alternate_cost

    groups=np.asarray([group[r['instance']] for r in rows],dtype=object)
    oof_switch=np.zeros(len(rows),bool); oof_meta=[None]*len(rows)
    for g in sorted(set(groups.tolist())):
        tr=groups!=g; te=groups==g
        model=DecisionTreeRegressor(max_depth=MAX_DEPTH,min_samples_leaf=MIN_SAMPLES_LEAF,criterion='squared_error',random_state=RANDOM_STATE)
        model.fit(X[tr],y[tr])
        for idx in np.where(te)[0]:
            sw,meta=leaf_decision(model,X[tr],y[tr],X[idx]); oof_switch[idx]=sw; oof_meta[idx]=meta

    policy=np.where(oof_switch,alternate_cost,default_cost)+inspect
    sbs_mean=float(default_cost.mean())
    ratios=policy/np.maximum(default_cost,1e-12)
    harmful_switches=int(sum(bool(oof_switch[i]) and alternate_cost[i] > default_cost[i] for i in range(len(rows))))
    dev_metrics={
        'development_sbs_backend':default,
        'alternate_backend':alternate,
        'development_sbs_mean_s':sbs_mean,
        'oof_policy_mean_s':float(policy.mean()),
        'oof_policy_to_sbs':float(policy.mean()/sbs_mean),
        'switches':int(oof_switch.sum()),
        'switch_rate':float(oof_switch.mean()),
        'harmful_alternate_switches':harmful_switches,
        'p90_policy_to_default_route':float(np.quantile(ratios,.90)),
        'max_policy_minus_default_s':float(np.max(policy-default_cost)),
        'mean_inspection_s':float(inspect.mean()),
    }
    gates={
        'all_32_oof_rows_accounted':len(oof_meta)==32 and all(x is not None for x in oof_meta),
        'group_leakage_zero':True,
        'objective_mismatches_zero':True,
        'oof_policy_to_sbs_le_0_99':dev_metrics['oof_policy_to_sbs']<=DEV_POLICY_TO_SBS_MAX,
        'switch_count_in_range':DEV_MIN_SWITCHES<=dev_metrics['switches']<=DEV_MAX_SWITCHES,
        'harmful_switches_zero':harmful_switches==0,
        'p90_policy_to_default_le_1_10':dev_metrics['p90_policy_to_default_route']<=DEV_P90_TO_DEFAULT_MAX,
        'test_costs_absent':True,
    }
    dev_pass=all(gates.values())

    final=DecisionTreeRegressor(max_depth=MAX_DEPTH,min_samples_leaf=MIN_SAMPLES_LEAF,criterion='squared_error',random_state=RANDOM_STATE)
    final.fit(X,y)
    stats=leaf_stats(final,X,y)
    model_payload={
        'schema':'optimind.m27.selective_absolute_saving_tree.v1',
        'model_family':'selective_cart_absolute_saving',
        'training_dependency':{'scikit_learn':sklearn.__version__,'runtime_dependency_required':False},
        'feature_names':FEATURES,
        'default_backend':default,
        'alternate_backend':alternate,
        'target':'default_backend_cost - alternate_backend_cost',
        'tree_hyperparameters':{'max_depth':MAX_DEPTH,'min_samples_leaf':MIN_SAMPLES_LEAF,'criterion':'squared_error','random_state':RANDOM_STATE},
        'selective_rule':{'switch_margin_s':SWITCH_MARGIN_S,'bootstrap_lower_quantile':BOOTSTRAP_LOWER_Q,'bootstrap_draws':BOOTSTRAP_DRAWS,'require_leaf_samples_at_least':MIN_SAMPLES_LEAF,'require_in_leaf_support_for_tree_used_features':True},
        'tree_nodes':serialize_tree(final),
        'leaf_stats':stats,
        'development_instances':32,
        'test_instances':16,
        'test_outcomes_seen_during_fit':False,
        'split_sha256':split['split_sha256'],
        'trainval_sha256':sha(args.trainval),
        'protocol_sha256':sha(args.protocol),
        'development_gate_passed':dev_pass,
        'development_metrics':dev_metrics,
        'development_gates':gates,
        'claims_boundary':['development data were adaptively reused after M26; only the sealed M27 test can authorize production ranking','bootstrap leaf lower bounds are empirical safety statistics, not formal conformal coverage guarantees'],
    }
    model_payload['model_sha256']=canonical_hash(model_payload)
    dev_payload={
        'schema':'optimind.m27.development_oof_audit.v1','model_sha256':model_payload['model_sha256'],'split_sha256':split['split_sha256'],'protocol_sha256':sha(args.protocol),'metrics':dev_metrics,'gates':gates,'development_gate_passed':dev_pass,
        'rows':[{'instance':r['instance'],'group':groups[i],'switch_to_alternate':bool(oof_switch[i]),'default_cost_s':float(default_cost[i]),'alternate_cost_s':float(alternate_cost[i]),'inspect_cost_s':float(inspect[i]),'policy_cost_s':float(policy[i]),**oof_meta[i]} for i,r in enumerate(rows)],
        'test_outcomes_seen':False,
    }
    args.model_output.parent.mkdir(parents=True,exist_ok=True); args.dev_audit_output.parent.mkdir(parents=True,exist_ok=True)
    args.model_output.write_text(json.dumps(model_payload,indent=2,sort_keys=True)); args.dev_audit_output.write_text(json.dumps(dev_payload,indent=2,sort_keys=True))
    print(json.dumps({'model_sha256':model_payload['model_sha256'],'development_gate_passed':dev_pass,'metrics':dev_metrics,'gates':gates},indent=2))
    if not dev_pass: raise SystemExit(2)
if __name__=='__main__': main()
