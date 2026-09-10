from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Mapping, Any
from solverpilot.inspect import ProblemFingerprint

FEATURE_NAMES=('log1p_n','log1p_m','log1p_nnz','log_density','log1p_m_over_n')

@dataclass(frozen=True, slots=True)
class SelectiveLPDecision:
    backend: str
    default_backend: str
    alternate_backend: str
    switched: bool
    leaf: int
    predicted_saving_s: float
    lower_saving_s: float
    support_passed: bool
    reason: str


def _features(fp: ProblemFingerprint) -> tuple[float,...]:
    if fp.problem_class != 'lp':
        raise ValueError(f'M27 LP selector only supports continuous LP fingerprints, got {fp.problem_class!r}')
    n=max(int(fp.n_variables),0); m=max(int(fp.n_constraints),0); nnz=max(int(fp.nnz_a),0)
    return (
        math.log1p(n), math.log1p(m), math.log1p(nnz),
        math.log(max(float(fp.density_a),1e-12)), math.log1p(m/max(n,1)),
    )


def _traverse(nodes: list[Mapping[str,Any]], x: tuple[float,...]) -> int:
    by_id={int(n['node']):n for n in nodes}; node=0
    while True:
        row=by_id[node]
        if bool(row['is_leaf']): return node
        f=int(row['feature']); thr=float(row['threshold'])
        node=int(row['left'] if x[f] <= thr else row['right'])


def decide_selective_lp_backend(fingerprint: ProblemFingerprint, model: Mapping[str,Any]) -> SelectiveLPDecision:
    if str(model.get('schema')) != 'solverpilot.m27.selective_absolute_saving_tree.v1':
        raise ValueError('unsupported M27 selector schema')
    if bool(model.get('test_outcomes_seen_during_fit')):
        raise ValueError('selector artifact is contaminated: test outcomes seen during fit')
    if not bool(model.get('development_gate_passed')):
        raise ValueError('selector development gate did not pass')
    x=_features(fingerprint); nodes=list(model['tree_nodes']); leaf=_traverse(nodes,x)
    stats=model['leaf_stats'][str(leaf)]; rule=model['selective_rule']; margin=float(rule['switch_margin_s'])
    mean=float(stats['mean_saving_s']); lower=float(stats['bootstrap_one_sided_90_lower_mean_s']); samples=int(stats['samples'])
    support=True
    for key,bounds in stats.get('support_used_features',{}).items():
        f=int(key); v=x[f]
        if v < float(bounds['min'])-1e-12 or v > float(bounds['max'])+1e-12:
            support=False; break
    switch=bool(mean>margin and lower>margin and samples>=int(rule['require_leaf_samples_at_least']) and support)
    default=str(model['default_backend']); alternate=str(model['alternate_backend']); backend=alternate if switch else default
    reasons=[]
    if switch: reasons.append('alternate selected: leaf mean and one-sided empirical lower saving exceed margin within observed leaf support')
    else:
        if mean<=margin: reasons.append('fallback: predicted leaf mean saving does not exceed margin')
        if lower<=margin: reasons.append('fallback: empirical lower saving does not exceed margin')
        if not support: reasons.append('fallback: query extrapolates outside observed leaf support')
        if samples<int(rule['require_leaf_samples_at_least']): reasons.append('fallback: insufficient leaf support count')
    return SelectiveLPDecision(backend,default,alternate,switch,leaf,mean,lower,support,'; '.join(reasons))
