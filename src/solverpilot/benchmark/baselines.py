from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Iterable, Mapping

from .summary import summarize_rows


@dataclass(frozen=True, slots=True)
class BaselineFit:
    baseline_type: str
    selected_backend: str | None
    training_instances_sha256: str
    parameters: Mapping[str, object]


def _instance_digest(instances: Iterable[str]) -> str:
    canonical = "\n".join(sorted(map(str, instances))).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _entry(backend: str, overhead_s: float, *, component: str) -> dict:
    overhead=float(overhead_s)
    if overhead < 0: raise ValueError("overhead_s must be non-negative")
    return {"backend":str(backend),"overhead_s":overhead,"overhead_components_s":{component:overhead}}


def fixed_backend_policy(instances: Iterable[str], backend: str, *, overhead_s: float = 0.0) -> dict[str, dict]:
    return {str(instance):_entry(backend,overhead_s,component="fixed_policy") for instance in sorted(set(instances))}


def seeded_random_policy(instances: Iterable[str], backends: Iterable[str], *, seed: int, overhead_s: float = 0.0) -> dict[str, dict]:
    choices=tuple(sorted(set(map(str,backends))))
    if not choices: raise ValueError("backends must not be empty")
    out={}
    for instance in sorted(set(map(str,instances))):
        local_seed=int.from_bytes(hashlib.sha256(f"{seed}\0{instance}".encode()).digest()[:8],"big")
        out[instance]=_entry(random.Random(local_seed).choice(choices),overhead_s,component="random_selection")
    return out


def threshold_policy(features: Mapping[str, Mapping[str, float]], *, feature: str, threshold: float, le_backend: str, gt_backend: str, overhead_s: float) -> dict[str, dict]:
    out={}
    for instance in sorted(features):
        row=features[instance]
        if feature not in row: raise ValueError(f"feature {feature!r} missing for instance {instance!r}")
        value=float(row[feature]); backend=le_backend if value <= threshold else gt_backend
        entry=_entry(backend,overhead_s,component="rule_selection"); entry["rule"]={"feature":feature,"threshold":float(threshold),"value":value}; out[instance]=entry
    return out


def fit_sbs_backend(rows: list[dict], *, training_instances: Iterable[str], cutoff_s: float, par_penalty: float = 10.0, cost_field: str = "wall_s") -> BaselineFit:
    train=tuple(sorted(set(map(str,training_instances))))
    if not train: raise ValueError("training_instances must not be empty")
    train_set=set(train); filtered=[r for r in rows if str(r.get("instance")) in train_set and r.get("backend") != "@auto"]
    if not filtered: raise ValueError("no benchmark rows match training_instances")
    summary=summarize_rows(filtered,cutoff_s=cutoff_s,par_penalty=par_penalty,cost_field=cost_field,bootstrap_draws=100,bootstrap_seed=0)
    return BaselineFit("sbs-on-training-split",str(summary["sbs_solver"]),_instance_digest(train),{"cutoff_s":float(cutoff_s),"par_penalty":float(par_penalty),"cost_field":cost_field,"training_instances":len(train)})


def fit_sbs_backend_from_split(rows: list[dict], split_records, *, fit_splits: tuple[str, ...] = ("train",), cutoff_s: float, par_penalty: float = 10.0, cost_field: str = "wall_s") -> BaselineFit:
    normalized=tuple(str(x).lower() for x in fit_splits)
    if "test" in normalized: raise ValueError("held-out test split must not be used to fit an SBS baseline")
    from .splits import filter_rows_by_split, instances_by_split
    filtered=filter_rows_by_split(rows,split_records,allowed_splits=normalized); grouped=instances_by_split(split_records)
    training=[]
    for name in normalized: training.extend(grouped[name])
    return fit_sbs_backend(filtered,training_instances=training,cutoff_s=cutoff_s,par_penalty=par_penalty,cost_field=cost_field)
