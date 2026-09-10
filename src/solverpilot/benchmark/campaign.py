from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from solverpilot.evaluation import parse_test_manifest
from .acquire import dataset_spec_fingerprint, sha256_file
from .model import RunSpec
from .runner import run_benchmark
from .specs import MIPLIB2017_BENCHMARK_V2, QPLIB_ARCHIVE
from .splits import validate_split_file
from .verify import verify_acquired_dataset

CAMPAIGN_SCHEMA_VERSION = "solverpilot.benchmark-campaign.v1"
_DEFAULT_METRICS = ("par2","par10","solved_count","timeout_rate","optimality_gap","regret","selection_overhead")
_ALLOWED_COST_FIELDS = {"wall_s","worker_solve_wall_s","trace.solve_s","trace.total_s"}


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    out={}
    for key,item in value.items():
        if isinstance(item,Mapping): out[str(key)]=_freeze_mapping(item)
        elif isinstance(item,list): out[str(key)]=tuple(item)
        else: out[str(key)]=item
    return MappingProxyType(out)

def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class CampaignSpec:
    campaign_id: str
    dataset_name: str
    dataset_spec_sha256: str
    backends: tuple[str, ...]
    repetitions: int = 1
    cutoff_s: float = 600.0
    hard_timeout_s: float | None = None
    objective_atol: float = 1e-6
    objective_rtol: float = 1e-7
    seed: int = 0
    thread_env_limit: int | None = 1
    solver_threads: int | None = 1
    worker_python_mode: str = "no_site"
    cost_field: str = "wall_s"
    metrics: tuple[str, ...] = _DEFAULT_METRICS
    split_sha256: str | None = None
    require_scientific_provenance: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = CAMPAIGN_SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != CAMPAIGN_SCHEMA_VERSION: raise ValueError(f"unsupported campaign schema: {self.schema_version!r}")
        if not self.campaign_id.strip() or not self.dataset_name.strip(): raise ValueError("campaign_id/dataset_name must be non-empty")
        if len(self.dataset_spec_sha256)!=64 or any(c not in "0123456789abcdefABCDEF" for c in self.dataset_spec_sha256): raise ValueError("dataset_spec_sha256 must be a SHA-256 digest")
        if self.split_sha256 is not None and (len(self.split_sha256)!=64 or any(c not in "0123456789abcdefABCDEF" for c in self.split_sha256)): raise ValueError("split_sha256 must be a SHA-256 digest")
        if not self.backends or len(set(self.backends)) != len(self.backends): raise ValueError("backends must be non-empty and unique")
        if self.repetitions < 1: raise ValueError("repetitions must be >= 1")
        if self.cutoff_s <= 0: raise ValueError("cutoff_s must be positive")
        if self.hard_timeout_s is not None and self.hard_timeout_s < self.cutoff_s: raise ValueError("hard_timeout_s must be >= cutoff_s when provided")
        if self.thread_env_limit is not None and self.thread_env_limit < 1: raise ValueError("thread_env_limit must be >= 1")
        if self.solver_threads is not None and self.solver_threads < 1: raise ValueError("solver_threads must be >= 1")
        if self.worker_python_mode not in {"normal","no_site"}: raise ValueError("worker_python_mode must be 'normal' or 'no_site'")
        if self.cost_field not in _ALLOWED_COST_FIELDS: raise ValueError(f"unsupported campaign cost_field: {self.cost_field!r}")
        object.__setattr__(self,"backends",tuple(map(str,self.backends))); object.__setattr__(self,"metrics",tuple(map(str,self.metrics))); object.__setattr__(self,"metadata",_freeze_mapping(dict(self.metadata)))

    def to_dict(self) -> dict[str,Any]:
        def thaw(v):
            if isinstance(v,Mapping): return {str(k):thaw(x) for k,x in v.items()}
            if isinstance(v,tuple): return [thaw(x) for x in v]
            return v
        return {"campaign_id":self.campaign_id,"dataset_name":self.dataset_name,"dataset_spec_sha256":self.dataset_spec_sha256,"backends":list(self.backends),"repetitions":self.repetitions,"cutoff_s":self.cutoff_s,"hard_timeout_s":self.hard_timeout_s,"objective_atol":self.objective_atol,"objective_rtol":self.objective_rtol,"seed":self.seed,"thread_env_limit":self.thread_env_limit,"solver_threads":self.solver_threads,"worker_python_mode":self.worker_python_mode,"cost_field":self.cost_field,"metrics":list(self.metrics),"split_sha256":self.split_sha256,"require_scientific_provenance":self.require_scientific_provenance,"metadata":thaw(self.metadata),"schema_version":self.schema_version}
    @property
    def sha256(self): return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class CampaignValidation:
    ok: bool
    campaign_sha256: str
    scientific_provenance_verified: bool | None
    split_verified: bool | None
    errors: tuple[str,...]=()
    warnings: tuple[str,...]=()
    def to_dict(self): return asdict(self)


def save_campaign_spec(spec: CampaignSpec,path:str|Path)->None: Path(path).write_text(json.dumps(spec.to_dict(),indent=2,sort_keys=True),encoding="utf-8")
def load_campaign_spec(path:str|Path)->CampaignSpec:
    p=json.loads(Path(path).read_text(encoding="utf-8")); p["backends"]=tuple(p.get("backends",())); p["metrics"]=tuple(p.get("metrics",_DEFAULT_METRICS)); return CampaignSpec(**p)

def miplib2017_campaign_template(backends: tuple[str,...], *, repetitions:int=1, cutoff_s:float=3600.0, hard_timeout_s:float|None=None, seed:int=0, solver_threads:int|None=1)->CampaignSpec:
    return CampaignSpec(campaign_id="miplib2017-benchmark-v2",dataset_name=MIPLIB2017_BENCHMARK_V2.name,dataset_spec_sha256=dataset_spec_fingerprint(MIPLIB2017_BENCHMARK_V2),backends=backends,repetitions=repetitions,cutoff_s=cutoff_s,hard_timeout_s=hard_timeout_s,seed=seed,solver_threads=solver_threads,require_scientific_provenance=True,metadata={"benchmark_instances":240,"benchmark_version":2,"reference_solution_version":36,"reference_solution_release":"2026-01-26","official_source":"https://miplib.zib.de/download.html","status":"protocol-ready; execution requires fully pinned official inputs"})

def qplib_campaign_template(backends: tuple[str,...], *, cutoff_s:float=3600.0, seed:int=0)->CampaignSpec:
    return CampaignSpec(campaign_id="qplib-supported-subset-v1",dataset_name=QPLIB_ARCHIVE.name,dataset_spec_sha256=dataset_spec_fingerprint(QPLIB_ARCHIVE),backends=backends,cutoff_s=cutoff_s,seed=seed,require_scientific_provenance=True,metadata={"official_continuous_instances":134,"official_discrete_instances":319,"supported_subset_count":None,"execution_supported":False,"blocked_on":"strict QPLIB parser/filter and pinned supported-subset manifest/reference","official_source":"https://qplib.zib.de/"})

def _lock_payload(target:Path):
    lock=target/"acquisition-lock.json"
    if not lock.exists(): raise FileNotFoundError(lock)
    return json.loads(lock.read_text(encoding="utf-8"))
def _role_path(lock:Mapping[str,Any],role:str)->Path|None:
    for item in lock.get("files",[]):
        if isinstance(item,Mapping) and item.get("role")==role: return Path(str(item["path"]))
    return None

def validate_campaign(spec:CampaignSpec, *, dataset_target:str|Path|None=None, split_path:str|Path|None=None)->CampaignValidation:
    errors=[]; warnings=[]; scientific=None; split_verified=None; expected_split_instances=None
    if dataset_target is not None:
        target=Path(dataset_target); report=verify_acquired_dataset(target); scientific=report.scientific_provenance_verified; lock=_lock_payload(target)
        if str(lock.get("dataset"))!=spec.dataset_name: errors.append("campaign dataset_name does not match acquisition lock")
        if str(lock.get("spec_fingerprint"))!=spec.dataset_spec_sha256: errors.append("campaign dataset_spec_sha256 does not match acquisition lock")
        if not report.ok: errors.extend(report.errors)
        if spec.require_scientific_provenance and not report.scientific_provenance_verified: errors.append("campaign requires scientifically verified provenance, but acquisition is not fully pinned")
        if not report.scientific_provenance_verified: warnings.append("dataset integrity may be locally verified without authenticating the official snapshot")
        manifest=_role_path(lock,"manifest")
        if manifest is not None and manifest.exists():
            names=parse_test_manifest(manifest.read_text(encoding="utf-8"))
            def key(filename):
                lower=filename.lower()
                for suffix in (".mps.gz",".mps",".lp.gz",".lp"):
                    if lower.endswith(suffix): return filename[:-len(suffix)]
                return Path(filename).stem
            expected_split_instances=tuple(key(n) for n in names)
    if split_path is None and spec.split_sha256 is not None: errors.append("campaign binds split_sha256 but no split_path was supplied")
    if split_path is not None:
        path=Path(split_path); digest=sha256_file(path)
        if spec.split_sha256 is not None and digest.lower()!=spec.split_sha256.lower(): errors.append("split artifact SHA-256 does not match campaign")
        sr=validate_split_file(path,expected_instances=expected_split_instances); split_verified=sr.ok
        if not sr.ok: errors.extend(sr.errors)
    return CampaignValidation(not errors,spec.sha256,scientific,split_verified,tuple(errors),tuple(warnings))

def run_campaign(spec:CampaignSpec, *, dataset_target:str|Path, output_jsonl:str|Path, split_path:str|Path|None=None, allow_unverified_dataset:bool=False, registry=None)->dict[str,Any]:
    validation=validate_campaign(spec,dataset_target=dataset_target,split_path=split_path)
    if not validation.ok:
        if not allow_unverified_dataset: raise ValueError("campaign validation failed: "+"; ".join(validation.errors))
        remaining=[e for e in validation.errors if "scientifically verified provenance" not in e]
        if remaining: raise ValueError("campaign validation failed: "+"; ".join(remaining))
    target=Path(dataset_target); lock=_lock_payload(target); manifest=_role_path(lock,"manifest"); reference=_role_path(lock,"reference")
    if manifest is None: raise ValueError("campaign dataset acquisition has no manifest")
    if spec.require_scientific_provenance and reference is None: raise ValueError("scientific campaign requires a reference file")
    rs=RunSpec(dataset_dir=target/"instances",manifest=manifest,reference=reference,backends=spec.backends,output_jsonl=Path(output_jsonl),repetitions=spec.repetitions,time_limit_s=spec.cutoff_s,hard_timeout_s=spec.hard_timeout_s,objective_atol=spec.objective_atol,objective_rtol=spec.objective_rtol,seed=spec.seed,thread_env_limit=spec.thread_env_limit,solver_threads=spec.solver_threads,worker_python_mode=spec.worker_python_mode)
    summary=run_benchmark(rs,registry=registry); result_path=Path(output_jsonl)
    sidecar={"schema_version":"solverpilot.campaign-run.v1","campaign_id":spec.campaign_id,"campaign_sha256":spec.sha256,"scientific_provenance_verified":bool(validation.scientific_provenance_verified),"allow_unverified_dataset":bool(allow_unverified_dataset),"deployable_scientific_evidence":bool(validation.scientific_provenance_verified and not allow_unverified_dataset),"result_jsonl":str(result_path),"result_jsonl_sha256":sha256_file(result_path),"run_summary":summary}
    Path(str(result_path)+".campaign.json").write_text(json.dumps(sidecar,indent=2,sort_keys=True,default=str),encoding="utf-8"); return sidecar
