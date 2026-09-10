from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from .campaign import CampaignSpec


def scientific_report_payload(campaign: CampaignSpec, summary: Mapping[str,Any], *, campaign_run: Mapping[str,Any]|None=None, limitations: tuple[str,...]=()) -> dict[str,Any]:
    vbs=summary.get("vbs") or {"cost":summary.get("vbs_cost"),"is_oracle":True,"deployable":False}
    if vbs.get("deployable") is not False or vbs.get("is_oracle") is not True:
        raise ValueError("scientific report requires VBS to be explicitly oracle/non-deployable")
    provenance_ok=None if campaign_run is None else bool(campaign_run.get("scientific_provenance_verified"))
    deployable=bool(provenance_ok and not summary.get("mixed_environments_allowed",False) and summary.get("instance_identity_verified") is True)
    return {"schema_version":"solverpilot.scientific-report.v1","generated_at_utc":datetime.now(timezone.utc).isoformat(),"campaign":campaign.to_dict(),"campaign_sha256":campaign.sha256,"scientific_provenance_verified":provenance_ok,"deployable_scientific_evidence":deployable,"instances":summary.get("instances"),"backends":summary.get("backends"),"cost_field":summary.get("cost_field"),"sbs":{"backend":summary.get("sbs_solver"),"cost":summary.get("sbs_cost")},"vbs":vbs,"sbs_vbs_gap":summary.get("sbs_vbs_gap"),"relative_sbs_vbs_gap":summary.get("relative_sbs_vbs_gap"),"solver_stats":summary.get("solver_stats"),"auto_policy_metrics":summary.get("auto_policy_metrics"),"limitations":list(limitations),"claim_boundary":"performance claims may be made only when deployable_scientific_evidence=true; VBS is an oracle baseline and is never a deployable selector"}


def render_scientific_report_markdown(payload: Mapping[str,Any])->str:
    sbs=payload.get("sbs") or {}; vbs=payload.get("vbs") or {}
    lines=["# SolverPilot Scientific Benchmark Report","",f"- Campaign: `{payload.get('campaign',{}).get('campaign_id')}`",f"- Campaign SHA-256: `{payload.get('campaign_sha256')}`",f"- Scientific provenance verified: `{payload.get('scientific_provenance_verified')}`",f"- Deployable scientific evidence: `{payload.get('deployable_scientific_evidence')}`",f"- Instances: `{payload.get('instances')}`",f"- Cost field: `{payload.get('cost_field')}`","","## Baselines","",f"- SBS: `{sbs.get('backend')}` — cost `{sbs.get('cost')}`",f"- VBS oracle cost: `{vbs.get('cost')}` (oracle=`{vbs.get('is_oracle')}`, deployable=`{vbs.get('deployable')}`)","","## Claim boundary","",str(payload.get("claim_boundary"))]
    if payload.get("limitations"): lines += ["","## Limitations",""]+[f"- {x}" for x in payload["limitations"]]
    return "\n".join(lines)+"\n"


def write_scientific_report(payload:Mapping[str,Any], *, json_path:str|Path, markdown_path:str|Path)->None:
    Path(json_path).write_text(json.dumps(payload,indent=2,sort_keys=True,default=str),encoding="utf-8")
    Path(markdown_path).write_text(render_scientific_report_markdown(payload),encoding="utf-8")


def compute_scientific_metrics(rows:list[dict[str,Any]], *, cutoff_s:float, cost_field:str="wall_s")->dict[str,Any]:
    from .summary import summarize_rows
    par2=summarize_rows(rows,cutoff_s=cutoff_s,par_penalty=2.0,cost_field=cost_field); par10=summarize_rows(rows,cutoff_s=cutoff_s,par_penalty=10.0,cost_field=cost_field)
    direct=[r for r in rows if r.get("backend") != "@auto"]; backends=sorted({str(r.get("backend")) for r in direct})
    timeout={b:sum(1 for r in direct if str(r.get("backend"))==b and (r.get("state")=="hard_timeout" or r.get("public_status")=="feasible_limit")) for b in backends}
    gaps={b:[] for b in backends}
    for row in direct:
        raw=row.get("relative_gap",row.get("optimality_gap"))
        if raw is None: continue
        try: value=float(raw)
        except (TypeError,ValueError): continue
        if value>=0: gaps[str(row.get("backend"))].append(value)
    return {"schema_version":"solverpilot.scientific-metrics.v1","cost_field":cost_field,"cutoff_s":float(cutoff_s),"par2":{"sbs_solver":par2["sbs_solver"],"sbs_cost":par2["sbs_cost"],"vbs_cost":par2["vbs_cost"],"solver_stats":par2["solver_stats"]},"par10":{"sbs_solver":par10["sbs_solver"],"sbs_cost":par10["sbs_cost"],"vbs_cost":par10["vbs_cost"],"solver_stats":par10["solver_stats"]},"timeout_count":timeout,"mean_reported_relative_gap":{b:(None if not v else sum(v)/len(v)) for b,v in gaps.items()},"optimality_gap_availability":"explicit_row_field_only; null means the backend/run did not expose a normalized gap","regret":"evaluate through solverpilot.evaluation oracle/regret APIs on a frozen held-out split","selection_overhead":"included only through deployable policy evaluation with measured overhead"}
