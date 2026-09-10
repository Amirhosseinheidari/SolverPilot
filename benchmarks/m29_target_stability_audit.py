from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def _delta(ds: float, ipm: float) -> float:
    return float(ipm - ds)  # positive => simplex/DS faster


def _winner(delta: float, tol: float = 0.0) -> str:
    if delta > tol:
        return "simplex"
    if delta < -tol:
        return "ipm"
    return "tie"


def _opportunity_concentration(rows: list[dict], *, sbs: str) -> dict:
    gains = []
    for r in rows:
        ds = float(r["simplex"])
        ipm = float(r["ipm"])
        sbs_cost = ds if sbs == "simplex" else ipm
        vbs = min(ds, ipm)
        gains.append(max(0.0, sbs_cost - vbs))
    gains = np.sort(np.asarray(gains, dtype=float))[::-1]
    total = float(gains.sum())
    if total <= 0:
        return {"total_s": 0.0, "top1_share": 0.0, "top5_share": 0.0, "top10_share": 0.0, "positive_instances": 0}
    return {
        "total_s": total,
        "top1_share": float(gains[:1].sum() / total),
        "top5_share": float(gains[:5].sum() / total),
        "top10_share": float(gains[:10].sum() / total),
        "positive_instances": int(np.sum(gains > 0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m25-run1", type=Path, required=True)
    ap.add_argument("--m25-run2", type=Path, required=True)
    ap.add_argument("--m28", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    r1 = json.loads(args.m25_run1.read_text())
    r2 = json.loads(args.m25_run2.read_text())
    r28 = json.loads(args.m28.read_text())

    a1 = {r["instance"]: r for r in r1["rows"]}
    a2 = {r["instance"]: r for r in r2["rows"]}
    names25 = sorted(set(a1) & set(a2))
    d1 = np.asarray([_delta(a1[i]["ds_median_cost_s"], a1[i]["ipm_median_cost_s"]) for i in names25])
    d2 = np.asarray([_delta(a2[i]["ds_median_cost_s"], a2[i]["ipm_median_cost_s"]) for i in names25])
    robust25 = (d1 + d2) / 2.0
    sign_agree25 = np.sign(d1) == np.sign(d2)
    rel1 = np.asarray([d1[j] / max((a1[i]["ds_median_cost_s"] + a1[i]["ipm_median_cost_s"]) / 2.0, 1e-12) for j, i in enumerate(names25)])
    rel2 = np.asarray([d2[j] / max((a2[i]["ds_median_cost_s"] + a2[i]["ipm_median_cost_s"]) / 2.0, 1e-12) for j, i in enumerate(names25)])
    decisive25 = np.maximum(np.abs(rel1), np.abs(rel2)) >= 0.05

    rows25 = []
    for i in names25:
        ds = float(np.median([a1[i]["ds_median_cost_s"], a2[i]["ds_median_cost_s"]]))
        ipm = float(np.median([a1[i]["ipm_median_cost_s"], a2[i]["ipm_median_cost_s"]]))
        rows25.append({"instance": i, "simplex": ds, "ipm": ipm})
    mean_ds25 = float(np.mean([x["simplex"] for x in rows25]))
    mean_ipm25 = float(np.mean([x["ipm"] for x in rows25]))
    sbs25 = "simplex" if mean_ds25 <= mean_ipm25 else "ipm"

    # M28 round-level target stability.
    names28 = sorted(r["instance"] for r in r28["rows"])
    round_deltas = []
    unanim = []
    usable_round_counts = []
    rows28 = []
    for row in sorted(r28["rows"], key=lambda x: x["instance"]):
        ds_samples = row["samples"]["simplex"]
        ipm_samples = row["samples"]["ipm"]
        ds_times = [float(x["execute_wall_s"]) for x in ds_samples]
        ipm_times = [float(x["execute_wall_s"]) for x in ipm_samples]
        k = min(len(ds_times), len(ipm_times))
        deltas = np.asarray([_delta(ds_times[j], ipm_times[j]) for j in range(k)], dtype=float)
        round_deltas.append(deltas)
        nz = np.sign(deltas)
        unanim.append(bool(len(nz) > 0 and np.all(nz == nz[0])))
        usable_round_counts.append(k)
        rows28.append({"instance": row["instance"], "simplex": float(row["cost_medians_s"]["simplex"]), "ipm": float(row["cost_medians_s"]["ipm"])})
    max_rounds = min(len(x) for x in round_deltas)
    round_rhos = []
    for a in range(max_rounds):
        for b in range(a + 1, max_rounds):
            xa = np.asarray([x[a] for x in round_deltas])
            xb = np.asarray([x[b] for x in round_deltas])
            rho = float(spearmanr(xa, xb).statistic)
            round_rhos.append({"round_a": a, "round_b": b, "spearman_delta": rho})
    mean_ds28 = float(np.mean([x["simplex"] for x in rows28]))
    mean_ipm28 = float(np.mean([x["ipm"] for x in rows28]))
    sbs28 = "simplex" if mean_ds28 <= mean_ipm28 else "ipm"

    payload = {
        "schema": "optimind.m29.target_stability.v1",
        "claim_boundary": "post-outcome diagnostic only; cannot authorize M30 or production routing",
        "m25": {
            "instances": len(names25),
            "delta_rank_spearman_run1_vs_run2": float(spearmanr(d1, d2).statistic),
            "winner_sign_agreement_fraction": float(np.mean(sign_agree25)),
            "winner_sign_flip_count": int(np.sum(~sign_agree25)),
            "winner_sign_flip_instances": [names25[j] for j in range(len(names25)) if not sign_agree25[j]],
            "decisive_instances": int(np.sum(decisive25)),
            "decisive_sign_agreement_fraction": float(np.mean(sign_agree25[decisive25])) if np.any(decisive25) else None,
            "median_abs_robust_delta_s": float(np.median(np.abs(robust25))),
            "sbs": sbs25,
            "opportunity_concentration": _opportunity_concentration(rows25, sbs=sbs25),
        },
        "m28": {
            "instances": len(names28),
            "rounds_per_instance_min": int(min(usable_round_counts)),
            "unanimous_round_winner_fraction": float(np.mean(unanim)),
            "nonunanimous_instance_count": int(np.sum(~np.asarray(unanim, dtype=bool))),
            "round_pair_delta_spearman": round_rhos,
            "median_round_pair_delta_spearman": float(np.median([x["spearman_delta"] for x in round_rhos])) if round_rhos else None,
            "sbs": sbs28,
            "opportunity_concentration": _opportunity_concentration(rows28, sbs=sbs28),
        },
        "interpretation": {
            "target_noise_is_not_a_complete_explanation": bool(float(spearmanr(d1, d2).statistic) >= 0.8 and float(np.mean(sign_agree25)) >= 0.85),
            "opportunity_not_only_single_outlier_m25": bool(_opportunity_concentration(rows25, sbs=sbs25)["top1_share"] < 0.5),
            "opportunity_not_only_single_outlier_m28": bool(_opportunity_concentration(rows28, sbs=sbs28)["top1_share"] < 0.5),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
