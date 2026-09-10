from __future__ import annotations

import argparse
import ctypes
import json
import math
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from solverpilot.problem import LinearProblem, ObjectiveSense
from solverpilot.backends.bundled_capi import (
    _HIGHS_MATRIX_COLWISE,
    _HIGHS_OBJ_MAX,
    _HIGHS_OBJ_MIN,
    _HIGHS_STATUS_OK,
    _bundled_paths,
    _h_double,
    _h_int,
    _load_highs_capi,
)


def _qstats(values: np.ndarray, prefix: str) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"{prefix}_{k}": 0.0 for k in ("min","q10","q25","median","mean","q75","q90","max","std")}
    q10,q25,med,q75,q90=np.quantile(x,[0.10,0.25,0.50,0.75,0.90])
    return {
        f"{prefix}_min": float(np.min(x)), f"{prefix}_q10": float(q10), f"{prefix}_q25": float(q25),
        f"{prefix}_median": float(med), f"{prefix}_mean": float(np.mean(x)), f"{prefix}_q75": float(q75),
        f"{prefix}_q90": float(q90), f"{prefix}_max": float(np.max(x)), f"{prefix}_std": float(np.std(x)),
    }


def _safe_log_range(values: np.ndarray) -> float:
    x=np.abs(np.asarray(values,dtype=np.float64)); x=x[np.isfinite(x)&(x>0)]
    if x.size==0: return 0.0
    return float(math.log10(float(np.max(x))/float(np.min(x)))) if np.min(x)>0 else 0.0


def _gini(values: np.ndarray) -> float:
    x=np.asarray(values,dtype=np.float64)
    if x.size==0 or np.all(x==0): return 0.0
    x=np.sort(np.maximum(x,0.0)); n=x.size
    return float((2.0*np.sum((np.arange(1,n+1))*x)/(n*np.sum(x)))-((n+1)/n))


def base_features(p: LinearProblem) -> dict[str,float]:
    m,n=p.A.shape; nnz=p.A.nnz; den=float(nnz/(m*n)) if m and n else 0.0
    return {
        "log1p_n": float(np.log1p(n)), "log1p_m": float(np.log1p(m)), "log1p_nnz": float(np.log1p(nnz)),
        "density": den, "m_over_n": float(m/n) if n else 0.0,
    }


def static_features(p: LinearProblem) -> dict[str,float]:
    A=p.A.tocsr(); m,n=A.shape; out={}
    row_deg=np.diff(A.indptr).astype(float); col_deg=np.diff(A.tocsc().indptr).astype(float)
    out.update(_qstats(row_deg,"row_deg")); out.update(_qstats(col_deg,"col_deg"))
    out["row_singleton_fraction"]=float(np.mean(row_deg==1)) if m else 0.0
    out["col_singleton_fraction"]=float(np.mean(col_deg==1)) if n else 0.0

    # MIPLIB-style representation hardening: normalize each constraint row by
    # its maximum absolute matrix coefficient. This makes coefficient/RHS
    # statistics invariant to arbitrary row scaling in the MPS formulation.
    absA=A.copy(); absA.data=np.abs(absA.data)
    if m and A.nnz:
        row_max=np.asarray(absA.max(axis=1).toarray()).ravel()
        row_scale=np.where(row_max>0,row_max,1.0)
        row_ids=np.repeat(np.arange(m,dtype=np.int32),row_deg.astype(np.int64))
        norm_data=A.data/row_scale[row_ids]
    else:
        row_scale=np.ones(m,dtype=float); row_ids=np.zeros(0,dtype=np.int32); norm_data=A.data.copy()
    An=A.copy(); An.data=np.asarray(norm_data,dtype=np.float64)
    absAn=An.copy(); absAn.data=np.abs(absAn.data)

    data=An.data
    out["a_log10_dynamic_range"]=_safe_log_range(data)
    out["a_positive_fraction"]=float(np.mean(data>0)) if data.size else 0.0
    out["a_negative_fraction"]=float(np.mean(data<0)) if data.size else 0.0
    out.update(_qstats(np.abs(data),"a_abs"))

    # Objective normalization by ||c||_inf, following the same principle.
    c=p.c.astype(float,copy=False); cscale=float(np.max(np.abs(c))) if c.size and np.any(c!=0) else 1.0; cn=c/cscale
    out["objective_nnz_fraction"]=float(np.count_nonzero(c)/n) if n else 0.0
    out["objective_log10_dynamic_range"]=_safe_log_range(cn)
    out.update(_qstats(np.abs(cn[np.nonzero(cn)]),"objective_abs_nz"))

    cl,cu=p.constraint_lower,p.constraint_upper
    finite_l=np.isfinite(cl); finite_u=np.isfinite(cu)
    out["row_lower_finite_fraction"]=float(np.mean(finite_l)) if m else 0.0
    out["row_upper_finite_fraction"]=float(np.mean(finite_u)) if m else 0.0
    out["row_two_sided_fraction"]=float(np.mean(finite_l&finite_u)) if m else 0.0
    out["row_equality_fraction"]=float(np.mean(finite_l&finite_u&(cl==cu))) if m else 0.0
    ncl=cl.copy(); ncu=cu.copy()
    if m:
        ncl[finite_l]=ncl[finite_l]/row_scale[finite_l]; ncu[finite_u]=ncu[finite_u]/row_scale[finite_u]
    rhs=np.concatenate([np.abs(ncl[finite_l]),np.abs(ncu[finite_u])])
    out["rhs_log10_dynamic_range"]=_safe_log_range(rhs); out.update(_qstats(rhs,"rhs_abs"))

    vl,vu=p.variable_lower,p.variable_upper; fl=np.isfinite(vl); fu=np.isfinite(vu)
    out["var_lower_finite_fraction"]=float(np.mean(fl)) if n else 0.0
    out["var_upper_finite_fraction"]=float(np.mean(fu)) if n else 0.0
    out["var_fully_bounded_fraction"]=float(np.mean(fl&fu)) if n else 0.0
    out["var_free_fraction"]=float(np.mean((~fl)&(~fu))) if n else 0.0
    widths=(vu-vl)[fl&fu]; widths=widths[np.isfinite(widths)]
    # siglog summaries preserve sign while compressing bound scale, matching the
    # representation principle used in MIPLIB 2017.
    siglog=lambda x: np.sign(x)*np.log10(np.abs(x)+1.0)
    out.update(_qstats(siglog(vl[fl]),"var_lower_siglog")); out.update(_qstats(siglog(vu[fu]),"var_upper_siglog"))
    out.update(_qstats(np.log10(np.abs(widths)+1.0),"bound_width_log10p1")); out["bound_width_log10_dynamic_range"]=_safe_log_range(widths)

    # Norm summaries are computed on the normalized matrix, so they capture
    # sparsity/coefficient shape rather than arbitrary model units.
    row_l1=np.asarray(absAn.sum(axis=1)).ravel(); col_l1=np.asarray(absAn.sum(axis=0)).ravel()
    row_l2=np.sqrt(np.asarray(An.multiply(An).sum(axis=1)).ravel()); col_l2=np.sqrt(np.asarray(An.multiply(An).sum(axis=0)).ravel())
    out.update(_qstats(row_l1,"row_l1")); out.update(_qstats(col_l1,"col_l1")); out.update(_qstats(row_l2,"row_l2")); out.update(_qstats(col_l2,"col_l2"))

    if m and data.size:
        pos=np.bincount(row_ids,weights=(data>0).astype(np.float64),minlength=m)
        neg=np.bincount(row_ids,weights=(data<0).astype(np.float64),minlength=m)
        denom=np.maximum(row_deg,1.0)
        sign_imb=np.abs(pos-neg)/denom
        row_abs_mean=row_l1/denom
        # row max is exactly one for non-empty normalized rows
        row_scale_ratio=np.where(row_deg>0,1.0/np.maximum(row_abs_mean,1e-300),1.0)
        row_log_scale_ratio=np.log10(np.maximum(row_scale_ratio,1.0))
    else:
        sign_imb=np.zeros(m,dtype=float); row_log_scale_ratio=np.zeros(m,dtype=float)
    out.update(_qstats(row_log_scale_ratio,"row_log10_dynamism")); out.update(_qstats(sign_imb,"row_sign_imbalance"))
    return out


def topology_features(p: LinearProblem) -> dict[str,float]:
    A=p.A.tocsr(); m,n=A.shape; row_deg=np.diff(A.indptr).astype(float); Ac=A.tocsc(); col_deg=np.diff(Ac.indptr).astype(float)
    out={
        "row_degree_gini":_gini(row_deg), "col_degree_gini":_gini(col_deg),
        "row_degree_max_over_mean":float(np.max(row_deg)/(np.mean(row_deg)+1e-12)) if row_deg.size else 0.0,
        "col_degree_max_over_mean":float(np.max(col_deg)/(np.mean(col_deg)+1e-12)) if col_deg.size else 0.0,
    }
    # Bipartite structural graph; values ignored.
    if m+n==0 or A.nnz==0:
        out.update({"bipartite_components":float(m+n),"bipartite_largest_component_fraction":0.0,"bipartite_component_entropy":0.0})
        return out
    B=sparse.bmat([[None,(A!=0).astype(np.int8)],[(A.T!=0).astype(np.int8),None]],format="csr")
    nc,labels=csgraph.connected_components(B,directed=False,return_labels=True)
    counts=np.bincount(labels,minlength=nc).astype(float); probs=counts/np.sum(counts)
    entropy=-float(np.sum(probs*np.log(probs+1e-300)))
    out["bipartite_components"]=float(nc)
    out["bipartite_largest_component_fraction"]=float(np.max(counts)/np.sum(counts))
    out["bipartite_component_entropy"]=entropy
    return out


def presolve_probe(p: LinearProblem) -> tuple[dict[str,float],float]:
    paths=_bundled_paths()
    if paths is None: raise RuntimeError("bundled HiGHS paths unavailable")
    lib=_load_highs_capi(paths)
    # Missing prototypes in the M13 bridge are declared here for the research probe.
    lib.Highs_presolve.argtypes=[ctypes.c_void_p]; lib.Highs_presolve.restype=ctypes.c_int32
    lib.Highs_getPresolvedNumCol.argtypes=[ctypes.c_void_p]; lib.Highs_getPresolvedNumCol.restype=ctypes.c_int32
    lib.Highs_getPresolvedNumRow.argtypes=[ctypes.c_void_p]; lib.Highs_getPresolvedNumRow.restype=ctypes.c_int32
    lib.Highs_getPresolvedNumNz.argtypes=[ctypes.c_void_p]; lib.Highs_getPresolvedNumNz.restype=ctypes.c_int32
    lib.Highs_resetGlobalScheduler(1)
    h=lib.Highs_create()
    if not h: raise RuntimeError("Highs_create returned null")
    try:
        lib.Highs_setBoolOptionValue(h,b"output_flag",0); lib.Highs_setIntOptionValue(h,b"threads",1)
        A=p.A.tocsc(copy=True); A.sort_indices(); sense=_HIGHS_OBJ_MIN if p.objective_sense is ObjectiveSense.MINIMIZE else _HIGHS_OBJ_MAX
        t0=perf_counter()
        rc=int(lib.Highs_passLp(h,p.n_variables,p.n_constraints,p.nnz,_HIGHS_MATRIX_COLWISE,sense,float(p.objective_offset),_h_double(p.c),_h_double(p.variable_lower),_h_double(p.variable_upper),_h_double(p.constraint_lower),_h_double(p.constraint_upper),_h_int(A.indptr),_h_int(A.indices),_h_double(A.data)))
        if rc!=_HIGHS_STATUS_OK: raise RuntimeError(f"Highs_passLp={rc}")
        prc=int(lib.Highs_presolve(h)); wall=perf_counter()-t0
        if prc<0: raise RuntimeError(f"Highs_presolve={prc}")
        pc=int(lib.Highs_getPresolvedNumCol(h)); pr=int(lib.Highs_getPresolvedNumRow(h)); pnz=int(lib.Highs_getPresolvedNumNz(h))
        out={
            "presolved_n":float(pc),"presolved_m":float(pr),"presolved_nnz":float(pnz),
            "presolve_col_reduction_fraction":float(1-pc/max(p.n_variables,1)),
            "presolve_row_reduction_fraction":float(1-pr/max(p.n_constraints,1)),
            "presolve_nnz_reduction_fraction":float(1-pnz/max(p.nnz,1)),
        }
        return out,wall
    finally:
        lib.Highs_destroy(h); lib.Highs_resetGlobalScheduler(1)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--instance',required=True); args=ap.parse_args()
    p=pickle.loads(args.cache.read_bytes())
    if not isinstance(p,LinearProblem): raise TypeError(type(p))
    t0=perf_counter(); fa=base_features(p); ta=perf_counter()-t0
    t0=perf_counter(); fb=static_features(p); tb=perf_counter()-t0
    t0=perf_counter(); fc=topology_features(p); tc=perf_counter()-t0
    probe_error=None
    try: fd,td=presolve_probe(p)
    except Exception as exc: fd={}; td=0.0; probe_error=f"{type(exc).__name__}: {exc}"
    print(json.dumps({"instance":args.instance,"features":{"A":fa,"B":fb,"C":fc,"D":fd},"cost_s":{"A":ta,"B":tb,"C":tc,"D":td},"probe_error":probe_error},sort_keys=True))

if __name__=='__main__': main()
