"""Prepare outcome-independent public LP selection; never solves a model.

Metadata inputs are captured from the official Netlib README and MIPLIB benchmark
table. Raw third-party instances stay outside the repository. Netlib's official
emps decoder must be supplied explicitly. All exclusions and byte hashes persist.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.request

import highspy
import numpy as np
from scipy import sparse
from solverpilot import LinearProblem, read_mps


def sha(data):
    return hashlib.sha256(data).hexdigest()


def family(name):
    return re.split(r"[0-9]", name.lower(), maxsplit=1)[0].rstrip("-_.") or name.lower()


def fetch(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        data = response.read(8_000_001)
    if len(data) > 8_000_000:
        raise ValueError("download exceeds 8 MB source cap")
    return data


def relax(p):
    return LinearProblem.from_data(A=p.A, c=p.c, variable_lower=p.variable_lower,
        variable_upper=p.variable_upper, constraint_lower=p.constraint_lower,
        constraint_upper=p.constraint_upper, objective_sense=p.objective_sense,
        objective_offset=p.objective_offset)


def crosscheck_reader(path, p):
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    if h.readModel(str(path)) != highspy.HighsStatus.kOk:
        raise ValueError("independent HiGHS MPS reader rejected file")
    lp = h.getLp()
    a = lp.a_matrix_
    if a.format_ == highspy.MatrixFormat.kColwise:
        matrix = sparse.csc_matrix((a.value_, a.index_, a.start_), shape=(lp.num_row_, lp.num_col_)).tocsr()
    else:
        matrix = sparse.csr_matrix((a.value_, a.index_, a.start_), shape=(lp.num_row_, lp.num_col_))
    matrix.sum_duplicates(); matrix.sort_indices()
    if matrix.shape != p.A.shape or (matrix != p.A).nnz:
        raise ValueError("independent MPS matrix disagreement")
    for left, right in [(lp.col_cost_, p.c), (lp.col_lower_, p.variable_lower),
                        (lp.col_upper_, p.variable_upper), (lp.row_lower_, p.constraint_lower),
                        (lp.row_upper_, p.constraint_upper)]:
        if not np.array_equal(np.asarray(left), right):
            raise ValueError("independent MPS vector disagreement")
    sense = "minimize" if lp.sense_ == highspy.ObjSense.kMinimize else "maximize"
    if float(lp.offset_) != p.objective_offset or sense != p.objective_sense.value:
        raise ValueError("independent MPS objective disagreement")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--decoder", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    audit = json.loads((args.metadata / "prior-audit.json").read_text())
    prior = set(audit["names"])
    metadata = json.loads((args.metadata / "miplib-metadata.json").read_text())
    prior_groups = {r["group"] for r in metadata if r["name"].lower() in prior and r["group"] not in {"", "-", "–"}}
    candidates = []
    for line in (args.metadata / "netlib-readme").read_text().splitlines():
        match = re.match(r"^([A-Z0-9][A-Z0-9.\-]*)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+[BR ]*\s+([+-]?[0-9.]+[ED][+-]\d+)", line)
        if match:
            name, m, n, nnz, size, reference = match.groups()
            candidates.append(dict(source="netlib", name=name.lower(), n=int(n), m=int(m)-1,
                nnz=int(nnz), group="netlib:"+family(name), reference=float(reference.replace("D", "E"))))
    for r in metadata:
        candidates.append({**r, "source": "miplib", "reference": None,
            "group": "miplib:" + (r["group"] if r["group"] not in {"", "-", "–"} else family(r["name"]))})
    prior_netlib_groups = {"netlib:"+family(n) for n in prior}
    report = {"selection": "SHA256(source:name) order; first 12 importable unique groups per source after prior and structural exclusions; no solve outcomes used",
              "caps": {"n":50000,"m":50000,"nnz":500000}, "prior_audit": audit,
              "metadata_sha256": {f: sha((args.metadata/f).read_bytes()) for f in
                                  ("prior-audit.json","netlib-readme","miplib-metadata.json","miplib-metadata.html","miplib-benchmark.test","emps.c")},
              "selected": [], "exclusions": []}
    selected_groups, hashes = set(), set()
    for r in sorted(candidates, key=lambda r: sha((r["source"]+":"+r["name"]).encode())):
        source, name = r["source"], r["name"]
        reason = None
        if name.lower() in prior:
            reason = "prior_instance"
        elif (source == "miplib" and r["group"].removeprefix("miplib:") in prior_groups) or (source == "netlib" and r["group"] in prior_netlib_groups):
            reason = "prior_family"
        elif r["n"] > 50000 or r["m"] > 50000 or r["nnz"] > 500000:
            reason = "structural_cap"
        elif r["group"] in selected_groups:
            reason = "already_selected_group"
        elif sum(x["source"] == source for x in report["selected"]) >= 12:
            reason = "source_quota_filled"
        if reason:
            report["exclusions"].append({**r, "reason": reason})
            continue
        try:
            if source == "netlib":
                url = "https://www.netlib.org/lp/data/" + name
                raw = fetch(url)
                proc = subprocess.run([str(args.decoder), "-b"], input=raw, capture_output=True, timeout=15, check=True)
                mps = proc.stdout
            else:
                detail_url = "https://miplib.zib.de/instance_details_" + name + ".html"
                detail = fetch(detail_url).decode()
                links = re.findall(r'href=["\x27]([^"\x27]+\.mps\.gz)["\x27]', detail)
                if len(links) != 1:
                    raise ValueError("missing or ambiguous official MPS link")
                url = urllib.parse.urljoin(detail_url, links[0])
                raw = fetch(url)
                mps = gzip.decompress(raw)
            if len(mps) > 80_000_000:
                raise ValueError("expanded MPS exceeds cap")
            path = args.output / (source + "-" + name + ".mps")
            path.write_bytes(mps)
            p = read_mps(path)
            crosscheck_reader(path, p)
            p = relax(p)
            if p.n_variables > 50000 or p.n_constraints > 50000 or p.A.nnz > 500000:
                raise ValueError("parsed structural cap")
            if p.data_hash in hashes:
                raise ValueError("duplicate canonical data")
            hashes.add(p.data_hash); selected_groups.add(r["group"])
            report["selected"].append({**r, "url":url, "raw_sha256":sha(raw), "mps_sha256":sha(mps),
                "file":path.name, "data_hash":p.data_hash, "n":p.n_variables,
                "m":p.n_constraints,"nnz":p.A.nnz,"reader_agreement":True})
            print(json.dumps({"selected":name,"source":source,"n":p.n_variables}),flush=True)
        except Exception as exc:
            report["exclusions"].append({**r,"reason":"pre_solve_import_failure","error":f"{type(exc).__name__}: {exc}"})
            print(json.dumps({"excluded":name,"error":str(exc)[:180]}),flush=True)
        (args.output/"cohort.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (args.output/"cohort.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    if len(report["selected"]) != 24:
        raise SystemExit("could not prepare the frozen 24-instance cohort; no solves performed")


if __name__ == "__main__":
    main()
