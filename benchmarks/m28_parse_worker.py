from __future__ import annotations
import argparse,gzip,json,pickle,zipfile
from pathlib import Path
from solverpilot import LinearProblem, VariableDomain, parse_mps

def relax(problem: LinearProblem, name: str) -> LinearProblem:
    return LinearProblem.from_data(
        A=problem.A,c=problem.c,
        variable_lower=problem.variable_lower,variable_upper=problem.variable_upper,
        constraint_lower=problem.constraint_lower,constraint_upper=problem.constraint_upper,
        domains=[VariableDomain.CONTINUOUS]*problem.n_variables,
        objective_sense=problem.objective_sense,objective_offset=problem.objective_offset,
        name=f'{name}-m28-lp-relaxation',
        metadata={**problem.metadata,'m28_source_instance':name,'m28_relaxation':True},
    )

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--zip',type=Path,required=True); ap.add_argument('--instance',required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    with zipfile.ZipFile(args.zip) as z:
        raw=gzip.decompress(z.read(args.instance)).decode('utf-8','strict')
    p=relax(parse_mps(raw),Path(args.instance).name.removesuffix('.mps.gz'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes(pickle.dumps(p,protocol=pickle.HIGHEST_PROTOCOL))
    print(json.dumps({'instance':args.instance,'n':p.n_variables,'m':p.n_constraints,'nnz':p.nnz,'output':str(args.output.resolve())},sort_keys=True))
if __name__=='__main__': main()
