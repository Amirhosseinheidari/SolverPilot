from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import scipy
import scipy.optimize._linprog as lp
import scipy.optimize._linprog_highs as lph


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def main() -> None:
    src_public = inspect.getsource(lp.linprog)
    src_highs = inspect.getsource(lph._linprog_highs)
    mapping_present = "'highs': None" in src_public or '"highs": None' in src_public
    none_means_simplex = 'If ``None``, "simplex" will be used.' in src_highs
    payload = {
        'schema': 'optimind.m28.scipy_highs_semantics.v1',
        'scipy_version': scipy.__version__,
        'linprog_source_sha256': sha_text(src_public),
        '_linprog_highs_source_sha256': sha_text(src_highs),
        'public_highs_maps_to_solver_none': mapping_present,
        'private_highs_none_documented_as_simplex': none_means_simplex,
        'm28_interpretation': (
            'SciPy method=highs is not treated as an independent DS-vs-IPM auto-selector in this environment.'
            if mapping_present and none_means_simplex else
            'SciPy implementation semantics differ from the M28 expected probe; manual review required.'
        ),
        'authority_for_performance_ranking': False,
    }
    out = Path('benchmarks/results/m28/m28-scipy-highs-semantics.json')
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
