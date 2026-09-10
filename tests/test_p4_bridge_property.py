from __future__ import annotations

import random
import numpy as np

from solverpilot.model import Model


def _exec_feasible(c,x,tol=1e-10):
    p=c.execution_ir; xx=np.asarray(x,float); ax=np.asarray(p.A@xx,float)
    return bool(np.all(xx>=p.variable_lower-tol) and np.all(xx<=p.variable_upper+tol) and np.all(ax>=p.constraint_lower-tol) and np.all(ax<=p.constraint_upper+tol))


def test_random_indicator_affine_big_m_matches_exhaustive_finite_domain_600_cases():
    rng=random.Random(20260906)
    checked=0
    for _ in range(60):
        lo=rng.randint(-4,0); hi=rng.randint(1,5); a=rng.choice([-3,-2,-1,1,2,3]); b=rng.randint(-3,3); rhs=rng.randint(-4,5); active=rng.randint(0,1); sense=rng.choice(["le","ge","eq"])
        m=Model(); z=m.binary(); x=m.integer(lower=lo,upper=hi); expr=a*x+b
        rel=expr<=rhs if sense=="le" else expr>=rhs if sense=="ge" else expr==rhs
        m.indicator(z,rel,active_value=active); m.minimize(x)
        c=m.compile(use_cache=False)
        for zv in (0,1):
            for xv in range(lo,hi+1):
                body=a*xv+b
                sem=(zv!=active) or (body<=rhs if sense=="le" else body>=rhs if sense=="ge" else body==rhs)
                assert _exec_feasible(c,[zv,xv])==sem
                checked+=1
    assert checked>=600
