# M12 Internet / Literature Audit

Checked on 2026-08-31. Primary sources are preferred.

1. **Benchopt 1.9.1 distributed execution** — parallel joblib execution can reduce C-level thread counts to avoid oversubscription; sequential and parallel timings should not be compared blindly. Benchopt also supports submitit/Dask distributed runs and cache reuse on shared filesystems.
   https://benchopt.github.io/stable/user_guide/distributed_run.html

2. **OSQP status semantics** — `*_INACCURATE` statuses satisfy the corresponding conditions at tolerances 10x larger than configured values. They must not be silently promoted to exact proof-level statuses.
   https://osqp.org/docs/interfaces/status_values.html

3. **OSQP infeasibility evidence** — solver result APIs expose primal and dual infeasibility certificates; M12 adds independent certificate checking rather than trusting status text alone.
   https://osqp.org/docs/solver/

4. **PySCIPOpt model semantics** — `addObjoffset()` exists for constant objective offset; changing an already optimized model requires lifecycle handling (`freeTransform()`), and explicit reoptimization APIs exist.
   https://pyscipopt.readthedocs.io/en/latest/api/model.html
   https://pyscipopt.readthedocs.io/en/latest/tutorials/model.html

5. **CasADi release drift** — current upstream release is 3.8.0 (August 2026), while the verification bridges in this runtime were executed on 3.7.2. M12 pins the optional bridge dependency to the verified version instead of extrapolating evidence across versions.
   https://web.casadi.org/get/

6. **MIPLIB 2017** — benchmark set version 2 contains 240 instances. Solution file v36 was released 2026-01-26.
   https://miplib.zib.de/
   https://miplib.zib.de/download.html

7. **QPLIB** — official collection contains 134 continuous and 319 discrete instances; official statistics report 32 continuous-convex. This is not equivalent to OptiMind's supported subset because QPLIB includes quadratic-constraint structures outside the current IR.
   https://qplib.zib.de/
   https://qplib.zib.de/statistics.html

8. **BenLOC (2025)** — identifies data leakage, homogeneous datasets, inconsistent train/test setups and baseline choice as sources of over-optimistic learned MIP-configuration claims. M12 therefore validates group-aware split artifacts before future ML evaluation.
   https://arxiv.org/abs/2506.02752

9. **Feature-computation budget in PIAS (2026)** — feature computation cost can materially consume the opportunity available to per-instance selection, so M12 keeps inspect/plan/selection overhead in deployable-policy accounting.
   https://arxiv.org/abs/2605.04954

10. **Current public Python solver packages** — highspy 1.15.1 (2026-07-02), OSQP 1.1.3 (2026-06-12), PySCIPOpt 6.2.1 (2026-05-16), and NLopt 2.11.0 (2026-07-17) are current targets verified from PyPI metadata. The first three remain unavailable in this runtime.
   https://pypi.org/project/highspy/
   https://pypi.org/project/osqp/1.1.3/
   https://pypi.org/project/PySCIPOpt/6.2.1/
   https://pypi.org/project/nlopt/2.11.0/
