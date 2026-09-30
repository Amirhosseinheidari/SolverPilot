"""One real public-LP route execution. Parent owns the process-tree deadline."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse
from solverpilot import LinearProblem, read_mps, solve, solve_production, SolveBudget
from solverpilot.backends import HighspyNativeBackend, PDLPBackend
from solverpilot.benchmark.environment import capture_environment, benchmark_environment_fingerprint
from solverpilot.experimental.learned_lp import LPSelector, decide_lp_backend
from solverpilot.experimental.lp_environment import bind_lp_environment


PACKAGES = ("numpy", "scipy", "highspy", "ortools")
CANDIDATES = ("highs-simplex", "highs-ipm", "ortools-pdlp")


def environment_id():
    env = capture_environment(packages=PACKAGES)
    return benchmark_environment_fingerprint(env, thread_env_limit=1, solver_threads=1,
                                              worker_python_mode="normal")


def candidate(name, cutoff):
    if name == "ortools-pdlp":
        return PDLPBackend(time_limit_s=cutoff, threads=1)
    if name not in {"highs-simplex", "highs-ipm"}:
        raise ValueError("unknown candidate")
    return HighspyNativeBackend(time_limit_s=cutoff, threads=1,
                               solver="simplex" if name == "highs-simplex" else "ipm")


def stress_problem(name):
    if name == "infeasible":
        return LinearProblem.from_data(A=[[1.], [1.]], c=[1.], variable_lower=[0.],
            variable_upper=[2.], constraint_lower=[1., -np.inf], constraint_upper=[np.inf, 0.])
    if name == "unbounded":
        return LinearProblem.from_data(A=[[1.]], c=[-1.], variable_lower=[0.],
            variable_upper=[np.inf], constraint_lower=[0.], constraint_upper=[np.inf])
    if name == "ill_scaled":
        return LinearProblem.from_data(A=[[1e-8, 1.], [1., 1e8]], c=[1., 1.],
            variable_lower=[0., 0.], variable_upper=[2., 2.],
            constraint_lower=[1., 1e8], constraint_upper=[np.inf, np.inf])
    if name == "deadline":
        rng = np.random.default_rng(20260930)
        n = 12000
        a = sparse.random(n//2, n, density=8/n, random_state=rng, format="csr")
        return LinearProblem.from_data(A=a, c=-rng.uniform(.1, 1, n),
            variable_lower=np.zeros(n), variable_upper=np.ones(n),
            constraint_lower=np.full(n//2, -np.inf), constraint_upper=np.asarray(a @ np.full(n, .4)))
    raise ValueError("unknown stress case")


def run_strategy(p, strategy, model_path, cutoff):
    """No retraining or changed policy: execute exactly the frozen recommendation.

    Feature/environment mismatch uses the original model's training baseline.
    Failed solver results are retained, never converted to a success or retried
    with a new full budget. This evaluates the existing policy, not a posthoc fix.
    """
    start = perf_counter()
    metadata = {}
    if strategy == "learned":
        model = LPSelector.load(model_path)
        binding = bind_lp_environment(model, Path(model_path).with_name("protocol.json"),
                                      capture_environment(packages=PACKAGES))
        metadata["environment_binding"] = binding
        available = tuple(c for c in CANDIDATES if candidate(c, cutoff).is_available())
        decision = decide_lp_backend(p, model, environment_id=binding["decision_environment_id"], available=available)
        metadata["decision"] = asdict(decision)
        metadata["model_sha256"] = model.payload()["sha256"]
        if decision.candidate is None:
            raise RuntimeError("no available trained fallback")
        selected = decision.candidate
    else:
        selected = strategy
    setup_s = perf_counter() - start
    remaining = cutoff - setup_s
    if remaining <= 0:
        return {**metadata, "status":"setup_timeout", "verified":False,
                "api_wall_s":perf_counter()-start,"setup_s":setup_s}
    if selected == "default":
        result = solve(p, budget=SolveBudget(wall_time_s=remaining))
    elif selected == "production":
        result, _ = solve_production(p, budget=SolveBudget(wall_time_s=remaining))
    else:
        result = solve(p, backend=candidate(selected, remaining))
    elapsed = perf_counter() - start
    return {**metadata, "status":result.status.value,"backend_status":result.backend_status,
            "backend":result.trace.backend,"selected":selected,"objective":result.objective,
            "primal_valid":bool(result.validation and result.validation.valid),
            "verified":result.optimality_evidence.independently_verified_optimal,
            "api_wall_s":elapsed,"setup_s":setup_s,"trace_timings":asdict(result.trace.timings)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mps", type=Path)
    ap.add_argument("--stress", choices=("infeasible","unbounded","ill_scaled","deadline"))
    ap.add_argument("--strategy", required=True, choices=("learned","default","production",*CANDIDATES))
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--cutoff", type=float, default=2)
    args = ap.parse_args()
    if args.stress:
        p = stress_problem(args.stress)
    else:
        p = read_mps(args.mps)
        p = LinearProblem.from_data(A=p.A,c=p.c,variable_lower=p.variable_lower,
            variable_upper=p.variable_upper,constraint_lower=p.constraint_lower,
            constraint_upper=p.constraint_upper,objective_sense=p.objective_sense,
            objective_offset=p.objective_offset)
    result = run_strategy(p,args.strategy,args.model,args.cutoff)
    result["data_hash"] = p.data_hash
    # Infeasible/unbounded backend status is never counted as an optimal certificate.
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__ == "__main__":
    main()
