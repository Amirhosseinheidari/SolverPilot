"""Bounded, paired LP workflow comparison using the same installed HiGHS kernel.

Run from an environment containing the checkout/wheel to evaluate::

    python benchmarks/compare_lp_workflows.py --preset smoke --repeats 2 --output NEW.json

Optional CVXPY/Pyomo are never installed here. Missing requested workflows are
recorded as unavailable and make the requested comparison incomplete (exit 2).
Cold means a fresh model after imports/probing, NOT a fresh process. Timings
include each wrapper's own work; only the additional original-data checker is
identical. SolverPilot.execute already validates candidates, so its pre-check
time is not an unchecked baseline. No evidence/report-export parity is claimed.

All families are synthetic and deliberately have a planted primal/dual optimum.
They test scaling and repeated updates, not industrial representativeness or
worst-case complexity. The same highspy native library is shared; agreement is
not an independent numerical-solver cross-check. Generated dyadic coefficients
give analytic references, evaluated in floating point by the common checker.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse

WORKFLOWS = ("solverpilot", "direct-highspy", "cvxpy-highs", "pyomo-appsi-highs")
PRESETS = {"smoke": (8,), "quick": (16, 64), "standard": (32, 128, 512)}
CHECK_TOLERANCES = {"atol": 1e-7, "rtol": 1e-9, "objective_atol": 1e-7,
                    "objective_rtol": 1e-7}


@dataclass(frozen=True)
class Config:
    preset: str = "quick"
    seed: int = 94173
    steps: int = 4
    repeats: int = 4
    warmups: int = 1
    max_wall_s: float = 120.0
    solve_limit_s: float = 5.0
    workflows: tuple[str, ...] = WORKFLOWS

    def __post_init__(self):
        if self.preset not in PRESETS:
            raise ValueError("unknown preset")
        for key in ("seed", "steps", "repeats", "warmups"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{key} must be an integer")
        if not 0 <= self.seed < 2**32 or not 2 <= self.steps <= 12:
            raise ValueError("seed or steps outside bounded range")
        if not 1 <= self.repeats <= 30 or not 0 <= self.warmups <= 5:
            raise ValueError("repeats or warmups outside bounded range")
        if not 0 < self.max_wall_s <= 3600 or not 0 < self.solve_limit_s <= 60:
            raise ValueError("time budgets must be positive and bounded")
        object.__setattr__(self, "workflows", tuple(self.workflows))
        if (not self.workflows or len(set(self.workflows)) != len(self.workflows)
                or not set(self.workflows) <= set(WORKFLOWS)):
            raise ValueError("workflows must be a nonempty unique supported selection")


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else str(float(value))
    return value


def digest(value):
    return sha256(json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def freeze_array(value):
    out = np.array(value, dtype=float, copy=True)
    out.flags.writeable = False
    return out


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    seed: int
    scale_power: int
    step: int
    update: str
    A: sparse.csr_matrix
    c: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    row_lower: np.ndarray
    row_upper: np.ndarray
    optimum_x: np.ndarray
    reference_objective: float
    dual_witness: np.ndarray

    def payload(self):
        return json_safe({"name": self.name, "family": self.family, "seed": self.seed,
            "scale_power": self.scale_power, "step": self.step, "update": self.update,
            "shape": self.A.shape, "nnz": self.A.nnz,
            "A_csr": {"data": self.A.data, "indices": self.A.indices,
                      "indptr": self.A.indptr},
            "c": self.c, "lower": self.lower, "upper": self.upper,
            "row_lower": self.row_lower, "row_upper": self.row_upper,
            "optimum_x": self.optimum_x, "reference_objective": self.reference_objective,
            "dual_witness": self.dual_witness})


def make_sequence(family, size, scale_power, seed, steps):
    """Production: c=-A'y; network: c=A'y+r, r>=0, r'x*=0."""
    rng = np.random.default_rng(seed)
    if family == "production":
        n, m = size, max(3, size // 2)
        columns = np.repeat(np.arange(n), 3)
        rows = rng.integers(0, m, len(columns))
        raw = sparse.coo_matrix((rng.integers(1, 5, len(columns)), (rows, columns)),
                                shape=(m, n)).tocsr()
        # Every resource has a nonzero, including in the tiny smoke case.
        raw += sparse.coo_matrix((np.ones(m), (np.arange(m), np.arange(m) % n)),
                                 shape=(m, n)).tocsr()
        xbase = rng.integers(1, 6, n).astype(float)
        dual_base = rng.integers(1, 5, m).astype(float)
        reduced = np.zeros(n)
    elif family == "network":
        m, n = size, 3 * size - 1
        tails = np.r_[np.arange(m - 1), rng.integers(0, m, 2 * m)]
        heads = np.r_[np.arange(1, m), (tails[m - 1:] + rng.integers(1, m, 2 * m)) % m]
        raw = sparse.coo_matrix((np.r_[-np.ones(n), np.ones(n)],
                                (np.r_[tails, heads], np.tile(np.arange(n), 2))),
                               shape=(m, n)).tocsr()
        xbase = np.r_[rng.integers(1, 6, m - 1), np.zeros(2 * m)].astype(float)
        dual_base = rng.integers(-4, 5, m).astype(float)
        reduced = np.r_[np.zeros(m - 1), rng.integers(1, 5, 2 * m)].astype(float)
    else:
        raise ValueError("unknown family")
    scales = np.exp2(np.linspace(-scale_power // 2, scale_power // 2, m).astype(int))
    A = (sparse.diags(scales) @ raw).tocsr()
    A.sum_duplicates()
    A.sort_indices()
    for a in (A.data, A.indices, A.indptr):
        a.flags.writeable = False
    lower, upper = freeze_array(np.zeros(n)), freeze_array(np.maximum(1, xbase) * 3)
    cases = []
    rhs_epoch = objective_epoch = 0
    for step in range(steps):
        update = "baseline" if step == 0 else ("rhs", "objective", "rhs_and_objective")[(step - 1) % 3]
        rhs_epoch += int(update in ("rhs", "rhs_and_objective"))
        objective_epoch += int(update in ("objective", "rhs_and_objective"))
        xstar = xbase * (1 + rhs_epoch / 32)
        # A constant shift of all network potentials would cancel in A'y;
        # nonuniform dyadic shifts make the objective-only update substantive.
        y = (dual_base + (np.arange(m) % 7 + 1) * objective_epoch / 16) / scales
        b = np.asarray(A @ xstar)
        c = -np.asarray(A.T @ y) if family == "production" else np.asarray(A.T @ y) + reduced
        reference = float((-y if family == "production" else y) @ b)
        cases.append(Case(f"{family}-{size}-scale{scale_power}-s{seed}", family, seed,
            scale_power, step, update, A, freeze_array(c), lower, upper,
            freeze_array(np.full(m, -np.inf) if family == "production" else b),
            freeze_array(b), freeze_array(xstar), reference, freeze_array(y)))
    return tuple(cases)


def build_cases(config):
    return tuple(make_sequence(family, size, power, config.seed + i * 100 + j * 10 + k,
                               config.steps)
                 for i, family in enumerate(("production", "network"))
                 for j, size in enumerate(PRESETS[config.preset])
                 for k, power in enumerate((0, 16)))


def independent_check(case, x, objective):
    """Original arrays only: no solver/model/checker implementation is reused."""
    result = {"passed": False, "primal_feasible": False, "objective_consistent": False,
              "reference_agrees": False, "max_primal_violation": None,
              "objective_recomputed": None, "reference_error": None}
    if x is None or objective is None:
        return result
    x = np.asarray(x, dtype=float)
    if x.shape != case.c.shape or not np.isfinite(x).all() or not np.isfinite(objective):
        return result
    with np.errstate(over="ignore", invalid="ignore"):
        activity, actual = case.A @ x, float(case.c @ x)
    if not np.isfinite(activity).all() or not np.isfinite(actual):
        return result
    feasible, max_violation = True, 0.0
    tol = CHECK_TOLERANCES
    for values, bounds, direction in ((x, case.lower, -1), (x, case.upper, 1),
                                     (activity, case.row_lower, -1),
                                     (activity, case.row_upper, 1)):
        finite = np.isfinite(bounds)
        violation = direction * (values[finite] - bounds[finite])
        max_violation = max(max_violation, float(np.max(violation, initial=0)))
        feasible = feasible and bool(np.all(violation <= tol["atol"] + tol["rtol"] * np.abs(bounds[finite])))
    close = lambda a, b: abs(a - b) <= tol["objective_atol"] + tol["objective_rtol"] * max(abs(a), abs(b))
    consistent, reference = close(actual, objective), close(actual, case.reference_objective)
    result.update(passed=bool(feasible and consistent and reference), primal_feasible=feasible,
        objective_consistent=bool(consistent), reference_agrees=bool(reference),
        max_primal_violation=max_violation, objective_recomputed=actual,
        reference_error=abs(actual - case.reference_objective))
    return result


def counterbalanced_order(names, repeat):
    """Cyclic positions, then reversed cycles; partial blocks are disclosed."""
    names = tuple(names)
    if not names:
        return ()
    cycle, shift = divmod(repeat, len(names))
    base = names if cycle % 2 == 0 else names[::-1]
    return base[shift:] + base[:shift]


class DirectHighs:
    reuse = "native_model_retained"
    builtin_validation = False

    def __init__(self, case, settings):
        import highspy as hp
        self.hp, self.h = hp, hp.Highs()
        for key, value in settings.items():
            self.ok(self.h.setOptionValue(key, value))
        A = case.A.tocsc()
        lp = hp.HighsLp()
        lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
        lp.col_cost_, lp.col_lower_, lp.col_upper_ = case.c, case.lower, case.upper
        lp.row_lower_, lp.row_upper_ = case.row_lower, case.row_upper
        lp.a_matrix_.format_ = hp.MatrixFormat.kColwise
        lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
        self.ok(self.h.passModel(lp))

    def ok(self, status):
        if status != self.hp.HighsStatus.kOk:
            raise RuntimeError(f"HiGHS operation returned {status}")

    def update(self, case):
        self.ok(self.h.changeColsCost(len(case.c), np.arange(len(case.c), dtype=np.int32), case.c))
        self.ok(self.h.changeRowsBounds(case.A.shape[0], np.arange(case.A.shape[0], dtype=np.int32),
                                       case.row_lower, case.row_upper))

    def solve(self, warm):
        # kWarning may report a legitimate time/iteration limit; retain its
        # actual model status and any incumbent instead of hiding the outcome.
        run_status = self.h.run()
        if run_status == self.hp.HighsStatus.kError:
            raise RuntimeError(f"HiGHS run returned {run_status}")
        status = self.h.getModelStatus()
        solution, info = self.h.getSolution(), self.h.getInfo()
        present = bool(solution.value_valid)
        return {"status": self.h.modelStatusToString(status),
                "optimal": status == self.hp.HighsModelStatus.kOptimal,
                "x": np.asarray(solution.col_value) if present else None,
                "objective": self.h.getObjectiveValue() if present else None,
                "iterations": int(info.simplex_iteration_count)}


class SolverPilot:
    reuse = "canonical_rebuild_and_cold_native_solve"
    builtin_validation = True

    def __init__(self, case, settings):
        from solverpilot.backends import HighspyNativeBackend
        self.backend = HighspyNativeBackend(threads=settings["threads"], presolve=False,
                                           solver="simplex", time_limit_s=settings["time_limit"])
        self.update(case)

    def update(self, case):
        from solverpilot import LinearProblem
        self.problem = LinearProblem.from_data(A=case.A, c=case.c, variable_lower=case.lower,
            variable_upper=case.upper, constraint_lower=case.row_lower, constraint_upper=case.row_upper)

    def solve(self, warm):
        from solverpilot import execute
        r = execute(self.problem, self.backend)
        return {"status": r.status.value, "optimal": r.backend_status == "optimal",
                "x": r.x, "objective": r.raw_statistics.get("objective_internal"),
                "iterations": r.raw_statistics.get("simplex_iteration_count"),
                "internal_candidate_valid": bool(r.validation and r.validation.valid)}


class CvxpyHighs:
    reuse = "DPP_parameters_and_warm_start_requested_not_attested"
    builtin_validation = False

    def __init__(self, case, settings):
        import cvxpy as cp
        self.cp, self.settings = cp, settings
        # Explicit x>=lower/x<=upper constraints add 2*n native rows with
        # presolve off. Attribute bounds use the same native column bounds as
        # direct highspy, SolverPilot and Pyomo/APPSI.
        self.x = cp.Variable(len(case.c), bounds=[case.lower, case.upper])
        self.c, self.b = cp.Parameter(len(case.c)), cp.Parameter(case.A.shape[0])
        resource = case.A @ self.x <= self.b if case.family == "production" else case.A @ self.x == self.b
        self.model = cp.Problem(cp.Minimize(self.c @ self.x), [resource])
        if not self.model.is_dpp():
            raise RuntimeError("generated CVXPY model is not DPP")
        self.update(case)

    def update(self, case):
        self.c.value, self.b.value = case.c, case.row_upper

    def solve(self, warm):
        self.model.solve(solver="HIGHS", warm_start=warm, enforce_dpp=True,
                         highs_options=dict(self.settings))
        return {"status": self.model.status, "optimal": self.model.status == self.cp.OPTIMAL,
                "x": self.x.value, "objective": self.model.value,
                "iterations": self.model.solver_stats.num_iters}


def cvxpy_formulation_probe(adapter, case):
    """Untimed discovery check of the native conic HIGHS data representation.

    An unrecognized canonicalization path is unavailable for this comparison,
    rather than silently comparing an expanded/reordered formulation.
    """
    data, chain, _ = adapter.model.get_problem_data("HIGHS")
    A = data["A"]
    matches = (A.shape == case.A.shape and (A != case.A).nnz == 0
               and np.array_equal(data["c"], case.c)
               and np.array_equal(data["b"], case.row_upper)
               and np.array_equal(data.get("lower_bounds"), case.lower)
               and np.array_equal(data.get("upper_bounds"), case.upper)
               and data["dims"].zero == (case.A.shape[0] if case.family == "network" else 0)
               and data["dims"].nonneg == (case.A.shape[0] if case.family == "production" else 0))
    if not matches:
        raise ValueError("CVXPY canonical data differ from the shared LP formulation")
    return {"family": case.family, "source_shape": list(case.A.shape),
            "native_shape": list(A.shape), "native_nnz": A.nnz,
            "native_column_bounds": True, "exact_data_match": True,
            "solver_interface": type(chain.reductions[-1]).__module__,
            "scope": "small discovery probe; not a native matrix snapshot of every timed solve"}


class PyomoHighs:
    reuse = "APPSI_persistent_model_and_mutable_parameters"
    builtin_validation = False

    def __init__(self, case, settings):
        import pyomo.environ as pyo
        from pyomo.contrib.appsi.solvers.highs import Highs
        self.pyo, self.model, self.solver = pyo, pyo.ConcreteModel(), Highs()
        m, A = self.model, case.A
        m.x = pyo.Var(range(len(case.c)), bounds=lambda _, j: (case.lower[j], case.upper[j]))
        m.c = pyo.Param(range(len(case.c)), mutable=True, initialize=dict(enumerate(case.c)))
        m.b = pyo.Param(range(A.shape[0]), mutable=True, initialize=dict(enumerate(case.row_upper)))
        m.objective = pyo.Objective(expr=sum(m.c[j] * m.x[j] for j in range(len(case.c))))
        def row_rule(model, i):
            expr = sum(float(A.data[k]) * model.x[int(A.indices[k])]
                       for k in range(A.indptr[i], A.indptr[i + 1]))
            return expr <= model.b[i] if case.family == "production" else expr == model.b[i]
        m.rows = pyo.Constraint(range(A.shape[0]), rule=row_rule)
        self.solver.highs_options.update(settings)
        self.solver.config.load_solution = False
        self.solver.config.time_limit = settings["time_limit"]
        self.solver.config.stream_solver = False

    def update(self, case):
        for j, value in enumerate(case.c):
            self.model.c[j] = float(value)
        for i, value in enumerate(case.row_upper):
            self.model.b[i] = float(value)

    def solve(self, warm):
        from pyomo.contrib.appsi.base import TerminationCondition
        result = self.solver.solve(self.model)
        x = None
        if result.best_feasible_objective is not None:
            primals = self.solver.get_primals()
            x = np.asarray([primals[self.model.x[j]] for j in self.model.x])
        return {"status": str(result.termination_condition),
                "optimal": result.termination_condition == TerminationCondition.optimal,
                "x": x, "objective": result.best_feasible_objective, "iterations": None}


def discover(config):
    factories, availability = {}, {}
    settings = {"solver": "simplex", "presolve": "off", "threads": 1,
                "time_limit": config.solve_limit_s, "output_flag": False}
    native = {"version": None, "extension_sha256": None}
    try:
        import highspy
        h = highspy.Highs()
        native["version"] = h.version()
        extension = Path(import_module("highspy._core").__file__)
        native["extension_sha256"] = sha256(extension.read_bytes()).hexdigest()
        for key in ("primal_feasibility_tolerance", "dual_feasibility_tolerance",
                    "simplex_strategy", "random_seed"):
            status, value = h.getOptionValue(key)
            if status != highspy.HighsStatus.kOk:
                raise RuntimeError(f"cannot read native default {key}")
            settings[key] = value
    except Exception as exc:  # noqa: BLE001 -- discovery records optional/native import failures
        reason = f"shared highspy unavailable: {type(exc).__name__}: {exc}"
        return {}, {name: {"available": False, "reason": reason} for name in config.workflows}, settings, native
    classes = dict(zip(WORKFLOWS, (SolverPilot, DirectHighs, CvxpyHighs, PyomoHighs), strict=True))
    apis = dict(zip(WORKFLOWS, ("solverpilot.execute with HighspyNativeBackend",
        "highspy.Highs public matrix/update API", "cvxpy.Problem.solve(HIGHS), DPP parameters",
        "pyomo.contrib.appsi.solvers.highs.Highs.solve, mutable parameters"), strict=True))
    for name in config.workflows:
        try:
            formulation_probes = []
            if name == "solverpilot":
                import_module("solverpilot")
            elif name == "cvxpy-highs":
                cp = import_module("cvxpy")
                if "HIGHS" not in cp.installed_solvers():
                    raise RuntimeError("CVXPY HIGHS interface unavailable")
                # Outside measured model construction/solve calls. Older or
                # changed interfaces must not silently introduce extra rows.
                for family in ("production", "network"):
                    probe_case = make_sequence(family, 8, 16, config.seed, 2)[0]
                    formulation_probes.append(cvxpy_formulation_probe(
                        CvxpyHighs(probe_case, settings), probe_case))
            elif name == "pyomo-appsi-highs":
                mod = import_module("pyomo.contrib.appsi.solvers.highs")
                if not mod.Highs().available():
                    raise RuntimeError("Pyomo APPSI HiGHS unavailable")
            factories[name] = classes[name]
            availability[name] = {"available": True, "native_version": native["version"],
                "api": apis[name],
                "variable_bounds": "native_column_bounds",
                "canonicalization_probes": formulation_probes,
                "reuse_policy": classes[name].reuse,
                "builtin_candidate_validation": classes[name].builtin_validation}
        except Exception as exc:  # noqa: BLE001 -- a failed optional interface is an outcome
            availability[name] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    return factories, availability, settings, native


def run_sequence(factory, cases, mode, settings, deadline):
    adapter, rows = None, []
    for case in cases:
        row = {"case": case.name, "step": case.step, "update": case.update,
               "case_sha256": digest(case.payload()), "state": "budget_exhausted",
               "workflow_s": None, "construction_or_update_s": None, "solve_api_s": None,
               "independent_check_s": None, "checked_total_s": None, "accepted": False}
        if perf_counter() >= deadline:
            rows.append(row)
            continue
        start = perf_counter()
        try:
            fresh = mode == "cold" or adapter is None
            if fresh:
                adapter = factory(case, settings)
            else:
                adapter.update(case)
            built = perf_counter()
            output = adapter.solve(warm=not fresh)
            solved = perf_counter()
            check = independent_check(case, output["x"], output["objective"])
            checked = perf_counter()
            accepted = bool(output["optimal"] and check["passed"]
                            and output.get("internal_candidate_valid", True))
            row.update(state="completed", status=output["status"], objective=output["objective"],
                candidate=output["x"], iterations=output.get("iterations"), checks=check,
                accepted=accepted, fresh_model=fresh,
                reuse_policy="fresh_model" if fresh else adapter.reuse,
                workflow_s=solved - start, construction_or_update_s=built - start,
                solve_api_s=solved - built, independent_check_s=checked - solved,
                checked_total_s=checked - start)
        except Exception as exc:  # noqa: BLE001 -- preserve all failed benchmark observations
            row.update(state="error", error=f"{type(exc).__name__}: {exc}",
                       failed_elapsed_s=perf_counter() - start)
            adapter = None  # A failed update/solve never contaminates subsequent scenarios.
        rows.append(row)
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        if row["warmup"]:
            continue
        phase = "cold_model" if row["mode"] == "cold" else ("sequence_initial" if row["step"] == 0 else "sequence_update")
        groups.setdefault((row["workflow"], row["case"], phase), []).append(row)
    result = []
    for (workflow, case, phase), group in sorted(groups.items()):
        item = {"workflow": workflow, "case": case, "phase": phase, "attempts": len(group),
                "accepted": sum(r["accepted"] for r in group),
                "failures": sum(not r["accepted"] for r in group),
                "states": dict(Counter(r["state"] for r in group))}
        for metric in ("workflow_s", "independent_check_s", "checked_total_s"):
            values = [r[metric] for r in group if r["accepted"] and r[metric] is not None]
            item[metric] = {"count": len(values), "min": min(values) if values else None,
                "median": float(np.median(values)) if values else None,
                "p95": float(np.quantile(values, .95)) if values else None}
        result.append(item)
    return result


def run_benchmark(config, *, factories=None, availability=None, settings=None, native=None):
    start = perf_counter()
    if factories is None:
        factories, availability, settings, native = discover(config)
    settings, availability, native = settings or {}, availability or {}, native or {}
    cases = build_cases(config)
    config_payload = asdict(config)
    packages = {}
    for package in ("solverpilot", "numpy", "scipy", "highspy", "cvxpy", "pyomo"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    try:
        from solverpilot._identity import process_source_sha256
        source_digest = process_source_sha256()
    except ImportError:
        source_digest = None
    observations, orders = [], []
    names = tuple(name for name in config.workflows if name in factories)
    for sequence in cases:
        for mode in ("cold", "warm_sequence"):
            for iteration in range(config.warmups + config.repeats):
                warmup = iteration < config.warmups
                repeat = iteration if warmup else iteration - config.warmups
                order = counterbalanced_order(names, repeat)
                orders.append({"case": sequence[0].name, "mode": mode, "warmup": warmup,
                               "repeat": repeat, "order": list(order)})
                for position, name in enumerate(order):
                    outcome = run_sequence(factories[name], sequence, mode, settings,
                                           start + config.max_wall_s)
                    for row in outcome:
                        row.update(workflow=name, mode=mode, warmup=warmup, repeat=repeat,
                                   order_position=position)
                    observations.extend(outcome)
    measured = [row for row in observations if not row["warmup"]]
    warmup_rows = [row for row in observations if row["warmup"]]
    expected = len(cases) * 2 * config.repeats * config.steps * len(config.workflows)
    complete = (len(measured) == expected and bool(measured)
                and all(row["accepted"] for row in measured))
    dataset = [[case.payload() for case in sequence] for sequence in cases]
    return json_safe({"schema": "solverpilot.lp-workflow-comparison.v1", "config": config_payload,
        "config_sha256": digest(config_payload), "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "solverpilot_source_sha256": source_digest, "packages": packages,
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "native_highs": native, "native_options": settings,
        "native_tolerances_policy": "SolverPilot native defaults read from same highspy and applied to rivals",
        "check_tolerances": CHECK_TOLERANCES, "availability": availability,
        "dataset_sha256": digest(dataset), "dataset": dataset, "orders": orders,
        "observations": observations, "summary": summarize(observations),
        "warmup_outcomes": {"attempts": len(warmup_rows),
            "accepted": sum(row["accepted"] for row in warmup_rows),
            "failures": sum(not row["accepted"] for row in warmup_rows),
            "states": dict(Counter(row["state"] for row in warmup_rows))},
        "requested_matrix_complete": complete, "expected_measured_rows": expected,
        "actual_measured_rows": len(measured), "elapsed_s": perf_counter() - start,
        "protocol": {
            "objective": "minimize", "scope": "synthetic continuous LP only",
            "cold": "fresh model after imports/probing; process startup excluded",
            "warm": "retained wrapper, objective/RHS updates; native reuse depends on recorded workflow policy",
            "timing": "workflow_s includes construction/update and solve API, including wrapper-specific checks; checked_total_s adds common checker",
            "phase_caveat": "canonicalization/native construction can occur inside solve_api_s; phases are not identical across APIs",
            "formulation": "all adapters use native column bounds; CVXPY discovery checks representative canonical A/c/b/bounds exactly and records its interface; no extra canonicalization is timed",
            "order": "cyclic positions then reversed cycles; each case and mode counterbalanced separately",
            "complete_position_blocks": not names or config.repeats % len(names) == 0,
            "warmups_measured_in_summary": False,
            "completion_policy": "requested_matrix_complete covers measured rows only; warmup outcomes are separately counted and retained",
            "reference": "production: -y'Ax >= -y'b for y>=0; network: c=A'y+r, r>=0 and r'x*=0; planted x* feasible",
            "reference_limit": "analytic witnesses evaluated in floating point; shared HiGHS is not an independent backend cross-check",
            "family_limit": "planted optima retain support/tight rows across updates; active-set changes and difficult degeneracy are not qualified",
            "scaling": "power16 is coefficient row-scale spread, not a measured matrix condition number",
            "scope_exclusions": ["integer models", "infeasible/unbounded challenges", "evidence/export parity", "SolverPilot automatic routing and production contracts", "JuMP/MOI (Julia interface not implemented)", "real-user productivity", "industrial performance claims"],
            "budget": "global soft stop checked before each operation; native time limit excludes construction/checking",
            "statistics": "timing summaries include accepted measured rows only; failed/unavailable counts remain explicit; p95 descriptive without significance claims"}})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", choices=PRESETS, default="quick")
    parser.add_argument("--seed", type=int, default=94173)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-wall-s", type=float, default=120)
    parser.add_argument("--solve-limit-s", type=float, default=5)
    parser.add_argument("--workflows", default=",".join(WORKFLOWS))
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be a new file")
    config = Config(args.preset, args.seed, args.steps, args.repeats, args.warmups,
                    args.max_wall_s, args.solve_limit_s, tuple(args.workflows.split(",")))
    result = run_benchmark(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output), "requested_matrix_complete": result["requested_matrix_complete"],
                      "measured_rows": result["actual_measured_rows"], "availability": result["availability"]}))
    return 0 if result["requested_matrix_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
