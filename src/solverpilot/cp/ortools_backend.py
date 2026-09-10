from __future__ import annotations
from dataclasses import dataclass
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
from collections.abc import Mapping

from solverpilot._version import __version__ as SOLVERPILOT_VERSION
from .ir import *
from .validate import validate_cp_solution
from .reference import CPSolveResult

VERIFIED_ORTOOLS_VERSION = '9.15.6755'
CP_WORKER_PROTOCOL_VERSION = 2
CP_WORKER_REQUEST_SCHEMA = 'solverpilot.cp.ortools_worker_request.v2'
CP_WORKER_RESULT_SCHEMA = 'solverpilot.cp.ortools_worker_result.v2'


def _isolated_worker_command() -> list[str]:
    """Return an isolated worker command pinned to this SolverPilot tree.

    ``-I`` removes the current directory, user site, and ``PYTHON*`` environment
    variables from normal import discovery.  The bootstrap creates minimal package
    shells rooted at the exact package tree that supplied this backend.  This avoids
    eager top-level SolverPilot imports before OR-Tools and prevents CWD/PYTHONPATH
    shadowing of the worker package.
    """
    package_parent = str(Path(__file__).resolve().parents[2])
    bootstrap = (
        "import pathlib,runpy,sys,types; "
        "root=pathlib.Path(sys.argv[1]); "
        "sp=types.ModuleType('solverpilot'); "
        "sp.__path__=[str(root/'solverpilot')]; "
        "sp.__package__='solverpilot'; "
        "sys.modules['solverpilot']=sp; "
        "cp=types.ModuleType('solverpilot.cp'); "
        "cp.__path__=[str(root/'solverpilot'/'cp')]; "
        "cp.__package__='solverpilot.cp'; "
        "sys.modules['solverpilot.cp']=cp; "
        "runpy.run_module('solverpilot.cp.ortools_worker', run_name='__main__')"
    )
    return [sys.executable, '-I', '-c', bootstrap, package_parent]


def _isolated_worker_environment() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith('PYTHON')}
    env['PYTHONNOUSERSITE'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONSAFEPATH'] = '1'
    return env


def _require_json_int(value, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f'P9 isolated CP-SAT worker returned non-integer {where}')
    return int(value)


def _require_json_bool(value, *, where: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeError(f'P9 isolated CP-SAT worker returned non-boolean {where}')
    return value


def _bound_matches_integer(value, objective: int) -> bool:
    # CP-SAT exposes objective/bound as binary64.  Above 2**53 adjacent integers are
    # not distinguishable, so proof preservation deliberately fails closed.
    if abs(int(objective)) > 2**53:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and abs(numeric - float(objective)) < 0.5



def _translate_to_cp_sat(problem: CPProblem, cp_model):
    """Translate P9 CP IR into an OR-Tools CpModel.

    Kept separate from binding discovery/solve so the translation contract can
    be unit-tested without pretending the native CP-SAT runtime is present.
    """
    m = cp_model.CpModel()
    vmap = {}
    for v in problem.variables:
        # Use a true BoolVar for the canonical {0,1} Boolean domain because
        # ExactlyOne/Circuit consume literals. Fixed/subset Boolean domains are
        # still represented exactly through an integer domain.
        if v.kind == 'bool' and tuple(v.domain) == (0, 1):
            vv = m.new_bool_var(v.name)
        else:
            dom = cp_model.Domain.from_values(list(v.domain))
            vv = m.new_int_var_from_domain(dom, v.name)
        vmap[v.var_id] = vv

    for v in problem.variables:
        if v.derived_from is not None:
            start_id, size = v.derived_from
            m.add(vmap[v.var_id] == vmap[start_id] + size)

    imap = {}
    for iv in problem.intervals:
        imap[iv.interval_id] = m.new_interval_var(
            vmap[iv.start_var], iv.size, vmap[iv.end_var], iv.name
        )

    def expr(e):
        return e.constant + sum(c * vmap[i] for i, c in e.terms)

    for c in problem.constraints:
        if isinstance(c, CPLinearConstraintIR):
            ex = expr(c.expr)
            if c.lower is not None and c.upper is not None and c.lower == c.upper:
                m.add(ex == c.lower)
            else:
                if c.lower is not None:
                    m.add(ex >= c.lower)
                if c.upper is not None:
                    m.add(ex <= c.upper)
        elif isinstance(c, CPAllDifferentIR):
            m.add_all_different([vmap[i] for i in c.var_ids])
        elif isinstance(c, CPExactlyOneIR):
            m.add_exactly_one([vmap[i] for i in c.literal_ids])
        elif isinstance(c, CPTableIR):
            m.add_allowed_assignments([vmap[i] for i in c.var_ids], c.allowed_tuples)
        elif isinstance(c, CPElementIR):
            m.add_element(vmap[c.index_var], list(c.values), vmap[c.target_var])
        elif isinstance(c, CPCircuitIR):
            m.add_circuit([(t, h, vmap[l]) for t, h, l in c.arcs])
        elif isinstance(c, CPNoOverlapIR):
            m.add_no_overlap([imap[i] for i in c.interval_ids])
        elif isinstance(c, CPCumulativeIR):
            m.add_cumulative([imap[i] for i in c.interval_ids], list(c.demands), c.capacity)
        else:  # pragma: no cover - defensive against future IR expansion.
            raise TypeError(f'unsupported P9 CP constraint IR {type(c).__name__}')

    if problem.objective is not None:
        if problem.objective_sense is CPObjectiveSense.MINIMIZE:
            m.minimize(expr(problem.objective))
        else:
            m.maximize(expr(problem.objective))
    return m, vmap, imap


@dataclass(slots=True)
class ORToolsCPSATBackend:
    max_time_s: float | None = None
    num_workers: int = 1
    controller_timeout_s: float = 3600.0
    name: str = 'ortools-cp-sat'

    def __post_init__(self):
        if self.max_time_s is not None and self.max_time_s <= 0:
            raise ValueError('max_time_s must be positive')
        if self.controller_timeout_s <= 0:
            raise ValueError('controller_timeout_s must be positive')
        if isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int) or self.num_workers < 1:
            raise ValueError('num_workers must be a positive integer')

    @property
    def binding_version(self):
        try:
            return importlib.metadata.version("ortools")
        except importlib.metadata.PackageNotFoundError:
            return None

    def capability_manifest_v2(self):
        from solverpilot.capabilities.v2 import (
            BackendCapabilityManifestV2, CapabilityClaim, CapabilityEvidence, CapabilityKey,
            CapabilityMode, CapabilityStatus, VerificationLevel,
        )
        evidence = CapabilityEvidence(
            evidence_id='p9-ortools-cp-sat-9.15.6755',
            kind='native-integration',
            binding_versions=(VERIFIED_ORTOOLS_VERSION,),
            adapter_versions=(SOLVERPILOT_VERSION,),
            verified_on='2026-09-09',
            notes=('isolated worker; parent primal/objective revalidation',),
        )
        supported = (
            CapabilityKey.PROBLEM_CP, CapabilityKey.CONSTRAINT_LINEAR,
            CapabilityKey.CONSTRAINT_ALL_DIFFERENT, CapabilityKey.CONSTRAINT_EXACTLY_ONE,
            CapabilityKey.CONSTRAINT_TABLE, CapabilityKey.CONSTRAINT_ELEMENT,
            CapabilityKey.CONSTRAINT_CIRCUIT, CapabilityKey.CONSTRAINT_NO_OVERLAP,
            CapabilityKey.CONSTRAINT_CUMULATIVE, CapabilityKey.RESULT_PRIMAL,
            CapabilityKey.RESULT_OBJECTIVE, CapabilityKey.LIFECYCLE_THREAD_CONTROL,
        )
        claims = {
            key: CapabilityClaim(
                status=CapabilityStatus.SUPPORTED, mode=CapabilityMode.NATIVE,
                verification=VerificationLevel.VERIFIED, evidence=(evidence,),
            ) for key in supported
        }
        return BackendCapabilityManifestV2(
            backend=self.name, backend_version=self.binding_version, binding='ortools',
            binding_version=self.binding_version, adapter_version=SOLVERPILOT_VERSION,
            claims=claims, metadata={
                'worker_protocol_version': CP_WORKER_PROTOCOL_VERSION,
                'verified_ortools_version': VERIFIED_ORTOOLS_VERSION,
                'isolated_subprocess': True,
            },
        )

    def is_available(self):
        return (
            importlib.util.find_spec('ortools') is not None
            and self.binding_version == VERIFIED_ORTOOLS_VERSION
        )

    def _require(self):
        if importlib.util.find_spec('ortools') is None:
            raise RuntimeError(
                f'P9 fail-closed: OR-Tools CP-SAT is not installed; '
                f'verified target is {VERIFIED_ORTOOLS_VERSION}'
            )
        if self.binding_version != VERIFIED_ORTOOLS_VERSION:
            raise RuntimeError(
                f'P9 fail-closed: OR-Tools {self.binding_version} is not the '
                f'verified version {VERIFIED_ORTOOLS_VERSION}'
            )

    def _solve_in_process(self, problem: CPProblem):
        self._require()
        import ortools
        from ortools.sat.python import cp_model

        m, vmap, _ = _translate_to_cp_sat(problem, cp_model)
        validation_message = m.validate()
        if validation_message:
            return CPSolveResult(
                'model_invalid',
                None,
                None,
                validate_cp_solution(problem, {}),
                False,
                self.name,
                {
                    'ortools_version': ortools.__version__,
                    'status': 'MODEL_INVALID_PRE_SOLVE',
                    'model_validation': validation_message,
                },
            )

        solver = cp_model.CpSolver()
        solver.parameters.num_workers = int(self.num_workers)
        if self.max_time_s is not None:
            solver.parameters.max_time_in_seconds = float(self.max_time_s)

        st = solver.solve(m)
        status_name = solver.status_name(st)
        if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return CPSolveResult(
                status_name.lower(),
                None,
                None,
                validate_cp_solution(problem, {}),
                st == cp_model.INFEASIBLE,
                self.name,
                {
                    'ortools_version': ortools.__version__,
                    'status': status_name,
                    'model_validation': '',
                },
            )

        assignment = {i: int(solver.value(v)) for i, v in vmap.items()}
        pre_validation = validate_cp_solution(problem, assignment)
        # The canonical objective is integral. Recompute it from the returned
        # assignment instead of trusting/rounding CP-SAT's floating API value.
        objective = pre_validation.objective_recomputed
        report = validate_cp_solution(problem, assignment, objective)
        raw = {
            'ortools_version': ortools.__version__,
            'status': status_name,
            'model_validation': '',
        }
        if problem.objective is not None:
            raw.update(
                solver_objective_value=float(solver.objective_value),
                best_bound=float(solver.best_objective_bound),
            )
        return CPSolveResult(
            'optimal' if st == cp_model.OPTIMAL else 'feasible',
            assignment,
            objective,
            report,
            st == cp_model.OPTIMAL,
            self.name,
            raw,
        )
    def solve(self, problem: CPProblem):
        """Solve CP-SAT in a version- and request-bound isolated process.

        The worker starts under Python isolated mode, from a temporary CWD, with
        ``PYTHON*`` injection removed.  The request/result protocol binds the response
        to this exact SolverPilot version, verified OR-Tools version, request nonce,
        and both canonical problem hashes.  The parent independently revalidates every
        returned primal assignment/objective before preserving backend optimality.
        """
        self._require()
        request_id = secrets.token_hex(16)
        request = {
            'schema': CP_WORKER_REQUEST_SCHEMA,
            'protocol_version': CP_WORKER_PROTOCOL_VERSION,
            'request_id': request_id,
            'solverpilot_version': SOLVERPILOT_VERSION,
            'ortools_version': self.binding_version,
            'problem_structural_hash': problem.structural_hash,
            'problem_data_hash': problem.data_hash,
            'problem': problem.canonical_dict(),
            'max_time_s': self.max_time_s,
            'num_workers': self.num_workers,
        }
        solver_timeout = None if self.max_time_s is None else max(30.0, float(self.max_time_s) + 15.0)
        timeout = float(self.controller_timeout_s) if solver_timeout is None else min(float(self.controller_timeout_s), solver_timeout)
        try:
            with tempfile.TemporaryDirectory(prefix='solverpilot-cp-worker-') as worker_cwd:
                proc = subprocess.run(
                    _isolated_worker_command(),
                    input=json.dumps(request, sort_keys=True, separators=(',', ':')),
                    text=True, capture_output=True, timeout=timeout, cwd=worker_cwd,
                    env=_isolated_worker_environment(),
                )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f'P9 isolated CP-SAT worker exceeded controller timeout {timeout:.3f}s') from exc
        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip() or f'exit {proc.returncode}'
            raise RuntimeError(f'P9 isolated CP-SAT worker failed: {detail}')
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError('P9 isolated CP-SAT worker returned invalid JSON') from exc
        if not isinstance(payload, dict):
            raise RuntimeError('P9 isolated CP-SAT worker returned a non-object payload')
        if payload.get('schema') != CP_WORKER_RESULT_SCHEMA:
            raise RuntimeError('P9 isolated CP-SAT worker returned an unknown schema')
        if payload.get('protocol_version') != CP_WORKER_PROTOCOL_VERSION:
            raise RuntimeError('P9 isolated CP-SAT worker protocol version mismatch')
        expected_bindings = {
            'request_id': request_id,
            'solverpilot_version': SOLVERPILOT_VERSION,
            'ortools_version': self.binding_version,
            'problem_structural_hash': problem.structural_hash,
            'problem_data_hash': problem.data_hash,
        }
        for field, expected in expected_bindings.items():
            if payload.get(field) != expected:
                raise RuntimeError(f'P9 isolated CP-SAT worker binding mismatch: {field}')

        status = payload.get('status')
        if not isinstance(status, str) or not status:
            raise RuntimeError('P9 isolated CP-SAT worker returned invalid status')
        assignment_payload = payload.get('assignment')
        if assignment_payload is None:
            assignment = None
        else:
            if not isinstance(assignment_payload, Mapping):
                raise RuntimeError('P9 isolated CP-SAT worker returned invalid assignment')
            assignment = {}
            for key, value in assignment_payload.items():
                try:
                    var_id = int(key)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError('P9 isolated CP-SAT worker returned invalid assignment key') from exc
                if var_id in assignment:
                    raise RuntimeError('P9 isolated CP-SAT worker returned duplicate normalized assignment key')
                if var_id not in problem.variable_map:
                    raise RuntimeError(f'P9 isolated CP-SAT worker returned unknown variable id {var_id}')
                assignment[var_id] = _require_json_int(value, where='assignment value')

        objective_payload = payload.get('objective')
        objective = None if objective_payload is None else _require_json_int(objective_payload, where='objective')
        if problem.objective is None and objective is not None:
            raise RuntimeError('P9 isolated CP-SAT worker returned an objective for an objective-free problem')
        if problem.objective is not None and assignment is not None and objective is None:
            raise RuntimeError('P9 isolated CP-SAT worker omitted the objective for a primal solution')
        if status in {'optimal', 'feasible'} and assignment is None:
            raise RuntimeError(f'P9 isolated CP-SAT worker returned {status} without an assignment')
        if status == 'infeasible' and assignment is not None:
            raise RuntimeError('P9 isolated CP-SAT worker returned infeasible with an assignment')

        raw_payload = payload.get('raw_statistics', {})
        if not isinstance(raw_payload, Mapping):
            raise RuntimeError('P9 isolated CP-SAT worker returned invalid raw_statistics')
        raw = dict(raw_payload)
        if raw.get('ortools_version') != self.binding_version:
            raise RuntimeError('P9 isolated CP-SAT worker binding mismatch: raw_statistics.ortools_version')
        worker_proof = _require_json_bool(payload.get('optimality_proven'), where='optimality_proven')

        report = validate_cp_solution(problem, assignment or {}, objective)
        raw['isolated_worker'] = True
        raw['worker_protocol_version'] = CP_WORKER_PROTOCOL_VERSION
        raw['worker_request_bound'] = True
        raw['worker_binding_verified'] = True
        raw['parent_revalidated'] = assignment is not None
        raw['worker_optimality_proven'] = worker_proof
        raw['parent_validation_valid'] = bool(report.valid)

        proof_consistent = False
        raw_status = raw.get('status')
        if (worker_proof and status == 'optimal' and assignment is not None and report.valid
                and raw_status == 'OPTIMAL' and raw.get('model_validation', '') == ''):
            if problem.objective is None:
                proof_consistent = True
            else:
                proof_consistent = (objective is not None
                    and _bound_matches_integer(raw.get('solver_objective_value'), objective)
                    and _bound_matches_integer(raw.get('best_bound'), objective))
        elif (worker_proof and status == 'infeasible' and assignment is None and objective is None
                and raw_status == 'INFEASIBLE' and raw.get('model_validation', '') == ''):
            proof_consistent = True

        raw['worker_proof_consistent'] = proof_consistent
        proof_preserved = bool(worker_proof and proof_consistent)
        return CPSolveResult(status, assignment, objective, report, proof_preserved, self.name, raw)
