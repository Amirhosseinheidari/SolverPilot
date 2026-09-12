"""Direct Clarabel transport with original-space validation.

PSD uses column-major upper-triangle svec, with sqrt(2) off-diagonals.
Nonsymmetric affine PSD expressions also receive explicit symmetry equations.
"""
from dataclasses import dataclass, replace, field
from threading import RLock
import hashlib
from solverpilot._synchronization import serialized
from functools import lru_cache
from importlib.util import find_spec
from time import perf_counter

import numpy as np
from scipy import sparse

from solverpilot.backends.metadata import version
from solverpilot.problem import ConvexityStatus
from .backend import ConicSolveResult
from .ir import ConeKind, ConicProblem
from .validation import validate_conic_solution


@dataclass(slots=True)
class ClarabelBackend:
    max_iter: int = 200
    tolerance: float = 1e-9
    time_limit_s: float | None = None
    threads: int = 1
    verbose: bool = False
    reuse: bool = False
    _lock: object = field(default_factory=RLock, init=False, repr=False, compare=False)
    _solver: object = field(default=None, init=False, repr=False, compare=False)
    _signature: str | None = field(default=None, init=False, repr=False, compare=False)
    _active: bool = field(default=False, init=False, repr=False, compare=False)

    @property
    def name(self):
        return 'clarabel-native'

    @property
    def binding_version(self):
        return version('clarabel') if self.is_available() else None

    def is_available(self):
        return find_spec('clarabel') is not None

    @property
    def capability_manifest_v2(self):
        from solverpilot import __version__
        from solverpilot.capabilities.v2 import (
            BackendCapabilityManifestV2, CapabilityClaim, CapabilityEvidence,
            CapabilityKey, CapabilityMode, CapabilityStatus, VerificationLevel,
        )
        verified = self.is_available() and _runtime_conformance(self.binding_version)
        evidence = CapabilityEvidence(
            evidence_id='clarabel-analytic-runtime-v1', kind='native-runtime-conformance',
            backend_versions=(() if self.binding_version is None else (self.binding_version,)),
            binding_versions=(() if self.binding_version is None else (self.binding_version,)),
            adapter_versions=(__version__,),
            notes=('Runtime analytic SOC/RSOC/PSD/quadratic-conic checks; original-space validation.',),
        )
        claim = CapabilityClaim(
            status=CapabilityStatus.SUPPORTED if verified else CapabilityStatus.UNVERIFIED,
            mode=CapabilityMode.NATIVE,
            verification=VerificationLevel.VERIFIED if verified else VerificationLevel.UNVERIFIED,
            evidence=(evidence,) if verified else (),
        )
        keys = (CapabilityKey.PROBLEM_CONIC, CapabilityKey.PROBLEM_CONIC_QUADRATIC,
                CapabilityKey.CONSTRAINT_LINEAR, CapabilityKey.CONSTRAINT_SOC,
                CapabilityKey.CONSTRAINT_ROTATED_SOC, CapabilityKey.CONSTRAINT_PSD,
                CapabilityKey.CONSTRAINT_EXPONENTIAL, CapabilityKey.CONSTRAINT_POWER,
                CapabilityKey.CONSTRAINT_GENERALIZED_POWER,
                CapabilityKey.RESULT_PRIMAL, CapabilityKey.RESULT_OBJECTIVE)
        return BackendCapabilityManifestV2(
            backend=self.name, backend_version=self.binding_version or 'unavailable',
            binding='clarabel', binding_version=self.binding_version,
            adapter_version=__version__, claims={k: claim for k in keys},
            metadata={'direct_binding': True, 'independent_global_proof': False},
        )

    @serialized
    def solve(self, problem: ConicProblem, *, tolerances=None, progress=None, cancellation=None) -> ConicSolveResult:
        if self._active:
            raise RuntimeError('a progress callback cannot reenter the same Clarabel backend')
        import clarabel
        if problem.convexity_status is not ConvexityStatus.CONFIRMED:
            raise ValueError('Clarabel requires a confirmed convex objective')
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError('max_iter must be a positive integer')
        if isinstance(self.threads, bool) or not isinstance(self.threads, int) or self.threads <= 0:
            raise ValueError('threads must be a positive integer')
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError('tolerance must be finite and positive')
        if self.time_limit_s is not None and (not np.isfinite(self.time_limit_s) or self.time_limit_s <= 0):
            raise ValueError('time_limit_s must be finite and positive')
        start = perf_counter()
        from solverpilot.validate import ValidationTolerances
        from solverpilot.runtime.manifest import backend_configuration, json_value
        tol = tolerances or ValidationTolerances()
        parameters = {'backend_configuration': backend_configuration(self), 'validation_tolerances': json_value(tol)}
        if cancellation is not None and cancellation.cancelled:
            return ConicSolveResult(self.name, 'Cancelled', None, None,
                validate_conic_solution(problem, None), {'run_parameters': parameters}, problem.data_hash)
        matrices, rhs, cones = [], [], []
        linear_segments, cone_segments = [], []
        transport_size = 0

        def append(A, b, cone):
            nonlocal transport_size
            segment = slice(transport_size, transport_size+np.asarray(b).size)
            transport_size = segment.stop
            matrices.append(sparse.csc_matrix(A)); rhs.append(np.asarray(b).reshape(-1)); cones.append(cone)
            return segment

        A = sparse.vstack([problem.A, sparse.eye(problem.n_variables)], format='csr')
        lo = np.r_[problem.constraint_lower, problem.variable_lower]
        hi = np.r_[problem.constraint_upper, problem.variable_upper]
        eq = np.isfinite(lo) & (lo == hi)
        if np.any(eq):
            segment = append(A[eq], lo[eq], clarabel.ZeroConeT(int(eq.sum())))
            linear_segments.append((np.flatnonzero(eq),1,segment))
        for mask, sign, bound in [(np.isfinite(hi) & ~eq, 1, hi), (np.isfinite(lo) & ~eq, -1, lo)]:
            if np.any(mask):
                segment = append(sign*A[mask], sign*bound[mask], clarabel.NonnegativeConeT(int(mask.sum())))
                linear_segments.append((np.flatnonzero(mask),sign,segment))
        for block in problem.cones:
            F, g = block.F, block.g
            if block.kind is ConeKind.EXPONENTIAL:
                segment = append(-F, g, clarabel.ExponentialConeT())
            elif block.kind is ConeKind.POWER:
                segment = append(-F, g, clarabel.PowerConeT(float(block.metadata['alpha'])))
            elif block.kind is ConeKind.GENERALIZED_POWER:
                segment = append(-F, g, clarabel.GenPowerConeT(list(block.metadata["weights"]), block.metadata["tail_dimension"]))
            elif block.kind is ConeKind.SECOND_ORDER:
                segment = append(-F, g, clarabel.SecondOrderConeT(block.dimension))
            elif block.kind is ConeKind.ROTATED_SECOND_ORDER:
                d = block.dimension
                T = sparse.lil_matrix((d, d))
                T[0, 0] = T[0, 1] = 1
                T[1, 0] = 1; T[1, 1] = -1
                for i in range(2, d):
                    T[i, i] = np.sqrt(2.)
                T = T.tocsr()
                segment = append(-(T@F), T@g, clarabel.SecondOrderConeT(d))
            elif block.kind is ConeKind.POSITIVE_SEMIDEFINITE:
                n = block.output_shape[0]
                indices, factors, symmetry_rows, symmetry_rhs = [], [], [], []
                for j in range(n):
                    for i in range(j+1):
                        indices.append(i*n+j); factors.append(1. if i == j else np.sqrt(2.))
                        if i != j:
                            symmetry_rows.append(F.getrow(i*n+j)-F.getrow(j*n+i))
                            symmetry_rhs.append(g[j*n+i]-g[i*n+j])
                if symmetry_rows:
                    append(sparse.vstack(symmetry_rows), symmetry_rhs, clarabel.ZeroConeT(len(symmetry_rows)))
                S = sparse.diags(factors)
                segment = append(-(S@F[indices]), S@g[indices], clarabel.PSDTriangleConeT(n))
            else:
                raise ValueError(f'unsupported cone: {block.kind}')
            cone_segments.append(segment)
        settings = clarabel.DefaultSettings()
        settings.verbose = self.verbose; settings.max_iter = self.max_iter
        settings.max_threads = self.threads
        settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = self.tolerance
        if self.time_limit_s is not None:
            settings.time_limit = self.time_limit_s
        if self.reuse:
            settings.presolve_enable = False
            settings.chordal_decomposition_enable = False
        P = sparse.triu(problem.P, format='csc')
        transport_A = sparse.vstack(matrices, format='csc') if matrices else sparse.csc_matrix((0, problem.n_variables))
        transport_b = np.concatenate(rhs) if rhs else np.empty(0)
        digest = hashlib.sha256(repr((P.shape, transport_A.shape, cones, parameters)).encode())
        for matrix in (P, transport_A):
            digest.update(matrix.indptr.tobytes()); digest.update(matrix.indices.tobytes())
        signature = digest.hexdigest()
        reused = self.reuse and self._solver is not None and self._signature == signature
        if reused:
            solver = self._solver
            solver.update(P=P, q=problem.q, A=transport_A, b=transport_b)
        else:
            solver = clarabel.DefaultSolver(P, problem.q, transport_A, transport_b, cones, settings)
        self._solver = solver if self.reuse else None
        self._signature = signature if self.reuse else None
        callback_errors = []
        if progress is not None or cancellation is not None:
            from solverpilot.runtime.options import ProgressEvent
            def callback(info):
                try:
                    if cancellation is not None and cancellation.cancelled:
                        return True
                    event = ProgressEvent(self.name, 'iteration', info.iterations, info.solve_time,
                        info.cost_primal+problem.objective_offset, info.res_primal, info.res_dual)
                    return bool(progress(event)) if progress is not None else False
                except BaseException as exc:
                    callback_errors.append(exc)
                    return True
            solver.set_termination_callback(callback)
        built = perf_counter()
        self._active = True
        try:
            result = solver.solve()
        finally:
            self._active = False
            solver.unset_termination_callback()
        if callback_errors:
            raise RuntimeError('progress callback failed') from callback_errors[0]
        solved = perf_counter()
        status = str(result.status)
        candidate_statuses = {'Solved', 'AlmostSolved', 'MaxIterations', 'MaxTime', 'CallbackTerminated'}
        x = np.asarray(result.x, dtype=float) if status in candidate_statuses else None
        validation = validate_conic_solution(problem, x, atol=tol.feasibility, rtol=tol.feasibility_rel)
        objective = float(result.obj_val)+problem.objective_offset if x is not None else None
        if objective is not None and (not np.isfinite(objective) or validation.objective is None
                or abs(objective-validation.objective) > tol.objective_abs+tol.objective_rel*max(1., abs(objective), abs(validation.objective))):
            validation = replace(validation, valid=False)
        if x is not None:
            from solverpilot._immutability import readonly_array
            x = readonly_array(x, dtype=float)
        canonical_dual = np.zeros(problem.n_linear_constraints+problem.n_variables)
        transport_dual = np.asarray(result.z,dtype=float)
        for indices,sign,segment in linear_segments:
            canonical_dual[indices] += sign*transport_dual[segment]
        cone_duals = []
        from .optimality import repair_dual, verify_conic_optimality
        for block,segment in zip(problem.cones,cone_segments):
            z = transport_dual[segment].copy()
            if block.kind is ConeKind.ROTATED_SECOND_ORDER:
                z = np.r_[z[0]+z[1],z[0]-z[1],np.sqrt(2.)*z[2:]]
            elif block.kind is ConeKind.POSITIVE_SEMIDEFINITE:
                dim = block.output_shape[0];M = np.zeros((dim,dim));k = 0
                for j in range(dim):
                    for i in range(j+1):
                        M[i,j] = M[j,i] = z[k] if i==j else z[k]/np.sqrt(2.)
                        k += 1
                z = M
            cone_duals.append(repair_dual(block.kind,z))
        proof = verify_conic_optimality(problem,x,canonical_dual,cone_duals,tolerances=tol) if x is not None else None
        from dataclasses import asdict
        return ConicSolveResult(self.name, status, x, objective, validation, {
            'binding_version': self.binding_version, 'iterations': result.iterations,
            'backend_reported_optimal': status == 'Solved',
            'independently_verified_optimal': bool(proof and proof.verified and validation.valid),
            'optimality_check': None if proof is None else asdict(proof),
            'canonical_linear_dual': canonical_dual, 'canonical_cone_duals': cone_duals,
            'solverpilot_trust': {'primal_validated': validation.valid,
                'dual_verified': bool(proof and proof.dual_valid), 'gap_verified': bool(proof and proof.verified and validation.valid)},
            'backend_build_s': built-start, 'solve_s': solved-built,
            'validate_s': perf_counter()-solved,
            'primal_residual_reported': result.r_prim, 'dual_residual_reported': result.r_dual,
            'termination_evidence': 'backend_reported',
            'reuse_applied': reused, 'reuse_mode': 'same_sparsity_update' if reused else 'cold_setup',
            'reuse_report': {'workspace': 'observed' if reused else 'not_used',
                             'primal_dual_start': 'unknown', 'symbolic_factorization': 'unknown',
                             'numeric_factorization': 'unknown',
                             'reason': 'same sparsity data update' if reused else 'cold setup or changed structure/settings'},
            'transport_dual': result.z, 'transport_slack': result.s,
            'run_parameters': parameters,
        }, problem.data_hash)

    @serialized
    def close(self):
        if self._active:
            raise RuntimeError('cannot close a backend from its active progress callback')
        self._solver = self._signature = None


@lru_cache(maxsize=8)
def _runtime_conformance(binding_version):
    """Fail closed on an unavailable or numerically incompatible installed binding."""
    try:
        from solverpilot.model import Model
        cases = []
        m = Model(); x = m.variable(2, lower=[3, 4], upper=[3, 4]); t = m.variable(lower=0, upper=10)
        m.soc(t, x); m.minimize(t); cases.append((m, 5.))
        m = Model(); z = m.variable(1, lower=1, upper=1); u = m.variable(lower=0, upper=5); v = m.variable(lower=0, upper=5)
        m.rotated_soc(u, v, z); m.minimize(u+v); cases.append((m, np.sqrt(2)))
        m = Model(); t = m.variable(lower=0, upper=5)
        m.psd(t*m.constant(np.eye(2))+m.constant([[0., 1.], [1., 0.]])); m.minimize(t); cases.append((m, 1.))
        m = Model(); x = m.variable(2, lower=-10, upper=10)
        m.soc(1., x); m.minimize(((x-m.constant([3., 4.]))**2).sum()); cases.append((m, 16.))
        from solverpilot.model.sets import ExponentialCone, PowerCone
        m = Model(); x = m.variable(3, lower=[1., 1., 0.], upper=[1., 1., 10.])
        m.add_in_set(x, ExponentialCone()); m.minimize(x[2]); cases.append((m, np.e))
        m = Model(); x = m.variable(3, lower=[4., 1., -10.], upper=[4., 1., 10.])
        m.add_in_set(x, PowerCone(.5)); m.minimize(-x[2]); cases.append((m, -2.))
        from solverpilot.model.sets import GeneralizedPowerCone
        m = Model(); x = m.variable(5, lower=[1,4,9,-10,-10], upper=[1,4,9,10,10])
        m.add_in_set(x, GeneralizedPowerCone((.2,.3,.5), 2)); m.minimize(-x[3])
        cases.append((m, -float(np.exp(np.dot([.2,.3,.5], np.log([1,4,9]))))))
        for model, objective in cases:
            r = ClarabelBackend().solve(model.compile(use_cache=False).execution_ir)
            if not r.validated or r.objective_reported is None or abs(r.objective_reported-objective) > 2e-5:
                return False
        return True
    except Exception:
        return False
