"""Direct Clarabel transport with original-space validation.

PSD uses column-major upper-triangle svec, with sqrt(2) off-diagonals.
Nonsymmetric affine PSD expressions also receive explicit symmetry equations.
"""
from dataclasses import dataclass, replace
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
    tolerance: float = 1e-8
    time_limit_s: float | None = None
    threads: int = 1
    verbose: bool = False

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
                CapabilityKey.RESULT_PRIMAL, CapabilityKey.RESULT_OBJECTIVE)
        return BackendCapabilityManifestV2(
            backend=self.name, backend_version=self.binding_version or 'unavailable',
            binding='clarabel', binding_version=self.binding_version,
            adapter_version=__version__, claims={k: claim for k in keys},
            metadata={'direct_binding': True, 'independent_global_proof': False},
        )

    def solve(self, problem: ConicProblem) -> ConicSolveResult:
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
        matrices, rhs, cones = [], [], []

        def append(A, b, cone):
            matrices.append(sparse.csc_matrix(A)); rhs.append(np.asarray(b).reshape(-1)); cones.append(cone)

        A = sparse.vstack([problem.A, sparse.eye(problem.n_variables)], format='csr')
        lo = np.r_[problem.constraint_lower, problem.variable_lower]
        hi = np.r_[problem.constraint_upper, problem.variable_upper]
        eq = np.isfinite(lo) & (lo == hi)
        if np.any(eq):
            append(A[eq], lo[eq], clarabel.ZeroConeT(int(eq.sum())))
        for mask, sign, bound in [(np.isfinite(hi) & ~eq, 1, hi), (np.isfinite(lo) & ~eq, -1, lo)]:
            if np.any(mask):
                append(sign*A[mask], sign*bound[mask], clarabel.NonnegativeConeT(int(mask.sum())))
        for block in problem.cones:
            F, g = block.F, block.g
            if block.kind is ConeKind.EXPONENTIAL:
                append(-F, g, clarabel.ExponentialConeT())
            elif block.kind is ConeKind.POWER:
                append(-F, g, clarabel.PowerConeT(float(block.metadata['alpha'])))
            elif block.kind is ConeKind.SECOND_ORDER:
                append(-F, g, clarabel.SecondOrderConeT(block.dimension))
            elif block.kind is ConeKind.ROTATED_SECOND_ORDER:
                d = block.dimension
                T = sparse.lil_matrix((d, d))
                T[0, 0] = T[0, 1] = 1
                T[1, 0] = 1; T[1, 1] = -1
                for i in range(2, d):
                    T[i, i] = np.sqrt(2.)
                T = T.tocsr()
                append(-(T@F), T@g, clarabel.SecondOrderConeT(d))
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
                append(-(S@F[indices]), S@g[indices], clarabel.PSDTriangleConeT(n))
            else:
                raise ValueError(f'unsupported cone: {block.kind}')
        settings = clarabel.DefaultSettings()
        settings.verbose = self.verbose; settings.max_iter = self.max_iter
        settings.max_threads = self.threads
        settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = self.tolerance
        if self.time_limit_s is not None:
            settings.time_limit = self.time_limit_s
        solver = clarabel.DefaultSolver(sparse.triu(problem.P, format='csc'), problem.q,
                                       sparse.vstack(matrices, format='csc'), np.concatenate(rhs), cones, settings)
        built = perf_counter()
        result = solver.solve()
        solved = perf_counter()
        status = str(result.status)
        candidate_statuses = {'Solved', 'AlmostSolved', 'MaxIterations', 'MaxTime', 'CallbackTerminated'}
        x = np.asarray(result.x, dtype=float) if status in candidate_statuses else None
        validation = validate_conic_solution(problem, x, atol=1e-7, rtol=1e-7)
        objective = float(result.obj_val)+problem.objective_offset if x is not None else None
        if objective is not None and (not np.isfinite(objective) or validation.objective is None
                or abs(objective-validation.objective) > 1e-7+1e-7*max(1., abs(objective), abs(validation.objective))):
            validation = replace(validation, valid=False)
        if x is not None:
            from solverpilot._immutability import readonly_array
            x = readonly_array(x, dtype=float)
        return ConicSolveResult(self.name, status, x, objective, validation, {
            'binding_version': self.binding_version, 'iterations': result.iterations,
            'backend_reported_optimal': status == 'Solved', 'independently_verified_optimal': False,
            'backend_build_s': built-start, 'solve_s': solved-built,
            'validate_s': perf_counter()-solved,
            'primal_residual_reported': result.r_prim, 'dual_residual_reported': result.r_dual,
            'termination_evidence': 'backend_reported',
        })


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
        for model, objective in cases:
            r = ClarabelBackend().solve(model.compile(use_cache=False).execution_ir)
            if not r.validated or r.objective_reported is None or abs(r.objective_reported-objective) > 2e-5:
                return False
        return True
    except Exception:
        return False
