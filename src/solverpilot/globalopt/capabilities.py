"""Runtime evidence for the explicit global adapter, scoped to supported IRs."""

from functools import lru_cache
import numpy as np
from solverpilot.backends.metadata import version
from solverpilot.capabilities.v2 import (
    BackendCapabilityManifestV2,
    CapabilityClaim,
    CapabilityEvidence,
    CapabilityKey,
    CapabilityMode,
    CapabilityStatus,
    VerificationLevel,
)


@lru_cache(maxsize=8)
def global_conformance(binding_version):
    try:
        from .backend import SCIPGlobalBackend, solve_global
        from .problem import GlobalQuadraticProblem
        from .expressions import absolute
        from solverpilot.model import Model

        backend = SCIPGlobalBackend(time_limit_s=10)
        p = GlobalQuadraticProblem.from_data(
            P=[[-2]],
            A=np.empty((0, 1)),
            q=[0],
            variable_lower=[-2],
            variable_upper=[2],
            constraint_lower=[],
            constraint_upper=[],
            domains=["integer"],
        )
        r = solve_global(p, backend=backend)
        if not r.validation.valid or abs(r.objective + 4) > 1e-7:
            return False
        m = Model()
        x = m.variable(lower=-2, upper=2)
        b = m.variable(lower=1, upper=1, domain="binary")
        m.indicator(b, x * x <= 1)
        m.add(x * x == 1)
        m.minimize(-absolute(x))
        r = solve_global(m, backend=backend)
        return r.validation.valid and abs(r.objective + 1) < 1e-7
    except Exception:
        return False


def global_manifest(backend):
    from solverpilot import __version__

    binding = version("pyscipopt") if backend.is_available() else None
    verified = backend.is_available() and global_conformance(binding)
    native = backend.capabilities().get("native_version", "unavailable")
    evidence = CapabilityEvidence(
        "scip-global-analytic-v1",
        "native-runtime-conformance",
        backend_versions=(native,),
        binding_versions=(() if binding is None else (binding,)),
        adapter_versions=(__version__,),
        notes=("Nonconvex integer quadratic and bounded nonlinear equality/indicator/abs probes.",),
    )
    restrictions = {
        "accepted_ir": ["GlobalQuadraticProblem", "FactorableProblem"],
        "factorable_bounds": "finite",
        "expressions": "documented algebraic whitelist",
        "indicator_bounds": "exact polynomial interval enclosure required",
        "global_evidence": "solver_reported_numerical",
    }
    claim = CapabilityClaim(
        CapabilityStatus.RESTRICTED if verified else CapabilityStatus.UNVERIFIED,
        CapabilityMode.NATIVE,
        VerificationLevel.VERIFIED if verified else VerificationLevel.UNVERIFIED,
        restrictions=restrictions,
        evidence=(evidence,) if verified else (),
    )
    keys = (
        CapabilityKey.PROBLEM_NONCONVEX_QP,
        CapabilityKey.PROBLEM_MIQP,
        CapabilityKey.PROBLEM_NLP,
        CapabilityKey.PROBLEM_MINLP,
        CapabilityKey.CONSTRAINT_NONLINEAR,
        CapabilityKey.RESULT_PRIMAL,
        CapabilityKey.RESULT_OBJECTIVE,
    )
    # Indicators are an interval reformulation, never advertised as native.
    indicator = CapabilityClaim(
        CapabilityStatus.RESTRICTED if verified else CapabilityStatus.UNVERIFIED,
        CapabilityMode.EMULATED_SAFE,
        VerificationLevel.VERIFIED if verified else VerificationLevel.UNVERIFIED,
        restrictions=restrictions,
        evidence=(evidence,) if verified else (),
    )
    return BackendCapabilityManifestV2(
        backend=backend.name,
        backend_version=native,
        binding="pyscipopt",
        binding_version=binding,
        adapter_version=__version__,
        claims={**{k: claim for k in keys}, CapabilityKey.CONSTRAINT_INDICATOR: indicator},
        metadata={"independent_global_proof": False, "automatic_selection": False},
    )
