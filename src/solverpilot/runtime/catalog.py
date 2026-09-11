"""One discoverable catalog for core and specialized problem families."""


def backend_catalog():
    from .auto import builtin_backend_candidates
    from solverpilot.conic import ClarabelBackend, CasadiSuperSCSBackend
    from solverpilot.nlp import CasadiIpoptBackend
    from solverpilot.cp import ORToolsCPSATBackend

    result = {b.manifest.name: b for b in builtin_backend_candidates()}
    for backend in (
        ClarabelBackend(),
        CasadiSuperSCSBackend(),
        CasadiIpoptBackend(),
        ORToolsCPSATBackend(),
    ):
        result[backend.name] = backend
    return result


def capability_manifest(backend):
    if hasattr(backend, "capability_manifest_v2"):
        manifest = backend.capability_manifest_v2
        return manifest() if callable(manifest) else manifest
    from solverpilot.capabilities import project_legacy_manifest
    from solverpilot import __version__

    return project_legacy_manifest(backend.manifest, adapter_version=__version__)


def verify_specialized_backend(backend):
    if not backend.is_available():
        return False
    try:
        if backend.name == "clarabel-native":
            from solverpilot.conic.clarabel_backend import _runtime_conformance

            return _runtime_conformance(backend.binding_version)
        from solverpilot.model import Model

        if backend.name == "casadi-ipopt-nlp-bridge":
            import numpy as np

            m = Model()
            x = m.variable(lower=-1.0, upper=1.0)
            m.minimize(x.exp())
            r = m.compile().solve(backend=backend)
            return r.validation.valid and abs(r.objective_reported - np.exp(-1)) < 1e-6
        if "superscs" in backend.name:
            m = Model()
            x = m.variable(2, lower=[3.0, 4.0], upper=[3.0, 4.0])
            t = m.variable(lower=0, upper=10)
            m.soc(t, x)
            m.minimize(t)
            r = m.compile().solve(backend=backend)
            return r.validated and abs(r.objective_reported - 5.0) < 1e-5
        if "ortools" in backend.name:
            from solverpilot.cp import CPModel

            m = CPModel()
            x = m.int_var(0, 2, name="x")
            m.minimize(x)
            r = m.solve(backend=backend)
            return r.validation.valid and r.objective == 0
    except Exception:
        return False
    return False
