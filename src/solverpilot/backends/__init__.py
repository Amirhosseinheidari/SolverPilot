from .base import Backend, BackendSolveResult, BackendUnavailableError
from .registry import BackendRegistry
from .scipy_highs import ScipyHighsBackend
from .scipy_highs_lp import ScipyHighsLPBackend
from .scipy_slsqp_qp import ScipySLSQPQPBackend
from .highspy_native import HighspyNativeBackend
from .osqp_native import OSQPNativeBackend
from .pyscipopt_native import PySCIPOptNativeBackend, SCIPIISResult
from .nlopt_native import NLoptNativeBackend
from .casadi_conic import CasadiConicBackend, CasadiOSQPBridgeBackend, CasadiHighsBridgeBackend, CasadiCBCBridgeBackend
from .bundled_capi import BundledOSQPCAPIBackend, BundledHighsCAPIBackend

__all__ = [
    "Backend",
    "BackendRegistry",
    "BackendSolveResult",
    "BackendUnavailableError",
    "ScipyHighsBackend",
    "ScipyHighsLPBackend",
    "ScipySLSQPQPBackend",
    "HighspyNativeBackend",
    "OSQPNativeBackend",
    "PySCIPOptNativeBackend",
    "SCIPIISResult",
    "NLoptNativeBackend",
    "CasadiConicBackend",
    "CasadiOSQPBridgeBackend",
    "CasadiHighsBridgeBackend",
    "CasadiCBCBridgeBackend",
    "BundledOSQPCAPIBackend",
    "BundledHighsCAPIBackend",
    "ScipyVendoredHighsDevBackend",
    "HighsDevIISResult",
]

from .scipy_vendored_highs_dev import ScipyVendoredHighsDevBackend, HighsDevIISResult

from .health import BackendHealthReport, BackendProbeCheck, BackendProbeStatus, probe_backend, probe_backends

__all__ += ["BackendHealthReport", "BackendProbeCheck", "BackendProbeStatus", "probe_backend", "probe_backends"]

from .protocol_v2 import (
    BackendArtifactBundle, BackendCallbackControllerV2, BackendCallbackV2, BackendProtocolV2,
    BackendSolveRequestV2, BackendSolveResultV2, CallbackEvent, CallbackEventKind,
    IncrementalBackendProtocolV2, StructuralIncrementalBackendProtocolV2,
)
__all__ += [
    "BackendArtifactBundle", "BackendCallbackControllerV2", "BackendCallbackV2", "BackendProtocolV2",
    "BackendSolveRequestV2", "BackendSolveResultV2", "CallbackEvent", "CallbackEventKind",
    "IncrementalBackendProtocolV2", "StructuralIncrementalBackendProtocolV2",
]

from .metadata import refresh_backend_metadata
__all__.append("refresh_backend_metadata")
