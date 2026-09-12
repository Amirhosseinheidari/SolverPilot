"""Read-only integration probes. Installation is distinct from qualification.

No solver, GPU benchmark, certificate verifier, or downloaded program is run by
these probes. A positive feature flag does not enable a production routing rule.
"""

from dataclasses import dataclass
from importlib.util import find_spec
import platform
from shutil import which
from solverpilot._immutability import deep_freeze


@dataclass(frozen=True, slots=True)
class IntegrationReadiness:
    name: str
    available: bool
    qualified: bool
    details: dict
    reason: str

    def __post_init__(self):
        object.__setattr__(self, "details", deep_freeze(self.details))


def integration_readiness():
    """Report prerequisites for future exact MILP and cuOpt qualification.

    ``qualified`` deliberately remains false: neither integration has an
    independently tested SolverPilot execution adapter in this release.
    """
    from solverpilot.globalopt import SCIPGlobalBackend

    try:
        scip = SCIPGlobalBackend().capabilities()
        exact = bool(scip.get("exact_milp") and scip.get("certificate_output"))
        exact_reason = (
            "exact parameters detected; external certificate verification and adapter qualification still required"
            if exact
            else "installed SCIP does not expose both exact mode and certificate output"
        )
    except Exception as exc:
        scip = {"available": False, "probe_error": type(exc).__name__}
        exact = False
        exact_reason = "SCIP runtime could not be loaded"
    linux = platform.system() == "Linux"
    installed = find_spec("cuopt") is not None
    gpu = {
        "platform": platform.system(),
        "linux_runtime": linux,
        "cuopt_installed": installed,
        "nvidia_smi_on_path": which("nvidia-smi") is not None,
        "gpu_execution_tested": False,
        "driver_compatibility_tested": False,
    }
    return (
        IntegrationReadiness("scip-exact-milp", exact, False, scip, exact_reason),
        IntegrationReadiness(
            "cuopt-gpu",
            linux and installed,
            False,
            gpu,
            "cuOpt requires a compatible Linux/WSL2 CUDA runtime and a separate measured qualification; CPU PDLP is available now",
        ),
    )
