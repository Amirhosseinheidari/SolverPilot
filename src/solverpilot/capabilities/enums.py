from __future__ import annotations

from enum import Enum


class Capability(str, Enum):
    LP = "lp"
    CONVEX_QP = "convex_qp"
    MILP = "milp"
    MIP_START = "mip_start"
    PRIMAL_START = "primal_start"
    DUAL_START = "dual_start"
    BASIS_START = "basis_start"
    SAME_SPARSITY_DATA_UPDATE = "same_sparsity_data_update"
    STRUCTURAL_INCREMENTAL_UPDATE = "structural_incremental_update"
    NATIVE_REOPTIMIZATION = "native_reoptimization"
    CALLBACK_PROGRESS = "callback_progress"
    IIS = "iis"
    INFEASIBILITY_CERTIFICATE = "infeasibility_certificate"


class SupportLevel(str, Enum):
    NATIVE = "native"
    EMULATED_SAFE = "emulated_safe"
    EMULATED_RISKY = "emulated_risky"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"
