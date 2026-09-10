class VRPError(Exception):
    """Base error for the SolverPilot routing application pack."""


class VRPValidationError(VRPError, ValueError):
    """Raised for invalid VRP data or solution objects."""


class VRPCompileError(VRPError, RuntimeError):
    """Raised when a VRP formulation cannot be compiled or decoded safely."""


class VRPReferenceLimitError(VRPError, RuntimeError):
    """Raised when an exact reference solve exceeds its declared safety limit."""
