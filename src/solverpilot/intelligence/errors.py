"""Errors for versioned feature-intelligence contracts."""

class IntelligenceError(ValueError):
    """Base error for SolverPilot feature-intelligence validation."""


class FeatureSchemaError(IntelligenceError):
    """Raised when a feature schema or record is invalid."""


class LeakageDetectedError(IntelligenceError):
    """Raised when feature inputs cross the pre-solve/outcome boundary."""


class DistributionShiftError(IntelligenceError):
    """Raised when a distribution-shift profile cannot be built or compared safely."""
