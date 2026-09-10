from __future__ import annotations

from solverpilot.exceptions import SolverPilotError


class ExtensionError(SolverPilotError):
    """Base class for SolverPilot extension-system errors."""


class ExtensionManifestError(ExtensionError, ValueError):
    """Raised when an extension manifest is malformed or non-canonical."""


class ExtensionRegistrationError(ExtensionError, ValueError):
    """Raised when extension registration would violate registry invariants."""


class ExtensionDependencyError(ExtensionError, ValueError):
    """Raised when extension dependencies are missing, incompatible, or cyclic."""


class ExtensionRegistryFrozenError(ExtensionError, RuntimeError):
    """Raised when a frozen extension manager is mutated."""


class ExtensionValidationError(ExtensionError, RuntimeError):
    """Raised when an extension fails a self-check or contribution validation."""


class ExtensionActivationError(ExtensionError, RuntimeError):
    """Raised when a staged activation plan cannot be safely applied."""


class ExtensionResolutionError(ExtensionError, ImportError):
    """Raised when an explicitly requested extension entry point cannot be resolved."""
