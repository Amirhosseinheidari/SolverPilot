"""Explicit extension infrastructure for trusted in-process SolverPilot integrations.

Extensions are never auto-discovered or auto-executed merely by importing this package.
Backends and bridges continue to use SolverPilot's canonical registries.
"""

from .entrypoints import resolve_entrypoint, resolve_manifest_entrypoint
from .errors import (
    ExtensionActivationError,
    ExtensionDependencyError,
    ExtensionError,
    ExtensionManifestError,
    ExtensionRegistrationError,
    ExtensionRegistryFrozenError,
    ExtensionResolutionError,
    ExtensionValidationError,
)
from .manager import ExtensionManager
from .manifest import ExtensionDependency, ExtensionKind, ExtensionLifecycle, ExtensionManifest
from .model import (
    ExportKind,
    Extension,
    ExtensionActivationPlan,
    ExtensionContributions,
    ExtensionSelfCheck,
    NamedExtensionExport,
    StagedExtensionContribution,
)
from .versioning import SemanticVersion, version_satisfies

__all__ = [
    "ExportKind",
    "Extension",
    "ExtensionActivationError",
    "ExtensionActivationPlan",
    "ExtensionContributions",
    "ExtensionDependency",
    "ExtensionDependencyError",
    "ExtensionError",
    "ExtensionKind",
    "ExtensionLifecycle",
    "ExtensionManager",
    "ExtensionManifest",
    "ExtensionManifestError",
    "ExtensionRegistrationError",
    "ExtensionRegistryFrozenError",
    "ExtensionResolutionError",
    "ExtensionSelfCheck",
    "ExtensionValidationError",
    "NamedExtensionExport",
    "SemanticVersion",
    "StagedExtensionContribution",
    "resolve_entrypoint",
    "resolve_manifest_entrypoint",
    "version_satisfies",
]
