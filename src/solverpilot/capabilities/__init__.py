from .enums import Capability, SupportLevel
from .manifest import BackendManifest
from .requirements import CapabilityRequirements, compatible, requirements_for

__all__ = [
    "BackendManifest",
    "Capability",
    "CapabilityRequirements",
    "SupportLevel",
    "compatible",
    "requirements_for",
]

from .v2 import (
    CAPABILITY_SCHEMA_VERSION, BackendCapabilityManifestV2, CapabilityCheck, CapabilityClaim,
    CapabilityEvidence, CapabilityKey, CapabilityMode, CapabilityStatus, VerificationLevel,
    project_legacy_manifest, with_verified_claims, CapabilityRequirementsV2, requirements_v2_for, compatible_v2,
)
from .conformance import BackendConformanceReport, ConformanceCheck, conform_backend_problem_classes, conform_backends

__all__ += [
    "CAPABILITY_SCHEMA_VERSION", "BackendCapabilityManifestV2", "CapabilityCheck", "CapabilityClaim",
    "CapabilityEvidence", "CapabilityKey", "CapabilityMode", "CapabilityStatus", "VerificationLevel",
    "project_legacy_manifest", "with_verified_claims", "CapabilityRequirementsV2", "requirements_v2_for", "compatible_v2", "BackendConformanceReport", "ConformanceCheck",
    "conform_backend_problem_classes", "conform_backends",
]

from .resolve import BackendCapabilityProviderV2, resolve_backend_capabilities
__all__ += ["BackendCapabilityProviderV2", "resolve_backend_capabilities"]

from .ontology import ConvexityClass, IntegralityClass, ProblemClass, ProblemDescriptor, ProofRequirement, capability_requirements_for_descriptor, describe_problem
__all__ += ["ConvexityClass","IntegralityClass","ProblemClass","ProblemDescriptor","ProofRequirement","capability_requirements_for_descriptor","describe_problem"]
