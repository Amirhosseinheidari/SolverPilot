"""Feature contracts and conservative distribution-shift diagnostics.

This package does not select a solver. It only standardizes pre-solve features,
audits leakage boundaries, and reports simple explainable shift evidence.
"""
from .errors import DistributionShiftError, FeatureSchemaError, IntelligenceError, LeakageDetectedError
from .fingerprint import PROBLEM_FINGERPRINT_FEATURE_SCHEMA, feature_record_from_problem, problem_fingerprint_feature_schema
from .leakage import LeakageAuditReport, LeakageFinding, audit_feature_boundary
from .schema import (
    FeatureDataType,
    FeatureDefinition,
    FeatureRecord,
    FeatureSchema,
    MissingnessPolicy,
    build_feature_record,
    build_feature_schema,
)
from .shift import (
    DistributionShiftAssessment,
    DistributionShiftProfile,
    ShiftFlag,
    ShiftSeverity,
    assess_distribution_shift,
    fit_distribution_shift_profile,
    fit_training_shift_profile,
)

__all__ = [
    "IntelligenceError", "FeatureSchemaError", "LeakageDetectedError", "DistributionShiftError",
    "FeatureDataType", "MissingnessPolicy", "FeatureDefinition", "FeatureSchema", "FeatureRecord",
    "build_feature_schema", "build_feature_record",
    "LeakageFinding", "LeakageAuditReport", "audit_feature_boundary",
    "ShiftSeverity", "ShiftFlag", "DistributionShiftProfile", "DistributionShiftAssessment",
    "fit_distribution_shift_profile", "fit_training_shift_profile", "assess_distribution_shift",
    "PROBLEM_FINGERPRINT_FEATURE_SCHEMA", "problem_fingerprint_feature_schema", "feature_record_from_problem",
]
