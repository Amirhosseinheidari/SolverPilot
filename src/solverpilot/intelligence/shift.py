"""Simple, explainable distribution-shift profiles for feature records."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import math
from statistics import mean, pstdev
from typing import Any

from solverpilot.benchmark.splits import SplitRecord, validate_split_records
from solverpilot.io.hashing import deep_freeze_json, deep_thaw_json, sha256_json

from .errors import DistributionShiftError
from .schema import FeatureDataType, FeatureRecord, FeatureSchema


class ShiftSeverity(str, Enum):
    IN_DISTRIBUTION = "in_distribution"
    MILD_SHIFT = "mild_shift"
    STRONG_SHIFT = "strong_shift"


@dataclass(frozen=True, slots=True)
class DistributionShiftProfile:
    feature_schema_id: str
    training_record_ids: tuple[str, ...]
    training_instance_ids: tuple[str, ...]
    statistics: Mapping[str, Any]
    profile_id: str = field(init=False)

    def __post_init__(self) -> None:
        record_ids = tuple(sorted(map(str, self.training_record_ids)))
        instance_ids = tuple(sorted(map(str, self.training_instance_ids)))
        if not record_ids or len(record_ids) != len(set(record_ids)):
            raise DistributionShiftError("training_record_ids must be non-empty and unique")
        if not instance_ids or len(instance_ids) != len(set(instance_ids)):
            raise DistributionShiftError("training_instance_ids must be non-empty and unique")
        if len(record_ids) != len(instance_ids):
            raise DistributionShiftError("training record and instance counts must match")
        frozen = deep_freeze_json(self.statistics)
        payload = {
            "feature_schema_id": self.feature_schema_id,
            "training_record_ids": list(record_ids),
            "training_instance_ids": list(instance_ids),
            "statistics": deep_thaw_json(frozen),
        }
        object.__setattr__(self, "training_record_ids", record_ids)
        object.__setattr__(self, "training_instance_ids", instance_ids)
        object.__setattr__(self, "statistics", frozen)
        object.__setattr__(self, "profile_id", sha256_json(payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "feature_schema_id": self.feature_schema_id,
            "training_record_ids": list(self.training_record_ids),
            "training_instance_ids": list(self.training_instance_ids),
            "statistics": deep_thaw_json(self.statistics),
        }


@dataclass(frozen=True, slots=True)
class ShiftFlag:
    feature: str
    code: str
    severity: ShiftSeverity
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", deep_freeze_json(self.details))

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "code": self.code,
            "severity": self.severity.value,
            "details": deep_thaw_json(self.details),
        }


@dataclass(frozen=True, slots=True)
class DistributionShiftAssessment:
    profile_id: str
    record_id: str
    severity: ShiftSeverity
    flags: tuple[ShiftFlag, ...]
    assessment_id: str = field(init=False)

    def __post_init__(self) -> None:
        flags = tuple(self.flags)
        payload = {
            "profile_id": self.profile_id,
            "record_id": self.record_id,
            "severity": self.severity.value,
            "flags": [x.to_dict() for x in flags],
        }
        object.__setattr__(self, "flags", flags)
        object.__setattr__(self, "assessment_id", sha256_json(payload))

    @property
    def shifted(self) -> bool:
        return self.severity is not ShiftSeverity.IN_DISTRIBUTION

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "profile_id": self.profile_id,
            "record_id": self.record_id,
            "severity": self.severity.value,
            "flags": [x.to_dict() for x in self.flags],
        }


def fit_distribution_shift_profile(schema: FeatureSchema, records: Sequence[FeatureRecord]) -> DistributionShiftProfile:
    rows = tuple(records)
    if not rows:
        raise DistributionShiftError("at least one training record is required")
    if len({x.record_id for x in rows}) != len(rows):
        raise DistributionShiftError("training feature records must be unique")
    if len({x.instance_id for x in rows}) != len(rows):
        raise DistributionShiftError("training instance IDs must be unique")
    if any(x.feature_schema_id != schema.feature_schema_id for x in rows):
        raise DistributionShiftError("all training records must match the feature schema")
    definitions = {x.name: x for x in schema.definitions}
    statistics: dict[str, dict[str, Any]] = {}
    for name, definition in definitions.items():
        values = [row.values[name] for row in rows if row.values[name] is not None]
        missing = len(rows) - len(values)
        if definition.data_type in (FeatureDataType.INTEGER, FeatureDataType.NUMBER):
            numbers = [float(v) for v in values]
            statistics[name] = {
                "kind": "numeric", "count": len(numbers), "missing_count": missing,
                "min": min(numbers) if numbers else None,
                "max": max(numbers) if numbers else None,
                "mean": mean(numbers) if numbers else None,
                "std": pstdev(numbers) if len(numbers) > 1 else 0.0,
            }
        elif definition.data_type is FeatureDataType.BOOLEAN:
            statistics[name] = {
                "kind": "boolean", "count": len(values), "missing_count": missing,
                "true_rate": (sum(bool(v) for v in values) / len(values)) if values else None,
            }
        else:
            counts = {value: values.count(value) for value in sorted(set(values))}
            statistics[name] = {
                "kind": "categorical", "count": len(values), "missing_count": missing, "counts": counts,
            }
    return DistributionShiftProfile(
        schema.feature_schema_id,
        tuple(x.record_id for x in rows),
        tuple(x.instance_id for x in rows),
        statistics,
    )


def fit_training_shift_profile(
    schema: FeatureSchema,
    records: Sequence[FeatureRecord],
    split_records: Sequence[SplitRecord],
) -> DistributionShiftProfile:
    rows = tuple(records)
    split_rows = tuple(split_records)
    report = validate_split_records(split_rows, expected_instances=[x.instance_id for x in rows])
    if not report.ok:
        raise DistributionShiftError("invalid benchmark split for feature profile: " + "; ".join(report.errors))
    train_ids = {x.instance for x in split_rows if x.split == "train"}
    training = tuple(x for x in rows if x.instance_id in train_ids)
    if not training:
        raise DistributionShiftError("benchmark split contains no training feature records")
    return fit_distribution_shift_profile(schema, training)


def assess_distribution_shift(
    profile: DistributionShiftProfile,
    record: FeatureRecord,
    *,
    z_threshold: float = 3.0,
) -> DistributionShiftAssessment:
    if record.feature_schema_id != profile.feature_schema_id:
        raise DistributionShiftError("record schema does not match shift profile")
    if type(z_threshold) not in (int, float) or not math.isfinite(float(z_threshold)) or float(z_threshold) <= 0:
        raise DistributionShiftError("z_threshold must be a finite positive number")
    flags: list[ShiftFlag] = []
    for name, stat_frozen in profile.statistics.items():
        stat = deep_thaw_json(stat_frozen)
        value = record.values[name]
        if value is None:
            if stat["missing_count"] == 0:
                flags.append(ShiftFlag(name, "new_missingness", ShiftSeverity.MILD_SHIFT))
            continue
        if stat["kind"] == "numeric" and stat["count"]:
            numeric = float(value)
            if numeric < stat["min"] or numeric > stat["max"]:
                flags.append(ShiftFlag(name, "outside_training_range", ShiftSeverity.MILD_SHIFT, {
                    "value": numeric, "min": stat["min"], "max": stat["max"],
                }))
            if stat["std"]:
                z = abs(numeric - stat["mean"]) / stat["std"]
                if z > float(z_threshold):
                    flags.append(ShiftFlag(name, "large_standardized_distance", ShiftSeverity.STRONG_SHIFT, {"z": z}))
        elif stat["kind"] == "categorical" and value not in stat["counts"]:
            flags.append(ShiftFlag(name, "unseen_category", ShiftSeverity.STRONG_SHIFT, {"value": value}))
    severity = (
        ShiftSeverity.STRONG_SHIFT if any(x.severity is ShiftSeverity.STRONG_SHIFT for x in flags)
        else ShiftSeverity.MILD_SHIFT if flags else ShiftSeverity.IN_DISTRIBUTION
    )
    return DistributionShiftAssessment(profile.profile_id, record.record_id, severity, tuple(flags))
