from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable


_ALLOWED_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True, slots=True)
class SplitRecord:
    instance: str
    split: str
    group: str | None = None


@dataclass(frozen=True, slots=True)
class SplitValidation:
    ok: bool
    instances: int
    groups: int
    split_counts: dict[str, int]
    errors: tuple[str, ...]
    sha256: str

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "instances": self.instances,
            "groups": self.groups,
            "split_counts": self.split_counts,
            "errors": list(self.errors),
            "sha256": self.sha256,
        }


def load_split_records(path: str | Path) -> tuple[SplitRecord, ...]:
    raw = Path(path).read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("split artifact must be a JSON object keyed by instance")
    out: list[SplitRecord] = []
    for instance, entry in payload.items():
        if isinstance(entry, str):
            entry = {"split": entry}
        if not isinstance(entry, dict):
            raise ValueError(f"split entry for {instance!r} must be a string or object")
        split = str(entry.get("split", "")).lower()
        if split not in _ALLOWED_SPLITS:
            raise ValueError(f"invalid split for {instance!r}: {split!r}")
        group = entry.get("group")
        out.append(SplitRecord(str(instance), split, None if group is None else str(group)))
    return tuple(out)


def validate_split_records(
    records: Iterable[SplitRecord],
    *,
    expected_instances: Iterable[str] | None = None,
    artifact_bytes: bytes | None = None,
) -> SplitValidation:
    records = tuple(records)
    errors: list[str] = []
    names = [r.instance for r in records]
    if len(names) != len(set(names)):
        errors.append("duplicate instance in split artifact")
    if expected_instances is not None:
        expected = set(map(str, expected_instances))
        observed = set(names)
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if missing:
            errors.append(f"missing split assignments: {missing[:10]}")
        if extra:
            errors.append(f"unexpected split assignments: {extra[:10]}")

    group_splits: dict[str, set[str]] = {}
    for r in records:
        group = r.group if r.group is not None else r.instance
        group_splits.setdefault(group, set()).add(r.split)
    leaked = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaked:
        errors.append(f"group leakage across splits: {leaked[:10]}")

    counts = {name: sum(r.split == name for r in records) for name in sorted(_ALLOWED_SPLITS)}
    if artifact_bytes is None:
        canonical = json.dumps(
            {r.instance: {"split": r.split, "group": r.group} for r in sorted(records, key=lambda x: x.instance)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        artifact_bytes = canonical
    return SplitValidation(
        ok=not errors,
        instances=len(records),
        groups=len(group_splits),
        split_counts=counts,
        errors=tuple(errors),
        sha256=hashlib.sha256(artifact_bytes).hexdigest(),
    )


def validate_split_file(path: str | Path, *, expected_instances: Iterable[str] | None = None) -> SplitValidation:
    p = Path(path)
    return validate_split_records(load_split_records(p), expected_instances=expected_instances, artifact_bytes=p.read_bytes())


def generate_group_split(
    instance_to_group: dict[str, str | None],
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    test_fraction: float = 0.20,
    seed: int = 0,
) -> tuple[SplitRecord, ...]:
    """Create a deterministic, group-safe train/validation/test split.

    Groups are indivisible. Assignment is deterministic for a mapping/seed and does
    not consume benchmark outcomes, winners, runtimes, or objectives.
    """
    fractions = {"train": float(train_fraction), "validation": float(validation_fraction), "test": float(test_fraction)}
    if any(v < 0 or v > 1 for v in fractions.values()):
        raise ValueError("split fractions must be in [0, 1]")
    if abs(sum(fractions.values()) - 1.0) > 1e-12:
        raise ValueError("split fractions must sum to 1")
    if not instance_to_group:
        raise ValueError("instance_to_group must not be empty")
    groups: dict[str, list[str]] = {}
    for instance, raw_group in instance_to_group.items():
        instance = str(instance); group = instance if raw_group is None else str(raw_group)
        groups.setdefault(group, []).append(instance)
    n = len(instance_to_group); targets = {name: fractions[name] * n for name in fractions}; current = {name: 0 for name in fractions}
    def group_rank(item):
        group, members = item
        return (-len(members), hashlib.sha256(f"{seed}\0{group}".encode()).hexdigest())
    assignment: dict[str, str] = {}; split_order = ("train", "validation", "test")
    for group, members in sorted(groups.items(), key=group_rank):
        size=len(members); best_split=None; best_score=None
        for split in split_order:
            candidate=dict(current); candidate[split]+=size
            score=sum(((candidate[name]-targets[name])/max(targets[name],1.0))**2 for name in split_order)
            key=(score,split_order.index(split))
            if best_score is None or key < best_score: best_score=key; best_split=split
        if best_split is None: raise RuntimeError("failed to assign split")
        assignment[group]=best_split; current[best_split]+=size
    records=tuple(SplitRecord(str(instance), assignment[str(instance) if group is None else str(group)], None if group is None else str(group)) for instance,group in sorted(instance_to_group.items()))
    validation=validate_split_records(records, expected_instances=instance_to_group.keys())
    if not validation.ok: raise RuntimeError("generated split failed its own leakage/coverage validation: "+"; ".join(validation.errors))
    return records


def save_split_records(records: Iterable[SplitRecord], path: str | Path, *, seed: int | None = None) -> SplitValidation:
    records=tuple(records)
    payload={r.instance: ({"split":r.split} if r.group is None else {"split":r.split,"group":r.group}) for r in sorted(records,key=lambda x:x.instance)}
    p=Path(path); p.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8")
    validation=validate_split_file(p)
    meta={"schema_version":"solverpilot.split-meta.v1","split_sha256":validation.sha256,"seed":seed,"instances":validation.instances,"groups":validation.groups,"split_counts":validation.split_counts,"group_leakage_checked":validation.ok}
    Path(str(p)+".meta.json").write_text(json.dumps(meta,indent=2,sort_keys=True),encoding="utf-8")
    return validation


def instances_by_split(records: Iterable[SplitRecord]) -> dict[str, tuple[str, ...]]:
    grouped={name:[] for name in sorted(_ALLOWED_SPLITS)}
    for record in records: grouped[record.split].append(record.instance)
    return {name:tuple(sorted(values)) for name,values in grouped.items()}


def filter_rows_by_split(rows: Iterable[dict], records: Iterable[SplitRecord], *, allowed_splits: tuple[str, ...] = ("train",)) -> list[dict]:
    allowed=tuple(str(x).lower() for x in allowed_splits)
    if not allowed or any(name not in _ALLOWED_SPLITS for name in allowed):
        raise ValueError("allowed_splits must contain only train/validation/test")
    mapping={r.instance:r.split for r in records}; out=[]
    for row in rows:
        instance=str(row.get("instance") or "")
        if not instance: raise ValueError("benchmark row is missing instance")
        if instance not in mapping: raise ValueError(f"benchmark row instance {instance!r} is absent from frozen split")
        if mapping[instance] in allowed: out.append(dict(row))
    return out
