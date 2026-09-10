"""Safe local data ingestion with deterministic mapping and provenance.

This subpackage is additive and intentionally not re-exported from ``solverpilot``
so the frozen top-level public API remains unchanged.
"""

from .csv import load_csv
from .errors import (
    DataIOConfigurationError,
    DataIOError,
    DataIOMappingError,
    DataIOParseError,
    DataIOSourceError,
)
from .hashing import canonical_json_bytes, sha256_file, sha256_json
from .json import load_json
from .model import CsvMapping, CsvTable, DataIssue, LoadPolicy, MappingRule, ObjectMapping
from .provenance import SourceFileProvenance, SourceProvenance
from .result import LoadedData

__all__ = [
    "CsvMapping",
    "CsvTable",
    "DataIOConfigurationError",
    "DataIOError",
    "DataIOMappingError",
    "DataIOParseError",
    "DataIOSourceError",
    "DataIssue",
    "LoadPolicy",
    "LoadedData",
    "MappingRule",
    "ObjectMapping",
    "SourceFileProvenance",
    "SourceProvenance",
    "canonical_json_bytes",
    "load_csv",
    "load_json",
    "sha256_file",
    "sha256_json",
]
