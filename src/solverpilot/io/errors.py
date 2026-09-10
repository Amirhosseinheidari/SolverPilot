"""Fail-closed exceptions for local data ingestion."""

from __future__ import annotations

from solverpilot.exceptions import SolverPilotError


class DataIOError(SolverPilotError):
    """Base class for deterministic SolverPilot data-I/O failures."""


class DataIOConfigurationError(DataIOError, ValueError):
    """Raised when an I/O or mapping configuration is invalid."""


class DataIOSourceError(DataIOError):
    """Raised when a local source cannot be inspected or read safely."""


class DataIOParseError(DataIOError, ValueError):
    """Raised when source bytes cannot be parsed under the declared policy."""


class DataIOMappingError(DataIOError, ValueError):
    """Raised when explicit normalization or field mapping fails."""
