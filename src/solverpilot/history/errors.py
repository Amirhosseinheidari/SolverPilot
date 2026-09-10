from __future__ import annotations

from solverpilot.exceptions import SolverPilotError


class HistoryStoreError(SolverPilotError):
    """Base error for SolverPilot's optional local history store."""


class HistorySchemaError(HistoryStoreError):
    """Raised when a database schema is unsupported or inconsistent."""


class HistoryDataError(HistoryStoreError):
    """Raised when a record violates the history-store contract."""
