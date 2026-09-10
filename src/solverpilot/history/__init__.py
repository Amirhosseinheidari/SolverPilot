"""Optional local history storage for reproducible optimization research.

Nothing is persisted unless a caller explicitly constructs :class:`HistoryStore`
and records data. The schema intentionally excludes raw problem arrays and raw
solution vectors.
"""

from .errors import HistoryDataError, HistorySchemaError, HistoryStoreError
from .model import ProblemHistoryRecord, SolveHistoryRecord
from .sqlite import HISTORY_SCHEMA_LABEL, HISTORY_SCHEMA_NAME, HISTORY_SCHEMA_VERSION, HistoryStore

__all__ = [
    "HISTORY_SCHEMA_LABEL",
    "HISTORY_SCHEMA_NAME",
    "HISTORY_SCHEMA_VERSION",
    "HistoryDataError",
    "HistorySchemaError",
    "HistoryStoreError",
    "HistoryStore",
    "ProblemHistoryRecord",
    "SolveHistoryRecord",
]
