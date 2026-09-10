from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterable, Iterator, Mapping, Sequence

from solverpilot.inspect import inspect_problem
from solverpilot.intelligence import FeatureRecord
from solverpilot.io.provenance import SourceProvenance
from solverpilot.problem import LinearProblem, QuadraticProblem
from solverpilot.runtime import SolveResult

from .errors import HistoryDataError, HistorySchemaError, HistoryStoreError
from .model import ProblemHistoryRecord, SolveHistoryRecord

HISTORY_SCHEMA_NAME = "solverpilot.history"
HISTORY_SCHEMA_VERSION = 1
HISTORY_SCHEMA_LABEL = "1.0.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise HistoryDataError("history payload must contain JSON-safe finite values") from exc


def _json_load(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception as exc:
        raise HistoryStoreError("history database contains invalid JSON") from exc


def _bool_db(value: bool | None) -> int | None:
    return None if value is None else int(value)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _run_id_from_result(result: SolveResult) -> str:
    raw = "\0".join(
        [
            result.trace.problem_data_hash,
            str(result.trace.backend or ""),
            str(result.trace.created_at_utc),
            str(result.trace.termination or result.backend_status),
        ]
    ).encode("utf-8")
    return "solve_" + hashlib.sha256(raw).hexdigest()[:32]


def _require_sha(value: str, field_name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise HistoryDataError(f"{field_name} must be a SHA-256 digest")
    return text


class HistoryStore:
    """Explicitly enabled local SQLite history for research and reproducibility.

    SolverPilot never constructs this store automatically. The durable schema
    excludes raw problem arrays, raw request payloads, raw solution vectors, and
    backend ``raw_statistics``.
    """

    def __init__(self, path: str | Path, *, timeout_s: float = 5.0) -> None:
        self.path = Path(path)
        if not str(self.path).strip():
            raise HistoryStoreError("history path must be non-empty")
        if self.path.exists() and self.path.is_dir():
            raise HistoryStoreError("history path must point to a SQLite file")
        if self.path.is_symlink():
            raise HistoryStoreError("history database path may not be a symlink")
        if type(timeout_s) not in (int, float) or not math.isfinite(float(timeout_s)) or float(timeout_s) <= 0:
            raise HistoryStoreError("timeout_s must be finite and positive")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        try:
            self._connection = sqlite3.connect(
                str(self.path),
                timeout=float(timeout_s),
                isolation_level=None,
                check_same_thread=False,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute(f"PRAGMA busy_timeout = {int(float(timeout_s) * 1000)}")
            self._initialize_or_migrate()
        except HistoryStoreError:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            raise
        except sqlite3.DatabaseError as exc:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            raise HistoryStoreError("history database is not a valid/compatible SQLite database") from exc
        except Exception:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            raise

    def __enter__(self) -> "HistoryStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                yield self._connection
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            else:
                if not self._connection.in_transaction:
                    raise HistoryStoreError("history transaction ended unexpectedly")
                self._connection.execute("COMMIT")

    def _initialize_or_migrate(self) -> None:
        with self._lock:
            version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
            if version > HISTORY_SCHEMA_VERSION:
                raise HistorySchemaError(
                    f"history database schema {version} is newer than supported {HISTORY_SCHEMA_VERSION}"
                )
            while version < HISTORY_SCHEMA_VERSION:
                migrator = getattr(self, f"_migrate_{version}_to_{version + 1}", None)
                if migrator is None:
                    raise HistorySchemaError(f"missing history migration {version}->{version + 1}")
                migrator()
                version += 1
            self._validate_schema_identity()

    def _migrate_0_to_1(self) -> None:
        now = _utc_now()
        script = f"""
            BEGIN IMMEDIATE;
            CREATE TABLE history_metadata (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL
            );
            CREATE TABLE problems (
                problem_data_hash TEXT PRIMARY KEY,
                problem_structural_hash TEXT NOT NULL,
                problem_class TEXT NOT NULL,
                n_variables INTEGER NOT NULL CHECK(n_variables >= 0),
                n_constraints INTEGER NOT NULL CHECK(n_constraints >= 0),
                metadata_json TEXT NOT NULL,
                created_at_utc TEXT NOT NULL
            );
            CREATE TABLE problem_sources (
                source_id TEXT PRIMARY KEY,
                problem_data_hash TEXT NOT NULL,
                provenance_json TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                FOREIGN KEY(problem_data_hash) REFERENCES problems(problem_data_hash) ON DELETE CASCADE
            );
            CREATE TABLE feature_records (
                record_id TEXT PRIMARY KEY,
                problem_data_hash TEXT NOT NULL,
                instance_id TEXT NOT NULL,
                feature_schema_id TEXT NOT NULL,
                feature_schema_version TEXT NOT NULL,
                values_json TEXT NOT NULL,
                missingness_reasons_json TEXT NOT NULL,
                source_content_sha256 TEXT NOT NULL,
                extractor_id TEXT NOT NULL,
                extractor_version TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                FOREIGN KEY(problem_data_hash) REFERENCES problems(problem_data_hash) ON DELETE CASCADE
            );
            CREATE TABLE solve_runs (
                run_id TEXT PRIMARY KEY,
                problem_data_hash TEXT NOT NULL,
                problem_structural_hash TEXT NOT NULL,
                problem_class TEXT NOT NULL,
                backend TEXT,
                backend_version TEXT,
                public_status TEXT NOT NULL,
                backend_status TEXT NOT NULL,
                objective REAL,
                validation_valid INTEGER,
                independently_verified_optimal INTEGER NOT NULL,
                total_s REAL,
                environment_id TEXT,
                trace_created_at_utc TEXT,
                metadata_json TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                FOREIGN KEY(problem_data_hash) REFERENCES problems(problem_data_hash) ON DELETE RESTRICT
            );
            CREATE TABLE benchmark_rows (
                run_id TEXT PRIMARY KEY,
                protocol_id TEXT NOT NULL,
                environment_id TEXT,
                problem_data_hash TEXT,
                instance_name TEXT NOT NULL,
                instance_sha256 TEXT,
                backend TEXT NOT NULL,
                repetition INTEGER NOT NULL CHECK(repetition >= 0),
                state TEXT NOT NULL,
                public_status TEXT,
                validated INTEGER,
                objective REAL,
                wall_s REAL,
                reference_check TEXT,
                row_json TEXT NOT NULL,
                created_at_utc TEXT NOT NULL
            );
            CREATE TABLE artifacts (
                artifact_id TEXT PRIMARY KEY,
                run_id TEXT,
                artifact_type TEXT NOT NULL,
                path TEXT,
                size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
                sha256 TEXT,
                metadata_json TEXT NOT NULL,
                created_at_utc TEXT NOT NULL
            );
            CREATE INDEX idx_problem_sources_problem ON problem_sources(problem_data_hash);
            CREATE INDEX idx_features_problem ON feature_records(problem_data_hash);
            CREATE INDEX idx_solve_problem_backend ON solve_runs(problem_data_hash, backend);
            CREATE INDEX idx_solve_backend_status ON solve_runs(backend, public_status);
            CREATE INDEX idx_benchmark_protocol_backend ON benchmark_rows(protocol_id, backend);
            CREATE INDEX idx_benchmark_instance ON benchmark_rows(instance_name, instance_sha256);
            INSERT INTO history_metadata(key,value_json,updated_at_utc)
              VALUES('schema_name','\"solverpilot.history\"','{now}');
            INSERT INTO history_metadata(key,value_json,updated_at_utc)
              VALUES('schema_label','\"1.0.0\"','{now}');
            PRAGMA user_version = 1;
            COMMIT;
        """
        try:
            self._connection.executescript(script)
        except Exception:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def _validate_schema_identity(self) -> None:
        expected = {"schema_name": HISTORY_SCHEMA_NAME, "schema_label": HISTORY_SCHEMA_LABEL}
        try:
            rows = {
                row["key"]: _json_load(row["value_json"])
                for row in self._connection.execute(
                    "SELECT key,value_json FROM history_metadata WHERE key IN ('schema_name','schema_label')"
                )
            }
        except sqlite3.Error as exc:
            raise HistorySchemaError("history schema metadata is missing or unreadable") from exc
        if rows != expected:
            raise HistorySchemaError("history schema identity does not match SolverPilot")

    @staticmethod
    def _insert_or_verify(
        con: sqlite3.Connection,
        *,
        table: str,
        key_column: str,
        columns: Sequence[str],
        values: Sequence[Any],
        compare_columns: Sequence[str] | None = None,
    ) -> bool:
        key_index = columns.index(key_column)
        key = values[key_index]
        existing = con.execute(f"SELECT * FROM {table} WHERE {key_column}=?", (key,)).fetchone()
        if existing is None:
            placeholders = ",".join("?" for _ in columns)
            con.execute(
                f"INSERT INTO {table}({','.join(columns)}) VALUES({placeholders})",
                tuple(values),
            )
            return True
        checked = tuple(compare_columns or columns)
        mismatch = [name for name in checked if existing[name] != values[columns.index(name)]]
        if mismatch:
            raise HistoryDataError(
                f"conflicting history record for {table}.{key_column}={key!r}; fields={mismatch}"
            )
        return False

    def summary(self) -> dict[str, Any]:
        tables = ("problems", "problem_sources", "feature_records", "solve_runs", "benchmark_rows", "artifacts")
        with self._lock:
            counts = {
                name: int(self._connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
                for name in tables
            }
        return {
            "path": str(self.path),
            "schema_name": HISTORY_SCHEMA_NAME,
            "schema_version": HISTORY_SCHEMA_VERSION,
            "schema_label": HISTORY_SCHEMA_LABEL,
            "explicit_opt_in": True,
            "raw_problem_payloads_persisted": False,
            "raw_solution_vectors_persisted": False,
            "counts": counts,
        }

    def integrity_check(self) -> dict[str, Any]:
        with self._lock:
            integrity = [str(row[0]) for row in self._connection.execute("PRAGMA integrity_check").fetchall()]
            foreign = [tuple(row) for row in self._connection.execute("PRAGMA foreign_key_check").fetchall()]
        return {"ok": integrity == ["ok"] and not foreign, "integrity": integrity, "foreign_key_violations": foreign}

    def record_problem(self, record: ProblemHistoryRecord) -> None:
        if not isinstance(record, ProblemHistoryRecord):
            raise HistoryDataError("record must be ProblemHistoryRecord")
        now = _utc_now()
        record_payload = record.to_dict()
        metadata_json = _canonical_json(record_payload["metadata"])
        with self.transaction() as con:
            self._insert_or_verify(
                con,
                table="problems",
                key_column="problem_data_hash",
                columns=("problem_data_hash", "problem_structural_hash", "problem_class", "n_variables", "n_constraints", "metadata_json", "created_at_utc"),
                values=(record.problem_data_hash, record.problem_structural_hash, record.problem_class, record.n_variables, record.n_constraints, metadata_json, now),
                compare_columns=("problem_data_hash", "problem_structural_hash", "problem_class", "n_variables", "n_constraints", "metadata_json"),
            )
            if record.source_provenance:
                provenance_json = _canonical_json(record_payload["source_provenance"])
                source_id = _sha256_text(provenance_json)
                self._insert_or_verify(
                    con,
                    table="problem_sources",
                    key_column="source_id",
                    columns=("source_id", "problem_data_hash", "provenance_json", "created_at_utc"),
                    values=(source_id, record.problem_data_hash, provenance_json, now),
                    compare_columns=("source_id", "problem_data_hash", "provenance_json"),
                )

    def record_problem_object(
        self,
        problem: LinearProblem | QuadraticProblem,
        *,
        provenance: SourceProvenance | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ProblemHistoryRecord:
        fingerprint = inspect_problem(problem)
        record = ProblemHistoryRecord(
            problem_data_hash=problem.data_hash,
            problem_structural_hash=problem.structural_hash,
            problem_class=fingerprint.problem_class,
            n_variables=fingerprint.n_variables,
            n_constraints=fingerprint.n_constraints,
            source_provenance={} if provenance is None else provenance.to_dict(),
            metadata=dict(metadata or {}),
        )
        self.record_problem(record)
        return record

    def record_feature(self, record: FeatureRecord, *, problem_data_hash: str | None = None) -> None:
        if not isinstance(record, FeatureRecord):
            raise HistoryDataError("record must be FeatureRecord")
        problem_hash = _require_sha(
            record.source_content_sha256 if problem_data_hash is None else problem_data_hash,
            "problem_data_hash",
        )
        values_json = _canonical_json(dict(record.values))
        reasons_json = _canonical_json(dict(record.missingness_reasons))
        with self.transaction() as con:
            if con.execute("SELECT 1 FROM problems WHERE problem_data_hash=?", (problem_hash,)).fetchone() is None:
                raise HistoryDataError("feature record references an unknown problem_data_hash")
            self._insert_or_verify(
                con,
                table="feature_records",
                key_column="record_id",
                columns=(
                    "record_id", "problem_data_hash", "instance_id", "feature_schema_id",
                    "feature_schema_version", "values_json", "missingness_reasons_json",
                    "source_content_sha256", "extractor_id", "extractor_version", "created_at_utc",
                ),
                values=(
                    record.record_id, problem_hash, record.instance_id, record.feature_schema_id,
                    record.feature_schema_version, values_json, reasons_json,
                    record.source_content_sha256, record.extractor_id, record.extractor_version, _utc_now(),
                ),
                compare_columns=(
                    "record_id", "problem_data_hash", "instance_id", "feature_schema_id",
                    "feature_schema_version", "values_json", "missingness_reasons_json",
                    "source_content_sha256", "extractor_id", "extractor_version",
                ),
            )

    def record_solve(self, record: SolveHistoryRecord) -> None:
        if not isinstance(record, SolveHistoryRecord):
            raise HistoryDataError("record must be SolveHistoryRecord")
        metadata_json = _canonical_json(record.to_dict()["metadata"])
        with self.transaction() as con:
            if con.execute("SELECT 1 FROM problems WHERE problem_data_hash=?", (record.problem_data_hash,)).fetchone() is None:
                raise HistoryDataError("solve record references an unknown problem_data_hash")
            self._insert_or_verify(
                con,
                table="solve_runs",
                key_column="run_id",
                columns=(
                    "run_id", "problem_data_hash", "problem_structural_hash", "problem_class",
                    "backend", "backend_version", "public_status", "backend_status", "objective",
                    "validation_valid", "independently_verified_optimal", "total_s", "environment_id",
                    "trace_created_at_utc", "metadata_json", "created_at_utc",
                ),
                values=(
                    record.run_id, record.problem_data_hash, record.problem_structural_hash, record.problem_class,
                    record.backend, record.backend_version, record.public_status, record.backend_status, record.objective,
                    _bool_db(record.validation_valid), int(record.independently_verified_optimal), record.total_s,
                    record.environment_id, record.trace_created_at_utc, metadata_json, _utc_now(),
                ),
                compare_columns=(
                    "run_id", "problem_data_hash", "problem_structural_hash", "problem_class",
                    "backend", "backend_version", "public_status", "backend_status", "objective",
                    "validation_valid", "independently_verified_optimal", "total_s", "environment_id",
                    "trace_created_at_utc", "metadata_json",
                ),
            )

    def record_solve_result(
        self,
        result: SolveResult,
        *,
        run_id: str | None = None,
        environment_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SolveHistoryRecord:
        if not isinstance(result, SolveResult):
            raise HistoryDataError("result must be SolveResult")
        trace = result.trace
        record = SolveHistoryRecord(
            run_id=_run_id_from_result(result) if run_id is None else run_id,
            problem_data_hash=trace.problem_data_hash,
            problem_structural_hash=trace.problem_structural_hash,
            problem_class=trace.problem_class,
            backend=trace.backend,
            backend_version=trace.backend_version,
            public_status=result.status.value,
            backend_status=result.backend_status,
            objective=result.objective,
            validation_valid=None if result.validation is None else result.validation.valid,
            independently_verified_optimal=result.optimality_evidence.independently_verified_optimal,
            total_s=trace.timings.total_s,
            environment_id=environment_id,
            trace_created_at_utc=trace.created_at_utc,
            metadata=dict(metadata or {}),
        )
        self.record_solve(record)
        return record

    def record_benchmark_row(self, row: Mapping[str, Any]) -> None:
        if not isinstance(row, Mapping):
            raise HistoryDataError("benchmark row must be a mapping")
        required = ("run_id", "protocol_id", "instance", "backend", "repetition", "state")
        missing = [name for name in required if name not in row]
        if missing:
            raise HistoryDataError(f"benchmark row missing fields: {missing}")
        identity = {name: str(row[name]).strip() for name in ("run_id", "protocol_id", "instance", "backend", "state")}
        if not all(identity.values()):
            raise HistoryDataError("benchmark identity fields must be non-empty")
        repetition = row["repetition"]
        if type(repetition) is not int or repetition < 0:
            raise HistoryDataError("benchmark repetition must be a non-negative integer")
        for numeric in ("objective", "wall_s", "parse_s", "worker_parse_s", "worker_solve_wall_s"):
            value = row.get(numeric)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(float(value)) or (numeric.endswith("_s") and float(value) < 0)):
                raise HistoryDataError(f"benchmark {numeric} must be finite" + (" and non-negative" if numeric.endswith("_s") else ""))
        validated = row.get("validated")
        if validated is not None and type(validated) is not bool:
            raise HistoryDataError("benchmark validated must be boolean or None")
        allowed_fields = {
            "schema_version", "protocol_id", "environment_id", "thread_policy", "run_id", "complete",
            "instance", "filename", "instance_sha256", "problem_data_hash", "backend", "repetition",
            "parse_s", "state", "wall_s", "public_status", "objective", "validated",
            "reference_status", "reference_objective", "reference_check", "worker_parse_s",
            "worker_solve_wall_s", "error_type", "error",
        }
        compact = {key: row[key] for key in allowed_fields if key in row}
        row_json = _canonical_json(compact)
        values = (
            identity["run_id"], identity["protocol_id"],
            None if row.get("environment_id") is None else str(row.get("environment_id")),
            None if row.get("problem_data_hash") is None else _require_sha(str(row.get("problem_data_hash")), "problem_data_hash"),
            identity["instance"], None if row.get("instance_sha256") is None else _require_sha(str(row.get("instance_sha256")), "instance_sha256"),
            identity["backend"], repetition, identity["state"],
            None if row.get("public_status") is None else str(row.get("public_status")),
            _bool_db(validated), None if row.get("objective") is None else float(row["objective"]),
            None if row.get("wall_s") is None else float(row["wall_s"]),
            None if row.get("reference_check") is None else str(row.get("reference_check")),
            row_json, _utc_now(),
        )
        columns = (
            "run_id", "protocol_id", "environment_id", "problem_data_hash", "instance_name",
            "instance_sha256", "backend", "repetition", "state", "public_status", "validated",
            "objective", "wall_s", "reference_check", "row_json", "created_at_utc",
        )
        with self.transaction() as con:
            self._insert_or_verify(
                con,
                table="benchmark_rows",
                key_column="run_id",
                columns=columns,
                values=values,
                compare_columns=columns[:-1],
            )

    def record_benchmark_rows(self, rows: Iterable[Mapping[str, Any]]) -> int:
        # Each row is atomic. This intentionally does not make the entire iterable one giant transaction.
        count = 0
        for row in rows:
            self.record_benchmark_row(row)
            count += 1
        return count

    def record_artifact(
        self,
        *,
        artifact_id: str,
        artifact_type: str,
        run_id: str | None = None,
        path: str | None = None,
        size_bytes: int = 0,
        sha256: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        artifact_id = str(artifact_id).strip()
        artifact_type = str(artifact_type).strip()
        if not artifact_id or not artifact_type:
            raise HistoryDataError("artifact_id and artifact_type must be non-empty")
        if type(size_bytes) is not int or size_bytes < 0:
            raise HistoryDataError("size_bytes must be a non-negative integer")
        digest = None if sha256 is None else _require_sha(sha256, "artifact sha256")
        metadata_json = _canonical_json(dict(metadata or {}))
        columns = ("artifact_id", "run_id", "artifact_type", "path", "size_bytes", "sha256", "metadata_json", "created_at_utc")
        values = (artifact_id, None if run_id is None else str(run_id), artifact_type, None if path is None else str(path), size_bytes, digest, metadata_json, _utc_now())
        with self.transaction() as con:
            self._insert_or_verify(
                con,
                table="artifacts",
                key_column="artifact_id",
                columns=columns,
                values=values,
                compare_columns=columns[:-1],
            )

    def problem(self, problem_data_hash: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute("SELECT * FROM problems WHERE problem_data_hash=?", (problem_data_hash,)).fetchone()
            sources = self._connection.execute("SELECT source_id,provenance_json,created_at_utc FROM problem_sources WHERE problem_data_hash=? ORDER BY source_id", (problem_data_hash,)).fetchall()
        if row is None:
            return None
        out = dict(row)
        out["metadata"] = _json_load(out.pop("metadata_json"))
        out["sources"] = [
            {"source_id": item["source_id"], "provenance": _json_load(item["provenance_json"]), "created_at_utc": item["created_at_utc"]}
            for item in sources
        ]
        return out

    def solve_runs(self, *, problem_data_hash: str | None = None, backend: str | None = None) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if problem_data_hash is not None:
            clauses.append("problem_data_hash=?"); params.append(problem_data_hash)
        if backend is not None:
            clauses.append("backend=?"); params.append(backend)
        sql = "SELECT * FROM solve_runs" + (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY created_at_utc, run_id"
        with self._lock:
            rows = self._connection.execute(sql, tuple(params)).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            item["validation_valid"] = None if item["validation_valid"] is None else bool(item["validation_valid"])
            item["independently_verified_optimal"] = bool(item["independently_verified_optimal"])
            item["metadata"] = _json_load(item.pop("metadata_json"))
            output.append(item)
        return output

    def benchmark_rows(self, *, protocol_id: str | None = None, backend: str | None = None) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if protocol_id is not None:
            clauses.append("protocol_id=?"); params.append(protocol_id)
        if backend is not None:
            clauses.append("backend=?"); params.append(backend)
        sql = "SELECT row_json FROM benchmark_rows" + (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY run_id"
        with self._lock:
            rows = self._connection.execute(sql, tuple(params)).fetchall()
        return [_json_load(row["row_json"]) for row in rows]

    def feature_rows(self, *, problem_data_hash: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM feature_records"
        params: tuple[Any, ...] = ()
        if problem_data_hash is not None:
            sql += " WHERE problem_data_hash=?"; params = (problem_data_hash,)
        sql += " ORDER BY created_at_utc, record_id"
        with self._lock:
            rows = self._connection.execute(sql, params).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["values"] = _json_load(item.pop("values_json"))
            item["missingness_reasons"] = _json_load(item.pop("missingness_reasons_json"))
            output.append(item)
        return output

    def solver_performance_summary(self, *, backend: str | None = None) -> list[dict[str, Any]]:
        where = " WHERE backend=?" if backend is not None else ""
        params: tuple[Any, ...] = (backend,) if backend is not None else ()
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT backend,
                       COUNT(*) AS runs,
                       SUM(CASE WHEN validation_valid=1 THEN 1 ELSE 0 END) AS validated_runs,
                       SUM(CASE WHEN independently_verified_optimal=1 THEN 1 ELSE 0 END) AS independently_verified_optimal_runs,
                       AVG(total_s) AS average_total_s
                FROM solve_runs{where}
                GROUP BY backend ORDER BY backend
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]
