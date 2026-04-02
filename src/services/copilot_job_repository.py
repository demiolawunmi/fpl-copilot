from __future__ import annotations

from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path
from typing import Any


_JOB_TABLE = "copilot_jobs"
_VALID_STATUSES = {"queued", "running", "completed", "failed"}
_ALLOWED_TRANSITIONS = {
    "queued": {"running", "failed"},
    "running": {"completed", "failed"},
    "completed": set(),
    "failed": set(),
}

_SENSITIVE_KEY_EXACT = {
    "apikey",
    "accesstoken",
    "refreshtoken",
    "authorization",
    "password",
    "secret",
    "clientsecret",
    "xairsenalrunkey",
}
_SENSITIVE_KEY_PARTIAL = ("password", "secret", "token", "apikey", "api_key", "authorization")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    normalized = "".join(ch for ch in lowered if ch.isalnum())
    if normalized in _SENSITIVE_KEY_EXACT:
        return True
    return any(part in lowered for part in _SENSITIVE_KEY_PARTIAL)


def redact_sensitive_fields(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if _is_sensitive_key(str(key)):
                redacted[key] = "[REDACTED]"
                continue
            redacted[key] = redact_sensitive_fields(item)
        return redacted
    if isinstance(value, list):
        return [redact_sensitive_fields(item) for item in value]
    return value


class CopilotJobRepository:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def _initialize_schema(self) -> None:
        with self._connect() as con:
            con.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_JOB_TABLE} (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    type TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    result_json TEXT,
                    error_json TEXT,
                    input_hash TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    correlation_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            con.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{_JOB_TABLE}_status ON {_JOB_TABLE}(status);"
            )
            con.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{_JOB_TABLE}_input_hash ON {_JOB_TABLE}(input_hash);"
            )

            existing_columns = {
                row["name"]
                for row in con.execute(f"PRAGMA table_info({_JOB_TABLE});").fetchall()
            }
            required_columns = {
                "job_id": "TEXT",
                "status": "TEXT",
                "type": "TEXT",
                "input_json": "TEXT",
                "result_json": "TEXT",
                "error_json": "TEXT",
                "input_hash": "TEXT",
                "schema_version": "TEXT",
                "correlation_id": "TEXT",
                "created_at": "TEXT",
                "updated_at": "TEXT",
            }
            for column_name, column_type in required_columns.items():
                if column_name in existing_columns:
                    continue
                con.execute(
                    f"ALTER TABLE {_JOB_TABLE} ADD COLUMN {column_name} {column_type};"
                )

    def create_job(
        self,
        *,
        job_id: str,
        status: str,
        job_type: str,
        input_payload: dict[str, Any],
        input_hash: str,
        schema_version: str,
        correlation_id: str,
    ) -> None:
        if status not in _VALID_STATUSES:
            raise ValueError(f"Unsupported status: {status}")

        created_at = _utc_now_iso()
        with self._connect() as con:
            con.execute(
                f"""
                INSERT INTO {_JOB_TABLE} (
                    job_id,
                    status,
                    type,
                    input_json,
                    result_json,
                    error_json,
                    input_hash,
                    schema_version,
                    correlation_id,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    job_id,
                    status,
                    job_type,
                    json.dumps(redact_sensitive_fields(input_payload), separators=(",", ":"), sort_keys=True),
                    None,
                    None,
                    input_hash,
                    schema_version,
                    correlation_id,
                    created_at,
                    created_at,
                ),
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as con:
            row = con.execute(
                f"SELECT * FROM {_JOB_TABLE} WHERE job_id = ?;",
                (job_id,),
            ).fetchone()
        if row is None:
            return None

        return {
            "job_id": row["job_id"],
            "status": row["status"],
            "type": row["type"],
            "input_json": json.loads(row["input_json"]) if row["input_json"] else None,
            "result_json": json.loads(row["result_json"]) if row["result_json"] else None,
            "error_json": json.loads(row["error_json"]) if row["error_json"] else None,
            "input_hash": row["input_hash"],
            "schema_version": row["schema_version"],
            "correlation_id": row["correlation_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def claim_next_queued_job(self) -> dict[str, Any] | None:
        updated_at = _utc_now_iso()
        with self._connect() as con:
            row = con.execute(
                f"""
                SELECT *
                FROM {_JOB_TABLE}
                WHERE status = 'queued'
                ORDER BY created_at ASC, job_id ASC
                LIMIT 1;
                """
            ).fetchone()
            if row is None:
                return None

            con.execute(
                f"""
                UPDATE {_JOB_TABLE}
                SET status = 'running', updated_at = ?
                WHERE job_id = ?;
                """,
                (updated_at, row["job_id"]),
            )

            refreshed = con.execute(
                f"SELECT * FROM {_JOB_TABLE} WHERE job_id = ?;",
                (row["job_id"],),
            ).fetchone()

        if refreshed is None:
            return None

        return {
            "job_id": refreshed["job_id"],
            "status": refreshed["status"],
            "type": refreshed["type"],
            "input_json": json.loads(refreshed["input_json"]) if refreshed["input_json"] else None,
            "result_json": json.loads(refreshed["result_json"]) if refreshed["result_json"] else None,
            "error_json": json.loads(refreshed["error_json"]) if refreshed["error_json"] else None,
            "input_hash": refreshed["input_hash"],
            "schema_version": refreshed["schema_version"],
            "correlation_id": refreshed["correlation_id"],
            "created_at": refreshed["created_at"],
            "updated_at": refreshed["updated_at"],
        }

    def update_job_status(
        self,
        *,
        job_id: str,
        new_status: str,
        result_payload: dict[str, Any] | None = None,
        error_payload: dict[str, Any] | None = None,
    ) -> bool:
        if new_status not in _VALID_STATUSES:
            raise ValueError(f"Unsupported status: {new_status}")

        with self._connect() as con:
            row = con.execute(
                f"SELECT status FROM {_JOB_TABLE} WHERE job_id = ?;",
                (job_id,),
            ).fetchone()
            if row is None:
                return False

            current_status = row["status"]
            if current_status != new_status and new_status not in _ALLOWED_TRANSITIONS.get(current_status, set()):
                raise ValueError(f"Invalid status transition: {current_status} -> {new_status}")

            if new_status == "completed" and result_payload is None:
                raise ValueError("Completed jobs must persist result_payload")
            if new_status == "failed" and error_payload is None:
                raise ValueError("Failed jobs must persist error_payload")
            if new_status == "completed" and error_payload is not None:
                raise ValueError("Completed jobs cannot persist error_payload")
            if new_status == "failed" and result_payload is not None:
                raise ValueError("Failed jobs cannot persist result_payload")

            updated_at = _utc_now_iso()
            result_json = (
                json.dumps(redact_sensitive_fields(result_payload), separators=(",", ":"), sort_keys=True)
                if result_payload is not None
                else None
            )
            error_json = (
                json.dumps(redact_sensitive_fields(error_payload), separators=(",", ":"), sort_keys=True)
                if error_payload is not None
                else None
            )

            con.execute(
                f"""
                UPDATE {_JOB_TABLE}
                SET
                    status = ?,
                    result_json = ?,
                    error_json = ?,
                    updated_at = ?
                WHERE job_id = ?;
                """,
                (new_status, result_json, error_json, updated_at, job_id),
            )
            return True

    def delete_job(self, job_id: str) -> bool:
        with self._connect() as con:
            cursor = con.execute(
                f"DELETE FROM {_JOB_TABLE} WHERE job_id = ?;",
                (job_id,),
            )
            return cursor.rowcount > 0

    def cleanup_expired_jobs(self, retention_days: int = 30) -> int:
        from datetime import timedelta

        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
        with self._connect() as con:
            cursor = con.execute(
                f"DELETE FROM {_JOB_TABLE} WHERE created_at < ?;",
                (cutoff,),
            )
            return cursor.rowcount

    def find_jobs_by_input_hash(self, input_hash: str) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                f"SELECT * FROM {_JOB_TABLE} WHERE input_hash = ? ORDER BY created_at ASC;",
                (input_hash,),
            ).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "job_id": row["job_id"],
                    "status": row["status"],
                    "type": row["type"],
                    "input_json": json.loads(row["input_json"]) if row["input_json"] else None,
                    "result_json": json.loads(row["result_json"]) if row["result_json"] else None,
                    "error_json": json.loads(row["error_json"]) if row["error_json"] else None,
                    "input_hash": row["input_hash"],
                    "schema_version": row["schema_version"],
                    "correlation_id": row["correlation_id"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            )
        return results
