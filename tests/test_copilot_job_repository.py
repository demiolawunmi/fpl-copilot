from pathlib import Path
import sqlite3
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.services.copilot_job_repository import CopilotJobRepository


def test_create_and_get_job_persists_required_fields(tmp_path: Path) -> None:
    repo = CopilotJobRepository(tmp_path / "jobs.db")

    repo.create_job(
        job_id="job-1",
        status="queued",
        job_type="hybrid",
        input_payload={"schema_version": "1.0", "weights": {"fplcopilot": 0.6, "airsenal": 0.4}},
        input_hash="hash-1",
        schema_version="1.0",
        correlation_id="corr-1",
    )

    row = repo.get_job("job-1")

    assert row is not None
    assert row["job_id"] == "job-1"
    assert row["status"] == "queued"
    assert row["type"] == "hybrid"
    assert row["input_json"]["weights"]["fplcopilot"] == 0.6
    assert row["result_json"] is None
    assert row["error_json"] is None
    assert row["input_hash"] == "hash-1"
    assert row["schema_version"] == "1.0"
    assert row["correlation_id"] == "corr-1"
    assert row["created_at"]
    assert row["updated_at"]


def test_status_transitions_queued_running_completed(tmp_path: Path) -> None:
    repo = CopilotJobRepository(tmp_path / "jobs.db")
    repo.create_job(
        job_id="job-2",
        status="queued",
        job_type="hybrid",
        input_payload={"prompt": "blend"},
        input_hash="hash-2",
        schema_version="1.0",
        correlation_id="corr-2",
    )

    assert repo.update_job_status(job_id="job-2", new_status="running")
    running_row = repo.get_job("job-2")
    assert running_row is not None
    assert running_row["status"] == "running"
    assert running_row["input_json"] == {"prompt": "blend"}
    assert running_row["result_json"] is None
    assert running_row["error_json"] is None

    result_payload = {
        "schema_version": "1.0",
        "core": {"summary": "done", "confidence": 0.8},
    }
    assert repo.update_job_status(
        job_id="job-2",
        new_status="completed",
        result_payload=result_payload,
    )

    row = repo.get_job("job-2")
    assert row is not None
    assert row["status"] == "completed"
    assert row["result_json"] == result_payload
    assert row["error_json"] is None
    assert row["input_json"] == {"prompt": "blend"}


def test_failed_status_stores_error_json(tmp_path: Path) -> None:
    repo = CopilotJobRepository(tmp_path / "jobs.db")
    repo.create_job(
        job_id="job-3",
        status="queued",
        job_type="hybrid",
        input_payload={"prompt": "blend"},
        input_hash="hash-3",
        schema_version="1.0",
        correlation_id="corr-3",
    )

    assert repo.update_job_status(job_id="job-3", new_status="running")
    assert repo.update_job_status(
        job_id="job-3",
        new_status="failed",
        error_payload={"code": "LLM_TIMEOUT", "message": "provider timed out"},
    )

    row = repo.get_job("job-3")
    assert row is not None
    assert row["status"] == "failed"
    assert row["result_json"] is None
    assert row["error_json"] == {"code": "LLM_TIMEOUT", "message": "provider timed out"}
    assert row["input_json"] == {"prompt": "blend"}


def test_redaction_is_applied_before_persistence(tmp_path: Path) -> None:
    repo = CopilotJobRepository(tmp_path / "jobs.db")
    repo.create_job(
        job_id="job-4",
        status="queued",
        job_type="hybrid",
        input_payload={
            "api_key": "very-secret",
            "nested": {"access_token": "token-value"},
            "safe": "ok",
        },
        input_hash="hash-4",
        schema_version="1.0",
        correlation_id="corr-4",
    )

    repo.update_job_status(job_id="job-4", new_status="running")
    repo.update_job_status(
        job_id="job-4",
        new_status="failed",
        error_payload={"clientSecret": "abc123", "message": "bad response"},
    )

    row = repo.get_job("job-4")
    assert row is not None
    assert row["input_json"] == {
        "api_key": "[REDACTED]",
        "nested": {"access_token": "[REDACTED]"},
        "safe": "ok",
    }
    assert row["error_json"] == {"clientSecret": "[REDACTED]", "message": "bad response"}

    con = sqlite3.connect(tmp_path / "jobs.db")
    persisted = con.execute(
        "SELECT input_json, error_json FROM copilot_jobs WHERE job_id = ?;",
        ("job-4",),
    ).fetchone()
    con.close()

    assert persisted is not None
    assert "very-secret" not in persisted[0]
    assert "token-value" not in persisted[0]
    assert "abc123" not in persisted[1]


def test_terminal_state_payload_requirements_enforced(tmp_path: Path) -> None:
    repo = CopilotJobRepository(tmp_path / "jobs.db")
    repo.create_job(
        job_id="job-5",
        status="queued",
        job_type="hybrid",
        input_payload={"prompt": "blend"},
        input_hash="hash-5",
        schema_version="1.0",
        correlation_id="corr-5",
    )

    assert repo.update_job_status(job_id="job-5", new_status="running")

    try:
        repo.update_job_status(job_id="job-5", new_status="completed")
        assert False, "completed transition without result_payload should fail"
    except ValueError as exc:
        assert "result_payload" in str(exc)

    try:
        repo.update_job_status(job_id="job-5", new_status="failed")
        assert False, "failed transition without error_payload should fail"
    except ValueError as exc:
        assert "error_payload" in str(exc)

    try:
        repo.update_job_status(
            job_id="job-5",
            new_status="completed",
            result_payload={"ok": True},
            error_payload={"code": "BAD"},
        )
        assert False, "completed transition with error_payload should fail"
    except ValueError as exc:
        assert "cannot persist error_payload" in str(exc)

    try:
        repo.update_job_status(
            job_id="job-5",
            new_status="failed",
            result_payload={"ok": True},
            error_payload={"code": "BAD"},
        )
        assert False, "failed transition with result_payload should fail"
    except ValueError as exc:
        assert "cannot persist result_payload" in str(exc)
