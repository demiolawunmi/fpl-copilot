from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.services.copilot_job_repository import CopilotJobRepository
from src.services.copilot_job_service import CopilotJobService


def _repo(tmp_path: Path) -> CopilotJobRepository:
    return CopilotJobRepository(tmp_path / "jobs.db")


def _service(tmp_path: Path) -> CopilotJobService:
    repo = CopilotJobRepository(tmp_path / "jobs.db")
    return CopilotJobService(repository=repo, assembler=_AssemblerOk(), gemini_adapter=_AdapterOk())


class _AssemblerOk:
    def assemble_model_context(
        self,
        *,
        source_weights,
        player_name_contains=None,
        gameweek=None,
        bank=None,
        free_transfers=None,
        current_squad=None,
        fpl_team_id=None,
    ):
        return {
            "schema_version": "1.0",
            "weights": dict(source_weights),
            "sources": ["elo", "airsenal"],
            "blended_players": [{"player_id": 1, "player_name": "Saka", "team": "ARS", "position": "MID", "elo_score": 1650.0, "airsenal_predicted_points": 8.0}],
        }


class _AdapterOk:
    def generate_hybrid_payload(self, *, schema_version, correlation_id, model_context):
        return {
            "schema_version": schema_version,
            "correlation_id": correlation_id,
            "core": {"summary": "Ready", "confidence": 0.81},
            "recommended_transfers": [],
            "ask_copilot": {"answer": "ok", "rationale": [], "confidence": 0.7},
            "degraded_mode": {
                "is_degraded": False,
                "code": None,
                "message": None,
                "fallback_used": False,
            },
        }


def test_cleanup_expired_jobs_removes_only_expired(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    expired_ts = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
    recent_ts = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()

    repo.create_job(
        job_id="job-expired",
        status="completed",
        job_type="hybrid",
        input_payload={"prompt": "old"},
        input_hash="hash-expired",
        schema_version="1.0",
        correlation_id="corr-expired",
    )
    repo.create_job(
        job_id="job-recent",
        status="queued",
        job_type="hybrid",
        input_payload={"prompt": "new"},
        input_hash="hash-recent",
        schema_version="1.0",
        correlation_id="corr-recent",
    )

    con = sqlite3.connect(tmp_path / "jobs.db")
    con.execute(
        "UPDATE copilot_jobs SET created_at = ? WHERE job_id = ?;",
        (expired_ts, "job-expired"),
    )
    con.execute(
        "UPDATE copilot_jobs SET created_at = ? WHERE job_id = ?;",
        (recent_ts, "job-recent"),
    )
    con.commit()
    con.close()

    deleted = repo.cleanup_expired_jobs(retention_days=30)
    assert deleted == 1

    assert repo.get_job("job-expired") is None
    assert repo.get_job("job-recent") is not None


def test_cleanup_expired_boundary_exactly_at_cutoff(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    just_inside_ts = (datetime.now(timezone.utc) - timedelta(days=29, hours=23)).isoformat()

    repo.create_job(
        job_id="job-cutoff",
        status="completed",
        job_type="hybrid",
        input_payload={"prompt": "cutoff"},
        input_hash="hash-cutoff",
        schema_version="1.0",
        correlation_id="corr-cutoff",
    )
    repo.create_job(
        job_id="job-inside",
        status="queued",
        job_type="hybrid",
        input_payload={"prompt": "inside"},
        input_hash="hash-inside",
        schema_version="1.0",
        correlation_id="corr-inside",
    )

    con = sqlite3.connect(tmp_path / "jobs.db")
    con.execute(
        "UPDATE copilot_jobs SET created_at = ? WHERE job_id = ?;",
        (cutoff_ts, "job-cutoff"),
    )
    con.execute(
        "UPDATE copilot_jobs SET created_at = ? WHERE job_id = ?;",
        (just_inside_ts, "job-inside"),
    )
    con.commit()
    con.close()

    deleted = repo.cleanup_expired_jobs(retention_days=30)
    assert deleted == 1
    assert repo.get_job("job-cutoff") is None
    assert repo.get_job("job-inside") is not None


def test_cleanup_expired_default_retention(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    repo.create_job(
        job_id="job-old",
        status="completed",
        job_type="hybrid",
        input_payload={"prompt": "old"},
        input_hash="hash-old",
        schema_version="1.0",
        correlation_id="corr-old",
    )

    old_ts = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
    con = sqlite3.connect(tmp_path / "jobs.db")
    con.execute(
        "UPDATE copilot_jobs SET created_at = ? WHERE job_id = ?;",
        (old_ts, "job-old"),
    )
    con.commit()
    con.close()

    deleted = repo.cleanup_expired_jobs()
    assert deleted == 1
    assert repo.get_job("job-old") is None


def test_replay_queued_returns_existing_job_id(tmp_path: Path) -> None:
    service = _service(tmp_path)

    payload = {
        "schema_version": "1.0",
        "correlation_id": "corr-replay",
        "task": "hybrid",
        "source_weights": {"elo": 0.6, "airsenal": 0.4},
    }

    first = service.submit_job(payload)
    assert first["status"] == "queued"

    second = service.submit_job(payload)
    assert second["job_id"] == first["job_id"]
    assert second["status"] == "queued"
    assert "result_json" not in second


def test_replay_running_returns_existing_job_id(tmp_path: Path) -> None:
    service = _service(tmp_path)

    payload = {
        "schema_version": "1.0",
        "correlation_id": "corr-running",
        "task": "hybrid",
        "source_weights": {"elo": 0.5, "airsenal": 0.5},
    }

    first = service.submit_job(payload)
    service.repository.update_job_status(job_id=first["job_id"], new_status="running")

    second = service.submit_job(payload)
    assert second["job_id"] == first["job_id"]
    assert second["status"] == "running"


def test_replay_completed_returns_cached_result(tmp_path: Path) -> None:
    service = _service(tmp_path)

    payload = {
        "schema_version": "1.0",
        "correlation_id": "corr-completed",
        "task": "hybrid",
        "source_weights": {"elo": 0.7, "airsenal": 0.3},
    }

    first = service.submit_job(payload)
    service.execute_next_queued_job()

    second = service.submit_job(payload)
    assert second["job_id"] == first["job_id"]
    assert second["status"] == "completed"
    assert second["result_json"] is not None
    assert second["result_json"]["schema_version"] == "1.0"


def test_force_refresh_bypasses_completed_cache(tmp_path: Path) -> None:
    service = _service(tmp_path)

    payload = {
        "schema_version": "1.0",
        "correlation_id": "corr-force",
        "task": "hybrid",
        "source_weights": {"elo": 0.8, "airsenal": 0.2},
    }

    first = service.submit_job(payload)
    service.execute_next_queued_job()

    second = service.submit_job(payload, force_refresh=True)
    assert second["job_id"] != first["job_id"]
    assert second["status"] == "queued"


def test_force_refresh_bypasses_queued_replay(tmp_path: Path) -> None:
    service = _service(tmp_path)

    payload = {
        "schema_version": "1.0",
        "correlation_id": "corr-force-queued",
        "task": "hybrid",
        "source_weights": {"elo": 0.5, "airsenal": 0.5},
    }

    first = service.submit_job(payload)
    second = service.submit_job(payload, force_refresh=True)
    assert second["job_id"] != first["job_id"]
    assert second["status"] == "queued"


def test_redaction_on_input_and_result(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    repo.create_job(
        job_id="job-redact-input",
        status="queued",
        job_type="hybrid",
        input_payload={"api_key": "secret-123", "safe": "ok"},
        input_hash="hash-redact",
        schema_version="1.0",
        correlation_id="corr-redact",
    )

    row = repo.get_job("job-redact-input")
    assert row is not None
    assert row["input_json"]["api_key"] == "[REDACTED]"
    assert row["input_json"]["safe"] == "ok"

    repo.update_job_status(job_id="job-redact-input", new_status="running")
    repo.update_job_status(
        job_id="job-redact-input",
        new_status="completed",
        result_payload={"accessToken": "tok-abc", "data": "value"},
    )

    row = repo.get_job("job-redact-input")
    assert row is not None
    assert row["result_json"]["accessToken"] == "[REDACTED]"
    assert row["result_json"]["data"] == "value"


def test_redaction_nested_sensitive_fields(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    repo.create_job(
        job_id="job-nested",
        status="queued",
        job_type="hybrid",
        input_payload={
            "config": {"password": "hunter2", "safe": True},
            "list": [{"secret": "val"}],
        },
        input_hash="hash-nested",
        schema_version="1.0",
        correlation_id="corr-nested",
    )

    row = repo.get_job("job-nested")
    assert row is not None
    assert row["input_json"]["config"]["password"] == "[REDACTED]"
    assert row["input_json"]["config"]["safe"] is True
    assert row["input_json"]["list"][0]["secret"] == "[REDACTED]"


def test_redaction_on_error_payload(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    repo.create_job(
        job_id="job-error-redact",
        status="queued",
        job_type="hybrid",
        input_payload={"prompt": "test"},
        input_hash="hash-error",
        schema_version="1.0",
        correlation_id="corr-error",
    )

    repo.update_job_status(job_id="job-error-redact", new_status="running")
    repo.update_job_status(
        job_id="job-error-redact",
        new_status="failed",
        error_payload={"clientSecret": "abc", "message": "fail"},
    )

    row = repo.get_job("job-error-redact")
    assert row is not None
    assert row["error_json"]["clientSecret"] == "[REDACTED]"
    assert row["error_json"]["message"] == "fail"
