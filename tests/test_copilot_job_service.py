from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.services.copilot_job_repository import CopilotJobRepository
from src.services.copilot_job_service import CopilotJobService


class _AssemblerOk:
    def assemble_model_context(self, *, source_weights, player_name_contains=None, limit=25):
        return {
            "schema_version": "1.0",
            "weights": dict(source_weights),
            "player_name_contains": player_name_contains,
            "limit": limit,
            "blended_players": [
                {"player_id": 1, "player_name": "Saka", "blended_projected_points": 9.2}
            ],
        }


class _AdapterOk:
    def generate_hybrid_payload(self, *, schema_version, correlation_id, model_context):
        assert model_context["blended_players"][0]["player_id"] == 1
        return {
            "schema_version": schema_version,
            "correlation_id": correlation_id,
            "core": {"summary": "Ready", "confidence": 0.81},
            "recommended_transfers": [],
            "ask_copilot": {
                "answer": "Hold transfer.",
                "rationale": ["Small edge"],
                "confidence": 0.7,
            },
            "degraded_mode": {
                "is_degraded": False,
                "code": None,
                "message": None,
                "fallback_used": False,
            },
        }


class _AdapterFails:
    def generate_hybrid_payload(self, *, schema_version, correlation_id, model_context):
        raise RuntimeError("Gemini provider unavailable")


def _service(tmp_path: Path, adapter) -> CopilotJobService:
    repo = CopilotJobRepository(tmp_path / "jobs.db")
    return CopilotJobService(repository=repo, assembler=_AssemblerOk(), gemini_adapter=adapter)


def test_happy_path_submit_execute_and_complete(tmp_path: Path) -> None:
    service = _service(tmp_path, _AdapterOk())

    accepted = service.submit_job(
        {
            "schema_version": "1.0",
            "correlation_id": "corr-ok",
            "task": "hybrid",
            "source_weights": {"fplcopilot": 0.6, "airsenal": 0.4},
        }
    )
    assert accepted["status"] == "queued"

    finished = service.execute_next_queued_job()
    assert finished is not None
    assert finished["job_id"] == accepted["job_id"]
    assert finished["status"] == "completed"
    assert finished["result_json"]["schema_version"] == "1.0"
    assert finished["result_json"]["correlation_id"] == "corr-ok"
    assert finished["error_json"] is None


def test_provider_failure_marks_job_failed_with_structured_error(tmp_path: Path) -> None:
    service = _service(tmp_path, _AdapterFails())

    accepted = service.submit_job(
        {
            "schema_version": "1.0",
            "correlation_id": "corr-fail",
            "task": "hybrid",
            "source_weights": {"fplcopilot": 0.6, "airsenal": 0.4},
        }
    )

    finished = service.execute_next_queued_job()
    assert finished is not None
    assert finished["job_id"] == accepted["job_id"]
    assert finished["status"] == "failed"
    assert finished["result_json"] is None
    assert finished["error_json"]["schema_version"] == "1.0"
    assert finished["error_json"]["correlation_id"] == "corr-fail"
    assert finished["error_json"]["error"]["code"] == "JOB_FAILED"
    assert "provider unavailable" in finished["error_json"]["error"]["message"].lower()


def test_poll_status_reads_current_state_and_missing_job(tmp_path: Path) -> None:
    service = _service(tmp_path, _AdapterOk())

    accepted = service.submit_job(
        {
            "schema_version": "1.0",
            "correlation_id": "corr-poll",
            "task": "hybrid",
            "source_weights": {"fplcopilot": 0.7, "airsenal": 0.3},
        }
    )

    queued = service.get_job_status(accepted["job_id"])
    assert queued is not None
    assert queued["status"] == "queued"
    assert queued["result_json"] is None
    assert queued["error_json"] is None

    service.execute_next_queued_job()
    completed = service.get_job_status(accepted["job_id"])
    assert completed is not None
    assert completed["status"] == "completed"

    assert service.get_job_status("job-does-not-exist") is None
