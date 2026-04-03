from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from unittest.mock import MagicMock, patch

from src.services.copilot_elo_llm_assembler import CopilotEloLlmAssembler
from src.services.copilot_gemini_adapter import CopilotGeminiAdapter
from src.services.copilot_job_repository import CopilotJobRepository
from src.services.copilot_job_service import CopilotJobService


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text


class _ModelsAPI:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = outcomes

    def generate_content(self, *, model, contents, config, timeout):
        del model, contents, config, timeout
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _Response(outcome)


class _Client:
    def __init__(self, outcomes: list[object]) -> None:
        self.models = _ModelsAPI(outcomes=outcomes)


class _AdapterRaises:
    def generate_hybrid_payload(self, *, schema_version, correlation_id, model_context):
        del schema_version, correlation_id, model_context
        raise RuntimeError("provider hard failure")


def _make_model_context():
    return {
        "schema_version": "1.0",
        "weights": {"elo": 0.6, "airsenal": 0.4},
        "sources": ["elo", "airsenal"],
        "blended_players": [
            {"player_id": 101, "player_name": "Saka", "team": "ARS", "position": "MID", "elo_score": 1650.0, "airsenal_predicted_points": 8.0},
            {"player_id": 202, "player_name": "Haaland", "team": "MCI", "position": "FWD", "elo_score": 1850.5, "airsenal_predicted_points": 11.0},
        ],
    }


def _submit(service: CopilotJobService, correlation_id: str) -> str:
    accepted = service.submit_job(
        {
            "schema_version": "1.0",
            "correlation_id": correlation_id,
            "task": "hybrid",
            "source_weights": {"elo": 0.6, "airsenal": 0.4},
        }
    )
    return accepted["job_id"]


def test_happy_path_provider_response_never_uses_degraded_fallback(tmp_path: Path) -> None:
    repo = CopilotJobRepository(tmp_path / "jobs.db")
    assembler = MagicMock(spec=CopilotEloLlmAssembler)
    assembler.assemble_model_context.return_value = _make_model_context()
    adapter = CopilotGeminiAdapter(
        client=_Client(
            outcomes=[
                '{"core":{"summary":"ok","confidence":0.81},"recommended_transfers":[],"ask_copilot":{"answer":"Hold","rationale":["small edge"],"confidence":0.72}}'
            ]
        ),
        sleep_fn=lambda _: None,
    )
    service = CopilotJobService(repository=repo, assembler=assembler, gemini_adapter=adapter)

    job_id = _submit(service, "corr-happy")
    finished = service.execute_next_queued_job()

    assert finished is not None
    assert finished["job_id"] == job_id
    assert finished["status"] == "completed"
    assert finished["result_json"]["degraded_mode"]["is_degraded"] is False
    assert finished["result_json"]["degraded_mode"]["fallback_used"] is False
    assert finished["result_json"]["degraded_mode"]["code"] is None
    assert finished["error_json"] is None


def test_timeout_returns_completed_with_structured_llm_timeout_degraded_payload(tmp_path: Path) -> None:
    repo = CopilotJobRepository(tmp_path / "jobs.db")
    assembler = MagicMock(spec=CopilotEloLlmAssembler)
    assembler.assemble_model_context.return_value = _make_model_context()
    adapter = CopilotGeminiAdapter(
        client=_Client(
            outcomes=[TimeoutError("timeout-1"), TimeoutError("timeout-2"), TimeoutError("timeout-3")]
        ),
        sleep_fn=lambda _: None,
    )
    service = CopilotJobService(repository=repo, assembler=assembler, gemini_adapter=adapter)

    _submit(service, "corr-timeout")
    finished = service.execute_next_queued_job()

    assert finished is not None
    assert finished["status"] == "completed"
    assert finished["result_json"]["degraded_mode"]["is_degraded"] is True
    assert finished["result_json"]["degraded_mode"]["fallback_used"] is True
    assert finished["result_json"]["degraded_mode"]["code"] == "LLM_TIMEOUT"
    assert finished["error_json"] is None


def test_malformed_output_returns_completed_with_schema_validation_failed_degraded_payload(tmp_path: Path) -> None:
    repo = CopilotJobRepository(tmp_path / "jobs.db")
    assembler = MagicMock(spec=CopilotEloLlmAssembler)
    assembler.assemble_model_context.return_value = _make_model_context()
    adapter = CopilotGeminiAdapter(
        client=_Client(outcomes=["not-json", "still-not-json", "[]"]),
        sleep_fn=lambda _: None,
    )
    service = CopilotJobService(repository=repo, assembler=assembler, gemini_adapter=adapter)

    _submit(service, "corr-malformed")
    finished = service.execute_next_queued_job()

    assert finished is not None
    assert finished["status"] == "completed"
    assert finished["result_json"]["degraded_mode"]["is_degraded"] is True
    assert finished["result_json"]["degraded_mode"]["fallback_used"] is True
    assert finished["result_json"]["degraded_mode"]["code"] == "SCHEMA_VALIDATION_FAILED"
    assert finished["error_json"] is None


def test_unhandled_provider_failure_marks_job_failed_with_structured_error_payload(tmp_path: Path) -> None:
    repo = CopilotJobRepository(tmp_path / "jobs.db")
    assembler = MagicMock(spec=CopilotEloLlmAssembler)
    assembler.assemble_model_context.return_value = _make_model_context()
    fallback = MagicMock()
    fallback.build_fallback_payload.side_effect = RuntimeError("Fallback also broken")
    service = CopilotJobService(repository=repo, assembler=assembler, gemini_adapter=_AdapterRaises(), fallback=fallback)

    _submit(service, "corr-failed")
    finished = service.execute_next_queued_job()

    assert finished is not None
    assert finished["status"] == "failed"
    assert finished["result_json"] is None
    assert finished["error_json"]["error"]["code"] == "JOB_FAILED"
    assert "provider hard failure" in finished["error_json"]["error"]["message"]


def test_assembler_failure_cannot_be_recovered_by_fallback(tmp_path: Path) -> None:
    repo = CopilotJobRepository(tmp_path / "jobs.db")
    assembler = MagicMock(spec=CopilotEloLlmAssembler)
    assembler.assemble_model_context.side_effect = RuntimeError("Database connection lost")

    class _StubGeminiAdapter:
        def generate_hybrid_payload(self, *, schema_version, correlation_id, model_context):
            return {"schema_version": schema_version, "correlation_id": correlation_id, "core": {"summary": "ok", "confidence": 0.5}, "recommended_transfers": [], "ask_copilot": {"answer": "ok", "rationale": [], "confidence": 0.5}, "degraded_mode": {"is_degraded": False, "code": None, "message": None, "fallback_used": False}}

    service = CopilotJobService(repository=repo, assembler=assembler, gemini_adapter=_StubGeminiAdapter())

    _submit(service, "corr-assembler-fail")
    finished = service.execute_next_queued_job()

    assert finished is not None
    assert finished["status"] == "failed"
    assert finished["result_json"] is None
    assert finished["error_json"]["error"]["code"] == "JOB_FAILED"


def test_fallback_raises_marks_job_failed_and_error_payload_contains_original_provider_message(tmp_path: Path) -> None:
    """When Gemini fails and fallback also raises, job is failed with JOB_FAILED and original message preserved."""
    repo = CopilotJobRepository(tmp_path / "jobs.db")
    assembler = MagicMock(spec=CopilotEloLlmAssembler)
    assembler.assemble_model_context.return_value = _make_model_context()

    class _GeminiRaises:
        def generate_hybrid_payload(self, *, schema_version, correlation_id, model_context):
            raise RuntimeError("LLM provider key is not configured")

    fallback = MagicMock()
    fallback.build_fallback_payload.side_effect = RuntimeError("Fallback also broken")

    service = CopilotJobService(repository=repo, assembler=assembler, gemini_adapter=_GeminiRaises(), fallback=fallback)

    _submit(service, "corr-fallback-also-fails")
    finished = service.execute_next_queued_job()

    assert finished is not None
    assert finished["status"] == "failed"
    assert finished["result_json"] is None
    assert finished["error_json"]["error"]["code"] == "JOB_FAILED"
    assert "LLM provider key is not configured" in finished["error_json"]["error"]["message"]
