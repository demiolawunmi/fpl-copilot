from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import asyncio

import pytest

from src.services.copilot_blend_fallback import CopilotBlendFallback
from src.services.copilot_job_repository import CopilotJobRepository
from src.services.copilot_job_service import CopilotJobService
import src.main as main_module
from src.main import _worker_loop, WORKER_IDLE_SLEEP


class _AssemblerOk:
    def assemble_model_context(self, *, source_weights, player_name_contains=None, gameweek=None):
        return {
            "schema_version": "1.0",
            "weights": dict(source_weights),
            "sources": ["elo", "airsenal"],
            "player_name_contains": player_name_contains,
            "blended_players": [
                {"player_id": 1, "player_name": "Saka", "team": "ARS", "position": "MID", "elo_score": 1650.0, "airsenal_predicted_points": 8.0},
            ],
        }


class _AdapterOk:
    def generate_hybrid_payload(self, *, schema_version, correlation_id, model_context):
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


class _FallbackSelectiveFails:
    def __init__(self):
        self._call_count = 0

    def build_fallback_payload(self, *, schema_version, correlation_id, model_context):
        self._call_count += 1
        if self._call_count == 1:
            raise RuntimeError("Fallback first call fails")
        return {
            "schema_version": schema_version,
            "correlation_id": correlation_id,
            "core": {"summary": "Fallback", "confidence": 0.4},
            "recommended_transfers": [],
            "ask_copilot": {
                "answer": "Fallback captain.",
                "rationale": ["Weighted average"],
                "confidence": 0.4,
            },
            "degraded_mode": {
                "is_degraded": True,
                "code": "FALLBACK",
                "message": "LLM unavailable",
                "fallback_used": True,
            },
        }


def _service(tmp_path: Path, adapter, fallback=None) -> CopilotJobService:
    repo = CopilotJobRepository(tmp_path / "jobs.db")
    fb = fallback or CopilotBlendFallback()
    return CopilotJobService(repository=repo, assembler=_AssemblerOk(), gemini_adapter=adapter, fallback=fb)


def test_worker_loop_drains_queue_and_completes_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path, _AdapterOk())
    monkeypatch.setattr(main_module, "_get_copilot_job_service", lambda: service)
    monkeypatch.setattr(main_module, "_copilot_job_service", service)

    accepted = service.submit_job(
        {
            "schema_version": "1.0",
            "correlation_id": "corr-drain",
            "task": "hybrid",
            "source_weights": {"elo": 0.6, "airsenal": 0.4},
        }
    )
    assert accepted["status"] == "queued"

    async def _run_worker():
        shutdown_event = asyncio.Event()
        task = asyncio.create_task(_worker_loop(shutdown_event=shutdown_event))
        await asyncio.sleep(WORKER_IDLE_SLEEP + 1)
        shutdown_event.set()
        await asyncio.wait_for(task, timeout=10)

    asyncio.run(_run_worker())

    completed = service.get_job_status(accepted["job_id"])
    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["result_json"] is not None
    assert completed["result_json"]["correlation_id"] == "corr-drain"
    assert completed["error_json"] is None


def test_worker_loop_survives_failure_and_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fallback = _FallbackSelectiveFails()
    service = _service(tmp_path, _AdapterFails(), fallback)
    monkeypatch.setattr(main_module, "_get_copilot_job_service", lambda: service)
    monkeypatch.setattr(main_module, "_copilot_job_service", service)

    job1 = service.submit_job(
        {
            "schema_version": "1.0",
            "correlation_id": "corr-fail-first",
            "task": "hybrid",
            "source_weights": {"elo": 0.6, "airsenal": 0.4},
        }
    )
    job2 = service.submit_job(
        {
            "schema_version": "1.0",
            "correlation_id": "corr-succeeds-second",
            "task": "hybrid",
            "source_weights": {"elo": 0.5, "airsenal": 0.5},
        }
    )
    assert job1["status"] == "queued"
    assert job2["status"] == "queued"

    async def _run_worker():
        shutdown_event = asyncio.Event()
        task = asyncio.create_task(_worker_loop(shutdown_event=shutdown_event))
        await asyncio.sleep(WORKER_IDLE_SLEEP * 2 + 2)
        shutdown_event.set()
        await asyncio.wait_for(task, timeout=15)

    asyncio.run(_run_worker())

    result1 = service.get_job_status(job1["job_id"])
    assert result1 is not None
    assert result1["status"] == "failed"
    assert result1["result_json"] is None
    assert result1["error_json"] is not None
    assert result1["error_json"]["error"]["code"] == "JOB_FAILED"

    result2 = service.get_job_status(job2["job_id"])
    assert result2 is not None
    assert result2["status"] == "completed"
    assert result2["result_json"] is not None
    assert result2["result_json"]["correlation_id"] == "corr-succeeds-second"
    assert result2["error_json"] is None
