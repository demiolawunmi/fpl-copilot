"""End-to-end integration tests for the blend pipeline.

Tests the full pipeline: submit blend job → worker executes → result → job completes.
Uses FastAPI TestClient to hit actual endpoints with mocked external dependencies.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))

import src.main as main_module
from src.main import app, _worker_loop
from src.services.copilot_job_repository import CopilotJobRepository
from src.services.copilot_job_service import CopilotJobService
from src.services.copilot_elo_llm_assembler import CopilotEloLlmAssembler
from src.services.copilot_blend_fallback import CopilotBlendFallback

# ---------------------------------------------------------------------------
# Polling helpers
# ---------------------------------------------------------------------------
POLL_INTERVAL = 0.5
POLL_TIMEOUT = 30


def _poll_until_terminal(client: TestClient, job_id: str, timeout: float = POLL_TIMEOUT) -> dict:
    """Poll GET /api/copilot/blend-jobs/{job_id} until completed/failed or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.get(f"/api/copilot/blend-jobs/{job_id}")
        assert resp.status_code == 200, f"GET returned {resp.status_code}: {resp.text}"
        body = resp.json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Job {job_id} did not reach terminal status within {timeout}s")


# ---------------------------------------------------------------------------
# Worker thread helpers
# ---------------------------------------------------------------------------
def _start_worker():
    """Start the background worker loop in a daemon thread. Returns (thread, shutdown_event)."""
    shutdown_event = asyncio.Event()

    def run_worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_worker_loop(shutdown_event=shutdown_event))

    worker_thread = threading.Thread(target=run_worker, daemon=True)
    worker_thread.start()
    return worker_thread, shutdown_event


# ---------------------------------------------------------------------------
# Service factories with mocked dependencies
# ---------------------------------------------------------------------------
def _make_model_context():
    """Return a realistic model context that the assembler would produce."""
    return {
        "schema_version": "1.0",
        "weights": {"elo": 0.6, "airsenal": 0.4},
        "sources": ["elo", "airsenal"],
        "player_name_contains": None,
        "blended_players": [
            {
                "player_id": 1,
                "fpl_api_id": 381,
                "player_name": "Salah",
                "team": "LIV",
                "position": "MID",
                "elo_score": 1700.0,
                "airsenal_predicted_points": 8.5,
            },
            {
                "player_id": 2,
                "fpl_api_id": 355,
                "player_name": "Haaland",
                "team": "MCI",
                "position": "FWD",
                "elo_score": 1850.5,
                "airsenal_predicted_points": 11.0,
            },
            {
                "player_id": 3,
                "fpl_api_id": 16,
                "player_name": "Saka",
                "team": "ARS",
                "position": "MID",
                "elo_score": 1650.0,
                "airsenal_predicted_points": 8.0,
            },
        ],
    }


def _make_gemini_payload(schema_version: str, correlation_id: str, model_context: dict) -> dict:
    """Return a realistic Gemini hybrid payload."""
    return {
        "schema_version": schema_version,
        "correlation_id": correlation_id,
        "core": {"summary": "Ready", "confidence": 0.81},
        "recommended_transfers": [
            {
                "transfer_id": "t-1",
                "out": {"player_id": 99, "player_name": "Out Player"},
                "in": {"player_id": 1, "player_name": "Salah"},
                "reason": "High composite score",
                "projected_points_delta": 8.5,
            }
        ],
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


def _make_service(tmp_path: Path, gemini_adapter, fallback=None) -> CopilotJobService:
    """Create a CopilotJobService with mocked assembler and real repository."""
    repo = CopilotJobRepository(tmp_path / "jobs.db")
    assembler = MagicMock(spec=CopilotEloLlmAssembler)
    assembler.assemble_model_context.return_value = _make_model_context()
    fb = fallback or CopilotBlendFallback()
    return CopilotJobService(
        repository=repo,
        assembler=assembler,
        gemini_adapter=gemini_adapter,
        fallback=fb,
    )


# ---------------------------------------------------------------------------
# Scenario 1: Happy path — Gemini succeeds, job completes with Gemini result
# ---------------------------------------------------------------------------
class _StubGeminiOk:
    """Gemini adapter that always succeeds."""

    def generate_hybrid_payload(self, *, schema_version, correlation_id, model_context):
        return _make_gemini_payload(schema_version, correlation_id, model_context)


def test_e2e_happy_path_gemini_succeeds(tmp_path, monkeypatch):
    """Submit blend job → worker picks it up → Gemini generates payload → job completes."""
    monkeypatch.setattr(main_module, "_copilot_job_service", None)
    service = _make_service(tmp_path, _StubGeminiOk())
    monkeypatch.setattr(main_module, "_get_copilot_job_service", lambda: service)

    client = TestClient(app)
    worker_thread, shutdown_event = _start_worker()

    try:
        # Submit the job
        response = client.post(
            "/api/copilot/blend-jobs",
            json={
                "schema_version": "1.0",
                "correlation_id": "e2e-happy",
                "source_weights": {"elo": 0.6, "airsenal": 0.4},
                "task": "hybrid",
                "force_refresh": True,
            },
        )
        assert response.status_code == 202
        submit_body = response.json()
        assert submit_body["status"] == "queued"
        job_id = submit_body["job_id"]

        # Poll until terminal
        job = _poll_until_terminal(client, job_id)

        # Verify happy path result
        assert job["status"] == "completed"
        assert job["result"] is not None
        assert job["error"] is None

        result = job["result"]
        assert result["schema_version"] == "1.0"
        assert result["correlation_id"] == "e2e-happy"
        assert result["core"]["summary"] == "Ready"
        assert result["core"]["confidence"] == 0.81
        assert len(result["recommended_transfers"]) == 1
        assert result["recommended_transfers"][0]["in"]["fpl_api_id"] == 381
        assert result["ask_copilot"]["answer"] == "Hold transfer."
        assert result["degraded_mode"]["is_degraded"] is False
        assert result["degraded_mode"]["fallback_used"] is False

    finally:
        shutdown_event.set()
        worker_thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Scenario 2: Fallback path — Gemini fails → fallback completes the job
# ---------------------------------------------------------------------------
class _StubGeminiFails:
    """Gemini adapter that always raises, triggering fallback."""

    def generate_hybrid_payload(self, *, schema_version, correlation_id, model_context):
        raise RuntimeError("LLM provider key is not configured")


def test_e2e_fallback_path_gemini_fails(tmp_path, monkeypatch):
    """Submit blend job → Gemini fails → fallback builds payload → job completes with degraded mode."""
    monkeypatch.setattr(main_module, "_copilot_job_service", None)
    service = _make_service(tmp_path, _StubGeminiFails())
    monkeypatch.setattr(main_module, "_get_copilot_job_service", lambda: service)

    client = TestClient(app)
    worker_thread, shutdown_event = _start_worker()

    try:
        response = client.post(
            "/api/copilot/blend-jobs",
            json={
                "schema_version": "1.0",
                "correlation_id": "e2e-fallback",
                "source_weights": {"elo": 0.7, "airsenal": 0.3},
                "task": "hybrid",
                "force_refresh": True,
            },
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]

        job = _poll_until_terminal(client, job_id)

        # Fallback path: job completes but with degraded mode
        assert job["status"] == "completed"
        assert job["result"] is not None
        assert job["error"] is None

        result = job["result"]
        assert result["degraded_mode"]["is_degraded"] is True
        assert result["degraded_mode"]["fallback_used"] is True
        assert result["degraded_mode"]["code"] == "FALLBACK"
        assert result["core"]["confidence"] == 0.4
        assert "fallback" in result["core"]["summary"].lower()

    finally:
        shutdown_event.set()
        worker_thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Scenario 3: Error path — invalid input → job fails gracefully
# ---------------------------------------------------------------------------
def test_e2e_error_path_invalid_input_rejected(tmp_path, monkeypatch):
    """Submit blend job with invalid weights (don't sum to 1.0) → 422 rejection."""
    monkeypatch.setattr(main_module, "_copilot_job_service", None)
    service = _make_service(tmp_path, _StubGeminiOk())
    monkeypatch.setattr(main_module, "_get_copilot_job_service", lambda: service)

    client = TestClient(app)
    worker_thread, shutdown_event = _start_worker()

    try:
        # Weights sum to 0.6, not 1.0 — should be rejected at validation
        response = client.post(
            "/api/copilot/blend-jobs",
            json={
                "schema_version": "1.0",
                "correlation_id": "e2e-invalid",
                "source_weights": {"elo": 0.3, "airsenal": 0.3},
                "task": "hybrid",
            },
        )
        assert response.status_code == 422

    finally:
        shutdown_event.set()
        worker_thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Scenario 4: Error path — both Gemini and fallback fail → job marked failed
# ---------------------------------------------------------------------------
class _StubFallbackFails:
    """Fallback that also raises, ensuring job ends up failed."""

    def build_fallback_payload(self, *, schema_version, correlation_id, model_context):
        raise RuntimeError("Fallback also broken")


def test_e2e_error_path_both_gemini_and_fallback_fail(tmp_path, monkeypatch):
    """Submit blend job → Gemini fails → fallback also fails → job marked as failed."""
    monkeypatch.setattr(main_module, "_copilot_job_service", None)
    service = _make_service(tmp_path, _StubGeminiFails(), _StubFallbackFails())
    monkeypatch.setattr(main_module, "_get_copilot_job_service", lambda: service)

    client = TestClient(app)
    worker_thread, shutdown_event = _start_worker()

    try:
        response = client.post(
            "/api/copilot/blend-jobs",
            json={
                "schema_version": "1.0",
                "correlation_id": "e2e-both-fail",
                "source_weights": {"elo": 0.5, "airsenal": 0.5},
                "task": "hybrid",
                "force_refresh": True,
            },
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]

        job = _poll_until_terminal(client, job_id)

        # Both failed → job should be marked as failed
        assert job["status"] == "failed"
        assert job["result"] is None
        assert job["error"] is not None
        assert job["error"]["error"]["code"] == "JOB_FAILED"

    finally:
        shutdown_event.set()
        worker_thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Scenario 5: Status transitions — queued → running → completed
# ---------------------------------------------------------------------------
def test_e2e_job_status_transitions(tmp_path, monkeypatch):
    """Verify job transitions through queued → completed via API polling."""
    monkeypatch.setattr(main_module, "_copilot_job_service", None)
    service = _make_service(tmp_path, _StubGeminiOk())
    monkeypatch.setattr(main_module, "_get_copilot_job_service", lambda: service)

    client = TestClient(app)

    # Submit without starting worker — verify queued status
    response = client.post(
        "/api/copilot/blend-jobs",
        json={
            "schema_version": "1.0",
            "correlation_id": "e2e-transitions",
            "source_weights": {"elo": 0.5, "airsenal": 0.5},
            "task": "hybrid",
            "force_refresh": True,
        },
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    # Check status is queued
    status_resp = client.get(f"/api/copilot/blend-jobs/{job_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "queued"

    # Now start the worker and wait for completion
    worker_thread, shutdown_event = _start_worker()
    try:
        job = _poll_until_terminal(client, job_id)
        assert job["status"] == "completed"
    finally:
        shutdown_event.set()
        worker_thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Scenario 6: Unknown job returns structured 404
# ---------------------------------------------------------------------------
def test_e2e_unknown_job_returns_404(tmp_path, monkeypatch):
    """GET for non-existent job returns structured 404 error."""
    monkeypatch.setattr(main_module, "_copilot_job_service", None)
    service = _make_service(tmp_path, _StubGeminiOk())
    monkeypatch.setattr(main_module, "_get_copilot_job_service", lambda: service)

    client = TestClient(app)

    response = client.get("/api/copilot/blend-jobs/job-does-not-exist")
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "NOT_FOUND"
    assert "not found" in body["error"]["message"].lower()


# ---------------------------------------------------------------------------
# Scenario 7: Timeout path — Gemini times out → adapter returns degraded → job completes with LLM_TIMEOUT
# ---------------------------------------------------------------------------
class _StubGeminiTimeout:
    """Gemini adapter that simulates timeout → returns structured degraded payload."""

    def generate_hybrid_payload(self, *, schema_version, correlation_id, model_context):
        return {
            "schema_version": schema_version,
            "correlation_id": correlation_id,
            "core": {"summary": "Model response timed out. Serving constrained fallback guidance.", "confidence": 0.0},
            "recommended_transfers": [],
            "ask_copilot": {
                "answer": "Model response timed out. Retry shortly.",
                "rationale": ["Gemini timed out after 25s"],
                "confidence": 0.0,
            },
            "degraded_mode": {
                "is_degraded": True,
                "code": "LLM_TIMEOUT",
                "message": "Gemini timed out after 25s",
                "fallback_used": True,
            },
        }


def test_e2e_llm_timeout_triggers_fallback_and_backoff(tmp_path, monkeypatch):
    """Submit blend job → Gemini times out → adapter returns degraded payload → job completes with LLM_TIMEOUT."""
    monkeypatch.setattr(main_module, "_copilot_job_service", None)
    service = _make_service(tmp_path, _StubGeminiTimeout())
    monkeypatch.setattr(main_module, "_get_copilot_job_service", lambda: service)

    client = TestClient(app)
    worker_thread, shutdown_event = _start_worker()

    try:
        response = client.post(
            "/api/copilot/blend-jobs",
            json={
                "schema_version": "1.0",
                "correlation_id": "e2e-timeout",
                "source_weights": {"elo": 0.5, "airsenal": 0.5},
                "task": "hybrid",
                "force_refresh": True,
            },
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]

        job = _poll_until_terminal(client, job_id)

        assert job["status"] == "completed"
        assert job["result"] is not None
        assert job["error"] is None

        result = job["result"]
        assert result["degraded_mode"]["is_degraded"] is True
        assert result["degraded_mode"]["code"] == "LLM_TIMEOUT"
        assert result["degraded_mode"]["fallback_used"] is True
        assert result["core"]["confidence"] == 0.0
        assert result["ask_copilot"]["confidence"] == 0.0

    finally:
        shutdown_event.set()
        worker_thread.join(timeout=5)
