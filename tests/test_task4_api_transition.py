import asyncio
import json
import sys
import time
import threading
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
import pytest

import src.main as main_module
from src.main import app, _worker_loop
from src.services.copilot_job_service import CopilotJobService
from src.services.copilot_job_repository import CopilotJobRepository
from src.services.copilot_blend_sql import CopilotSqlBlendAssembler


POLL_INTERVAL = 0.5
POLL_TIMEOUT = 30


def _poll_until_terminal(client, job_id, timeout=POLL_TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.get(f"/api/copilot/blend-jobs/{job_id}")
        assert resp.status_code == 200, f"GET returned {resp.status_code}: {resp.text}"
        body = resp.json()
        status = body["status"]
        if status in ("completed", "failed"):
            return body
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Job {job_id} did not reach terminal status within {timeout}s")


class _StubGeminiAdapter:
    def generate_hybrid_payload(self, *, schema_version, correlation_id, model_context):
        return {
            "schema_version": schema_version,
            "correlation_id": correlation_id,
            "core": {"summary": "Ready", "confidence": 0.8},
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


class _FailingGeminiAdapter:
    def generate_hybrid_payload(self, *, schema_version, correlation_id, model_context):
        raise RuntimeError("GEMINI_API_KEY is required for Gemini adapter")


def _seed_db(db_path):
    import sqlite3
    con = sqlite3.connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS copilot_source_player_scores (
            source TEXT NOT NULL,
            player_id INTEGER NOT NULL,
            player_name TEXT NOT NULL,
            projected_points REAL NOT NULL
        )
    """)
    con.execute("INSERT INTO copilot_source_player_scores VALUES ('fplcopilot', 1, 'Salah', 8.5)")
    con.execute("INSERT INTO copilot_source_player_scores VALUES ('airsenal', 1, 'Salah', 7.2)")
    con.execute("INSERT INTO copilot_source_player_scores VALUES ('fplcopilot', 2, 'Haaland', 9.0)")
    con.execute("INSERT INTO copilot_source_player_scores VALUES ('airsenal', 2, 'Haaland', 8.0)")
    con.commit()
    con.close()


def _make_service(tmp_path, gemini_adapter):
    db_path = str(tmp_path / "test.db")
    _seed_db(db_path)
    repo = CopilotJobRepository(db_path)
    assembler = CopilotSqlBlendAssembler(db_path)
    return CopilotJobService(
        repository=repo,
        assembler=assembler,
        gemini_adapter=gemini_adapter,
    )


def _start_worker():
    shutdown_event = asyncio.Event()
    def run_worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_worker_loop(shutdown_event=shutdown_event))
    worker_thread = threading.Thread(target=run_worker, daemon=True)
    worker_thread.start()
    return worker_thread, shutdown_event


def test_happy_path_submit_poll_completed(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "_copilot_job_service", None)
    service = _make_service(tmp_path, _StubGeminiAdapter())
    monkeypatch.setattr(main_module, "_get_copilot_job_service", lambda: service)
    client = TestClient(app)
    worker_thread, shutdown_event = _start_worker()

    try:
        response = client.post(
            "/api/copilot/blend-jobs",
            json={
                "schema_version": "1.0",
                "correlation_id": "task4-happy",
                "source_weights": {"fplcopilot": 0.6, "airsenal": 0.4},
                "task": "hybrid",
                "force_refresh": False,
            },
        )

        assert response.status_code == 202
        submit_body = response.json()
        assert submit_body["status"] == "queued"
        job_id = submit_body["job_id"]

        job = _poll_until_terminal(client, job_id)

        assert job["status"] == "completed"
        assert job["result"] is not None
        assert "core" in job["result"]
        assert "recommended_transfers" in job["result"]
        assert "ask_copilot" in job["result"]
        assert "degraded_mode" in job["result"]
        assert job["error"] is None

        evidence_path = Path(__file__).resolve().parents[2] / ".sisyphus" / "evidence" / "task-4-api-transition.txt"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(
            f"POST /api/copilot/blend-jobs → {response.status_code}\n"
            f"Job ID: {job_id}\n"
            f"Final status: {job['status']}\n"
            f"Result keys: {list(job['result'].keys())}\n"
            f"Degraded mode: {json.dumps(job['result']['degraded_mode'])}\n"
            f"Core summary: {job['result']['core']['summary']}\n"
            f"Confidence: {job['result']['core']['confidence']}\n"
            f"Recommended transfers count: {len(job['result']['recommended_transfers'])}\n"
        )
    finally:
        shutdown_event.set()
        worker_thread.join(timeout=5)


def test_failure_path_submit_poll_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "_copilot_job_service", None)
    service = _make_service(tmp_path, _FailingGeminiAdapter())
    monkeypatch.setattr(main_module, "_get_copilot_job_service", lambda: service)
    client = TestClient(app)
    worker_thread, shutdown_event = _start_worker()

    try:
        response = client.post(
            "/api/copilot/blend-jobs",
            json={
                "schema_version": "1.0",
                "correlation_id": "task4-fail",
                "source_weights": {"fplcopilot": 0.5, "airsenal": 0.5},
                "task": "hybrid",
                "force_refresh": True,
            },
        )

        assert response.status_code == 202
        submit_body = response.json()
        assert submit_body["status"] == "queued"
        job_id = submit_body["job_id"]

        job = _poll_until_terminal(client, job_id)

        assert job["status"] == "failed"
        assert job["error"] is not None
        assert job["error"]["error"]["code"] in ("JOB_FAILED", "VALIDATION_ERROR")
        assert "message" in job["error"]["error"]
        assert job["result"] is None

        evidence_path = Path(__file__).resolve().parents[2] / ".sisyphus" / "evidence" / "task-4-api-transition-error.txt"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(
            f"POST /api/copilot/blend-jobs → {response.status_code}\n"
            f"Job ID: {job_id}\n"
            f"Final status: {job['status']}\n"
            f"Error code: {job['error']['error']['code']}\n"
            f"Error message: {job['error']['error']['message']}\n"
            f"Retryable: {job['error']['error']['retryable']}\n"
            f"Result is None: {job['result'] is None}\n"
        )
    finally:
        shutdown_event.set()
        worker_thread.join(timeout=5)
