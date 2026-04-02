from __future__ import annotations

import sqlite3
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.services.copilot_blend_sql import CopilotSqlBlendAssembler
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


def _seed_blend_db(db_path: Path) -> None:
    con = sqlite3.connect(db_path)
    con.execute(
        """
        CREATE TABLE copilot_source_player_scores (
            source TEXT NOT NULL,
            player_id INTEGER NOT NULL,
            player_name TEXT NOT NULL,
            projected_points REAL NOT NULL
        );
        """
    )
    con.executemany(
        """
        INSERT INTO copilot_source_player_scores (source, player_id, player_name, projected_points)
        VALUES (?, ?, ?, ?);
        """,
        [
            ("fplcopilot", 101, "Saka", 10.0),
            ("airsenal", 101, "Saka", 8.0),
            ("fplcopilot", 202, "Haaland", 9.0),
            ("airsenal", 202, "Haaland", 11.0),
        ],
    )
    con.commit()
    con.close()


def _submit(service: CopilotJobService, correlation_id: str) -> str:
    accepted = service.submit_job(
        {
            "schema_version": "1.0",
            "correlation_id": correlation_id,
            "task": "hybrid",
            "source_weights": {"fplcopilot": 0.6, "airsenal": 0.4},
            "player_name_contains": None,
            "limit": 25,
        }
    )
    return accepted["job_id"]


def test_happy_path_provider_response_never_uses_degraded_fallback(tmp_path: Path) -> None:
    db_path = tmp_path / "blend.db"
    _seed_blend_db(db_path)

    repo = CopilotJobRepository(db_path)
    assembler = CopilotSqlBlendAssembler(db_path)
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
    db_path = tmp_path / "blend.db"
    _seed_blend_db(db_path)

    repo = CopilotJobRepository(db_path)
    assembler = CopilotSqlBlendAssembler(db_path)
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
    db_path = tmp_path / "blend.db"
    _seed_blend_db(db_path)

    repo = CopilotJobRepository(db_path)
    assembler = CopilotSqlBlendAssembler(db_path)
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
    db_path = tmp_path / "blend.db"
    _seed_blend_db(db_path)

    repo = CopilotJobRepository(db_path)
    assembler = CopilotSqlBlendAssembler(db_path)
    service = CopilotJobService(repository=repo, assembler=assembler, gemini_adapter=_AdapterRaises())

    _submit(service, "corr-failed")
    finished = service.execute_next_queued_job()

    assert finished is not None
    assert finished["status"] == "failed"
    assert finished["result_json"] is None
    assert finished["error_json"]["error"]["code"] == "JOB_FAILED"
    assert "provider hard failure" in finished["error_json"]["error"]["message"]


def test_sql_malicious_filter_input_cannot_break_or_mutate_query_state(tmp_path: Path) -> None:
    db_path = tmp_path / "blend.db"
    _seed_blend_db(db_path)

    assembler = CopilotSqlBlendAssembler(db_path)
    malicious = "x'); DROP TABLE copilot_source_player_scores; --"

    context = assembler.assemble_model_context(
        source_weights={"fplcopilot": 0.6, "airsenal": 0.4},
        player_name_contains=malicious,
        limit=10,
    )

    assert context["blended_players"] == []

    con = sqlite3.connect(db_path)
    count_row = con.execute("SELECT COUNT(*) FROM copilot_source_player_scores;").fetchone()
    con.close()

    assert count_row is not None
    assert count_row[0] == 4
