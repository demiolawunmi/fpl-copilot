from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.services.copilot_blend_fallback import CopilotBlendFallback
from src.services.copilot_job_repository import CopilotJobRepository
from src.services.copilot_job_service import CopilotJobService


class _AssemblerOk:
    def assemble_model_context(self, *, source_weights, player_name_contains=None, gameweek=None, bank=None, free_transfers=None, current_squad=None, fpl_team_id=None):
        return {
            "schema_version": "1.0",
            "weights": dict(source_weights),
            "gameweek": gameweek,
            "bank": bank,
            "free_transfers": free_transfers,
            "sources": ["elo", "airsenal"],
            "player_name_contains": player_name_contains,
            "current_squad": current_squad or [],
            "blended_players": [
                {"player_id": 1, "player_name": "Saka", "team": "ARS", "position": "MID", "elo_score": 1650.0, "airsenal_predicted_points": 8.0},
                {"player_id": 2, "player_name": "Haaland", "team": "MCI", "position": "FWD", "elo_score": 1850.5, "airsenal_predicted_points": 11.0},
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


class _FallbackOk:
    def build_fallback_payload(self, *, schema_version, correlation_id, model_context):
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


class _FallbackFails:
    def build_fallback_payload(self, *, schema_version, correlation_id, model_context):
        raise RuntimeError("Fallback also broken")


def _service(tmp_path: Path, adapter, fallback=None) -> CopilotJobService:
    repo = CopilotJobRepository(tmp_path / "jobs.db")
    fb = fallback or CopilotBlendFallback()
    return CopilotJobService(repository=repo, assembler=_AssemblerOk(), gemini_adapter=adapter, fallback=fb)


def test_from_dependencies_selects_openrouter_provider(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")

    class _StubOpenRouterAdapter:
        def __init__(self):
            pass

        def generate_hybrid_payload(self, *, schema_version, correlation_id, model_context):
            return {
                "schema_version": schema_version,
                "correlation_id": correlation_id,
                "core": {"summary": "ok", "confidence": 0.5},
                "recommended_transfers": [],
                "ask_copilot": {"answer": "ok", "rationale": [], "confidence": 0.5},
                "degraded_mode": {
                    "is_degraded": False,
                    "code": None,
                    "message": None,
                    "fallback_used": False,
                },
            }

    from src.services import copilot_job_service as mod

    original = mod.CopilotOpenRouterAdapter
    mod.CopilotOpenRouterAdapter = _StubOpenRouterAdapter
    try:
        service = CopilotJobService.from_dependencies(db_path=str(tmp_path / "jobs.db"))
        assert service.gemini_adapter.__class__.__name__ == "_StubOpenRouterAdapter"
    finally:
        mod.CopilotOpenRouterAdapter = original


def test_happy_path_submit_execute_and_complete(tmp_path: Path) -> None:
    service = _service(tmp_path, _AdapterOk())

    accepted = service.submit_job(
        {
            "schema_version": "1.0",
            "correlation_id": "corr-ok",
            "task": "hybrid",
            "source_weights": {"elo": 0.6, "airsenal": 0.4},
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


def test_gemini_failure_uses_fallback_and_completes(tmp_path: Path) -> None:
    service = _service(tmp_path, _AdapterFails(), _FallbackOk())

    accepted = service.submit_job(
        {
            "schema_version": "1.0",
            "correlation_id": "corr-fallback",
            "task": "hybrid",
            "source_weights": {"elo": 0.7, "airsenal": 0.3},
        }
    )

    finished = service.execute_next_queued_job()
    assert finished is not None
    assert finished["job_id"] == accepted["job_id"]
    assert finished["status"] == "completed"
    result = finished["result_json"]
    assert result["degraded_mode"]["is_degraded"] is True
    assert result["degraded_mode"]["fallback_used"] is True
    assert result["degraded_mode"]["code"] == "FALLBACK"


def test_both_gemini_and_fallback_failure_marks_job_failed(tmp_path: Path) -> None:
    service = _service(tmp_path, _AdapterFails(), _FallbackFails())

    accepted = service.submit_job(
        {
            "schema_version": "1.0",
            "correlation_id": "corr-both-fail",
            "task": "hybrid",
            "source_weights": {"elo": 0.5, "airsenal": 0.5},
        }
    )

    finished = service.execute_next_queued_job()
    assert finished is not None
    assert finished["status"] == "failed"
    assert finished["result_json"] is None
    assert finished["error_json"]["error"]["code"] == "JOB_FAILED"


def test_poll_status_reads_current_state_and_missing_job(tmp_path: Path) -> None:
    service = _service(tmp_path, _AdapterOk())

    accepted = service.submit_job(
        {
            "schema_version": "1.0",
            "correlation_id": "corr-poll",
            "task": "hybrid",
            "source_weights": {"elo": 0.7, "airsenal": 0.3},
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


def test_execute_passes_gameweek_to_assembler(tmp_path: Path) -> None:
    class _AssemblerCapture:
        def __init__(self):
            self.captured = None

        def assemble_model_context(self, *, source_weights, player_name_contains=None, gameweek=None, bank=None, free_transfers=None, current_squad=None, fpl_team_id=None):
            self.captured = {
                "source_weights": source_weights,
                "player_name_contains": player_name_contains,
                "gameweek": gameweek,
                "bank": bank,
                "free_transfers": free_transfers,
                "current_squad": current_squad,
                "fpl_team_id": fpl_team_id,
            }
            return {
                "schema_version": "1.0",
                "weights": dict(source_weights),
                "gameweek": gameweek,
                "bank": bank,
                "free_transfers": free_transfers,
                "sources": ["elo", "airsenal"],
                "current_squad": current_squad or [],
                "blended_players": [
                    {"player_id": 1, "player_name": "Saka", "team": "ARS", "position": "MID", "elo_score": 1650.0, "airsenal_predicted_points": 8.0},
                ],
            }

    repo = CopilotJobRepository(tmp_path / "jobs.db")
    assembler = _AssemblerCapture()
    service = CopilotJobService(
        repository=repo,
        assembler=assembler,
        gemini_adapter=_AdapterOk(),
        fallback=CopilotBlendFallback(),
    )

    service.submit_job(
        {
            "schema_version": "1.0",
            "correlation_id": "corr-gw",
            "task": "hybrid",
            "source_weights": {"elo": 0.7, "airsenal": 0.3},
            "player_name_contains": "Saka",
            "gameweek": 27,
            "bank": 1.3,
            "free_transfers": 2,
            "current_squad": [
                {
                    "fpl_api_id": 16,
                    "player_name": "Saka",
                    "team": "Arsenal",
                    "position": "MID",
                    "price": 10.2,
                    "x_pts": 8.0,
                }
            ],
        }
    )
    service.execute_next_queued_job()

    assert assembler.captured["gameweek"] == 27
    assert assembler.captured["bank"] == 1.3
    assert assembler.captured["free_transfers"] == 2
    assert assembler.captured["current_squad"][0]["fpl_api_id"] == 16
    assert assembler.captured["player_name_contains"] == "Saka"
    assert assembler.captured["source_weights"] == {"elo": 0.7, "airsenal": 0.3}
    assert assembler.captured["fpl_team_id"] is None


class _AdapterWithTransfers:
    def generate_hybrid_payload(self, *, schema_version, correlation_id, model_context):
        return {
            "schema_version": schema_version,
            "correlation_id": correlation_id,
            "core": {"summary": "Ready", "confidence": 0.81},
            "recommended_transfers": [
                {
                    "out": {"player_id": 1, "player_name": "Saka"},
                    "in": {"player_id": 2, "player_name": "Haaland"},
                    "projected_points_delta": 999.0,
                    "reason": "test",
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


def test_transfer_delta_overwritten_from_blended_airsenal_xpts(tmp_path: Path) -> None:
    service = _service(tmp_path, _AdapterWithTransfers())

    accepted = service.submit_job(
        {
            "schema_version": "1.0",
            "correlation_id": "corr-delta",
            "task": "hybrid",
            "source_weights": {"elo": 0.6, "airsenal": 0.4},
        }
    )
    service.execute_next_queued_job()
    finished = service.get_job_status(accepted["job_id"])
    assert finished is not None
    transfers = finished["result_json"]["recommended_transfers"]
    assert len(transfers) == 1
    assert transfers[0]["projected_points_delta"] == 3.0


def test_execute_passes_fpl_team_id_to_assembler(tmp_path: Path) -> None:
    class _AssemblerCapture:
        def __init__(self):
            self.captured = None

        def assemble_model_context(self, *, source_weights, player_name_contains=None, gameweek=None, bank=None, free_transfers=None, current_squad=None, fpl_team_id=None):
            self.captured = {"fpl_team_id": fpl_team_id}
            return {
                "schema_version": "1.0",
                "weights": dict(source_weights),
                "gameweek": gameweek,
                "bank": bank,
                "free_transfers": free_transfers,
                "sources": ["elo", "airsenal"],
                "current_squad": current_squad or [],
                "blended_players": [
                    {"player_id": 1, "player_name": "Saka", "team": "ARS", "position": "MID", "elo_score": 1650.0, "airsenal_predicted_points": 8.0},
                ],
            }

    repo = CopilotJobRepository(tmp_path / "jobs.db")
    assembler = _AssemblerCapture()
    service = CopilotJobService(
        repository=repo,
        assembler=assembler,
        gemini_adapter=_AdapterOk(),
        fallback=CopilotBlendFallback(),
    )

    service.submit_job(
        {
            "schema_version": "1.0",
            "correlation_id": "corr-tid",
            "task": "hybrid",
            "source_weights": {"elo": 0.7, "airsenal": 0.3},
            "fpl_team_id": 8994418,
        }
    )
    service.execute_next_queued_job()

    assert assembler.captured["fpl_team_id"] == 8994418
