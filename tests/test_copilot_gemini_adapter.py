from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.services.copilot_gemini_adapter import CopilotGeminiAdapter


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text


class _ModelsAPI:
    def __init__(self, outcomes: list[object], call_log: list[dict]) -> None:
        self._outcomes = outcomes
        self._call_log = call_log

    def generate_content(self, *, model, contents, config, timeout):
        self._call_log.append(
            {
                "model": model,
                "contents": contents,
                "config": config,
                "timeout": timeout,
            }
        )
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _Response(outcome)


class _Client:
    def __init__(self, outcomes: list[object], call_log: list[dict]) -> None:
        self.models = _ModelsAPI(outcomes=outcomes, call_log=call_log)


_ELO_CONTEXT = {
    "schema_version": "1.0",
    "weights": {"elo": 0.7, "airsenal": 0.3},
    "sources": ["elo", "airsenal"],
    "blended_players": [
        {"player_id": 101, "player_name": "Saka", "team": "ARS", "position": "MID", "elo_score": 1650.0, "airsenal_predicted_points": 8.0},
        {"player_id": 202, "player_name": "Haaland", "team": "MCI", "position": "FWD", "elo_score": 1850.5, "airsenal_predicted_points": 11.0},
    ],
}


def test_success_with_valid_gemini_response() -> None:
    call_log: list[dict] = []
    adapter = CopilotGeminiAdapter(
        client=_Client(
            outcomes=[
                """
                {
                    "core": {"summary": "Blend complete", "confidence": 0.82},
                    "recommended_transfers": [
                        {
                            "transfer_id": "t1",
                            "out": {"player_id": 11, "player_name": "Player Out"},
                            "in": {"player_id": 22, "player_name": "Player In"},
                            "reason": "Better projected minutes",
                            "projected_points_delta": 1.9
                        }
                    ],
                    "ask_copilot": {
                        "answer": "Prioritize value and minutes.",
                        "rationale": ["Weighted projection favored transfer"],
                        "confidence": 0.8
                    }
                }
                """
            ],
            call_log=call_log,
        ),
        sleep_fn=lambda _: None,
    )

    result = adapter.generate_hybrid_payload(
        schema_version="1.0",
        correlation_id="corr-ok",
        model_context=_ELO_CONTEXT,
    )

    assert result["schema_version"] == "1.0"
    assert result["correlation_id"] == "corr-ok"
    assert result["degraded_mode"]["is_degraded"] is False
    assert result["recommended_transfers"][0]["in"]["player_id"] == 22
    assert call_log[0]["model"] == "gemini-2.5-flash"
    assert call_log[0]["timeout"] == 25
    assert call_log[0]["config"]["response_mime_type"] == "application/json"


def test_prompt_includes_elo_weight_instructions() -> None:
    call_log: list[dict] = []
    adapter = CopilotGeminiAdapter(
        client=_Client(
            outcomes=['{"core":{"summary":"ok","confidence":0.5},"recommended_transfers":[],"ask_copilot":{"answer":"ok","rationale":[],"confidence":0.5}}'],
            call_log=call_log,
        ),
        sleep_fn=lambda _: None,
    )

    adapter.generate_hybrid_payload(
        schema_version="1.0",
        correlation_id="corr-prompt",
        model_context=_ELO_CONTEXT,
    )

    prompt = call_log[0]["contents"]
    assert "ELO scores are weighted at 70%" in prompt
    assert "AIrsenal predictions at 30%" in prompt
    assert "Saka (ARS, MID): ELO=1650.0, AIrsenal=8.0" in prompt
    assert "Haaland (MCI, FWD): ELO=1850.5, AIrsenal=11.0" in prompt


def test_prompt_handles_elo_only_weighting() -> None:
    call_log: list[dict] = []
    adapter = CopilotGeminiAdapter(
        client=_Client(
            outcomes=['{"core":{"summary":"ok","confidence":0.5},"recommended_transfers":[],"ask_copilot":{"answer":"ok","rationale":[],"confidence":0.5}}'],
            call_log=call_log,
        ),
        sleep_fn=lambda _: None,
    )

    adapter.generate_hybrid_payload(
        schema_version="1.0",
        correlation_id="corr-elo-only",
        model_context={
            "schema_version": "1.0",
            "weights": {"elo": 1.0, "airsenal": 0.0},
            "sources": ["elo"],
            "blended_players": [
                {"player_id": 101, "player_name": "Saka", "team": "ARS", "position": "MID", "elo_score": 1650.0, "airsenal_predicted_points": 0.0},
            ],
        },
    )

    prompt = call_log[0]["contents"]
    assert "ELO scores are the sole signal (100%)" in prompt


def test_prompt_handles_airsenal_only_weighting() -> None:
    call_log: list[dict] = []
    adapter = CopilotGeminiAdapter(
        client=_Client(
            outcomes=['{"core":{"summary":"ok","confidence":0.5},"recommended_transfers":[],"ask_copilot":{"answer":"ok","rationale":[],"confidence":0.5}}'],
            call_log=call_log,
        ),
        sleep_fn=lambda _: None,
    )

    adapter.generate_hybrid_payload(
        schema_version="1.0",
        correlation_id="corr-airsenal-only",
        model_context={
            "schema_version": "1.0",
            "weights": {"elo": 0.0, "airsenal": 1.0},
            "sources": ["airsenal"],
            "blended_players": [
                {"player_id": 101, "player_name": "Saka", "team": "ARS", "position": "MID", "elo_score": 0.0, "airsenal_predicted_points": 8.0},
            ],
        },
    )

    prompt = call_log[0]["contents"]
    assert "AIrsenal predictions are the sole signal (100%)" in prompt


def test_timeout_retries_then_returns_structured_degraded_payload() -> None:
    call_log: list[dict] = []
    sleeps: list[float] = []
    adapter = CopilotGeminiAdapter(
        client=_Client(
            outcomes=[TimeoutError("t1"), TimeoutError("t2"), TimeoutError("t3")],
            call_log=call_log,
        ),
        sleep_fn=lambda seconds: sleeps.append(seconds),
    )

    result = adapter.generate_hybrid_payload(
        schema_version="1.0",
        correlation_id="corr-timeout",
        model_context=_ELO_CONTEXT,
    )

    assert len(call_log) == 3
    assert sleeps == [1, 2]
    assert result["schema_version"] == "1.0"
    assert result["correlation_id"] == "corr-timeout"
    assert result["degraded_mode"]["is_degraded"] is True
    assert result["degraded_mode"]["code"] == "LLM_TIMEOUT"
    assert result["degraded_mode"]["fallback_used"] is True
    assert result["recommended_transfers"] == []


def test_malformed_json_retries_then_returns_schema_validation_degraded_payload() -> None:
    call_log: list[dict] = []
    sleeps: list[float] = []
    adapter = CopilotGeminiAdapter(
        client=_Client(
            outcomes=["not-json", "still-not-json", "[]"],
            call_log=call_log,
        ),
        sleep_fn=lambda seconds: sleeps.append(seconds),
    )

    result = adapter.generate_hybrid_payload(
        schema_version="1.0",
        correlation_id="corr-malformed",
        model_context=_ELO_CONTEXT,
    )

    assert len(call_log) == 3
    assert sleeps == [1, 2]
    assert result["schema_version"] == "1.0"
    assert result["correlation_id"] == "corr-malformed"
    assert result["degraded_mode"]["is_degraded"] is True
    assert result["degraded_mode"]["code"] == "SCHEMA_VALIDATION_FAILED"
    assert result["degraded_mode"]["fallback_used"] is True
    assert result["ask_copilot"]["confidence"] == 0.0


def test_provider_error_retries_then_returns_provider_degraded_payload() -> None:
    call_log: list[dict] = []
    sleeps: list[float] = []
    adapter = CopilotGeminiAdapter(
        client=_Client(
            outcomes=[RuntimeError("upstream down"), RuntimeError("upstream down"), RuntimeError("upstream down")],
            call_log=call_log,
        ),
        sleep_fn=lambda seconds: sleeps.append(seconds),
    )

    result = adapter.generate_hybrid_payload(
        schema_version="1.0",
        correlation_id="corr-provider",
        model_context=_ELO_CONTEXT,
    )

    assert len(call_log) == 3
    assert sleeps == [1, 2]
    assert result["schema_version"] == "1.0"
    assert result["correlation_id"] == "corr-provider"
    assert result["degraded_mode"]["is_degraded"] is True
    assert result["degraded_mode"]["code"] == "PROVIDER_ERROR"
    assert result["degraded_mode"]["fallback_used"] is True
