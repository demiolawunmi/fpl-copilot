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
        model_context={"weights": {"fplcopilot": 0.6, "airsenal": 0.4}},
    )

    assert result["schema_version"] == "1.0"
    assert result["correlation_id"] == "corr-ok"
    assert result["degraded_mode"]["is_degraded"] is False
    assert result["recommended_transfers"][0]["in"]["player_id"] == 22
    assert call_log[0]["model"] == "gemini-2.5-flash"
    assert call_log[0]["timeout"] == 25
    assert call_log[0]["config"]["response_mime_type"] == "application/json"


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
        model_context={"weights": {"fplcopilot": 0.6, "airsenal": 0.4}},
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
        model_context={"weights": {"fplcopilot": 0.6, "airsenal": 0.4}},
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
        model_context={"weights": {"fplcopilot": 0.6, "airsenal": 0.4}},
    )

    assert len(call_log) == 3
    assert sleeps == [1, 2]
    assert result["schema_version"] == "1.0"
    assert result["correlation_id"] == "corr-provider"
    assert result["degraded_mode"]["is_degraded"] is True
    assert result["degraded_mode"]["code"] == "PROVIDER_ERROR"
    assert result["degraded_mode"]["fallback_used"] is True
