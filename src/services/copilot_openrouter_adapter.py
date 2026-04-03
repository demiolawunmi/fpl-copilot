from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import requests
from pydantic import ValidationError


class _HttpSession(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: int,
    ) -> Any:
        ...


@dataclass(frozen=True)
class OpenRouterAdapterConfig:
    model_name: str = "openrouter/free"
    timeout_seconds: int = 25
    max_retries: int = 2
    retry_backoff_seconds: tuple[int, ...] = (1, 2)
    base_url: str = "https://openrouter.ai/api/v1"


class CopilotOpenRouterAdapter:
    def __init__(
        self,
        *,
        session: _HttpSession | None = None,
        config: OpenRouterAdapterConfig | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or OpenRouterAdapterConfig()
        self.sleep_fn = sleep_fn
        self.session = session or requests.Session()
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter adapter")

        chosen_model = os.environ.get("OPENROUTER_MODEL", "").strip()
        if chosen_model:
            self.config = OpenRouterAdapterConfig(
                model_name=chosen_model,
                timeout_seconds=self.config.timeout_seconds,
                max_retries=self.config.max_retries,
                retry_backoff_seconds=self.config.retry_backoff_seconds,
                base_url=self.config.base_url,
            )

    def _build_prompt(self, model_context: dict[str, Any]) -> str:
        weights = model_context.get("weights", {})
        elo_weight = weights.get("elo", 0.0)
        airsenal_weight = weights.get("airsenal", 0.0)
        blended_players = model_context.get("blended_players", [])

        player_summaries = []
        for p in blended_players[:15]:
            player_summaries.append(
                f"- {p['player_name']} ({p['team']}, {p['position']}): "
                f"ELO={p['elo_score']:.1f}, AIrsenal={p['airsenal_predicted_points']:.1f}"
            )
        players_text = "\n".join(player_summaries) if player_summaries else "(no player data available)"

        weight_instruction = ""
        if elo_weight > 0 and airsenal_weight > 0:
            elo_pct = int(round(elo_weight * 100))
            airsenal_pct = int(round(airsenal_weight * 100))
            weight_instruction = (
                f"WEIGHTING: ELO scores are weighted at {elo_pct}%, AIrsenal predictions at {airsenal_pct}%. "
                f"Prioritize players who score well on BOTH metrics. "
                f"A player with high ELO but low AIrsenal prediction may be overvalued by team strength alone. "
                f"A player with high AIrsenal but low ELO may be in a weak team but have favorable fixtures."
            )
        elif elo_weight > 0:
            weight_instruction = (
                "WEIGHTING: ELO scores are the sole signal (100%). "
                "Focus on players from strong teams in favorable positions."
            )
        elif airsenal_weight > 0:
            weight_instruction = (
                "WEIGHTING: AIrsenal predictions are the sole signal (100%). "
                "Focus on players with the highest projected points."
            )

        context_json = json.dumps(model_context, separators=(",", ":"), sort_keys=True)
        return (
            "ROLE: You are a football analytics assistant for FPL Copilot. "
            "You must only use the provided context and never fabricate hidden sources.\n\n"
            "DATA SOURCES:\n"
            "1. ELO SCORES — measure team strength adjusted for player position. "
            "Higher ELO = player benefits from stronger team context.\n"
            "2. AIRSENAL PREDICTIONS — ML-projected points for the upcoming gameweek. "
            "Higher = more expected FPL points.\n\n"
            f"{weight_instruction}\n\n"
            "TOP PLAYERS (ELO + AIrsenal):\n"
            f"{players_text}\n\n"
            "TASKS:\n"
            "1. RECOMMEND TRANSFERS: Identify players to transfer IN and OUT. "
            "For each transfer, explain WHY referencing BOTH ELO scores and AIrsenal predictions. "
            "Example: 'Transfer IN Haaland — ELO 1850 (strongest team context) + AIrsenal 11.0 pts (highest projection).'\n"
            "2. CAPTAIN PICK: Recommend a captain based on the blended assessment of ELO and AIrsenal. "
            "The ideal captain combines high team strength (ELO) with high projected output (AIrsenal).\n"
            "3. EXPLAIN REASONING: Every recommendation must include a clear 'reason' field "
            "that references specific ELO scores and/or AIrsenal predictions from the data.\n\n"
            "OUTPUT POLICY: Return exactly one strict JSON object conforming to this schema and no markdown/text.\n"
            "{\n"
            '  "core": {"summary": "string", "confidence": 0.0},\n'
            '  "recommended_transfers": [\n'
            "    {\n"
            '      "transfer_id": "string",\n'
            '      "out": {"player_id": 0, "player_name": "string"},\n'
            '      "in": {"player_id": 0, "player_name": "string"},\n'
            '      "reason": "string (must reference ELO and/or AIrsenal data)",\n'
            '      "projected_points_delta": 0.0\n'
            "    }\n"
            "  ],\n"
            '  "ask_copilot": {"answer": "string", "rationale": ["string"], "confidence": 0.0}\n'
            "}\n"
            "If unsure, still return valid JSON with conservative confidence.\n"
            f"CONTEXT_JSON={context_json}"
        )

    def _extract_json(self, raw_text: str) -> str:
        import json as _json
        import re

        stripped = raw_text.strip()
        if stripped.startswith("{"):
            try:
                _json.loads(stripped)
                return stripped
            except _json.JSONDecodeError:
                pass

        for pattern in [r"```json\s*\r?\n(.*?)\r?\n```", r"```\s*\r?\n(.*?)\r?\n```"]:
            match = re.search(pattern, raw_text, re.DOTALL)
            if match:
                candidate = match.group(1).strip()
                try:
                    _json.loads(candidate)
                    return candidate
                except _json.JSONDecodeError:
                    pass

        start = raw_text.find("{")
        if start == -1:
            raise ValueError(f"No JSON object found in OpenRouter response: {raw_text[:200]}")

        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(raw_text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw_text[start : i + 1].strip()
                    try:
                        _json.loads(candidate)
                        return candidate
                    except _json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Malformed JSON extracted via brace counting: {exc}"
                        ) from exc

        raise ValueError(f"Unterminated JSON in OpenRouter response: {raw_text[:200]}")

    def _invoke_model(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "fpl-copilot"),
        }
        body = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
        response = self.session.post(
            f"{self.config.base_url}/chat/completions",
            headers=headers,
            json=body,
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            raise ValueError("OpenRouter response has no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("OpenRouter response content is empty")
        return content

    def _validate_hybrid_payload(
        self,
        parsed_json: dict[str, Any],
        *,
        schema_version: str,
        correlation_id: str,
    ) -> dict[str, Any]:
        from src.main import CopilotHybridResultPayload

        candidate = {
            "schema_version": schema_version,
            "correlation_id": correlation_id,
            "core": parsed_json.get("core"),
            "recommended_transfers": parsed_json.get("recommended_transfers"),
            "ask_copilot": parsed_json.get("ask_copilot"),
            "degraded_mode": {
                "is_degraded": False,
                "code": None,
                "message": None,
                "fallback_used": False,
            },
        }
        validated = CopilotHybridResultPayload.model_validate(candidate)
        return validated.model_dump(by_alias=True)

    def _build_degraded_payload(
        self,
        *,
        schema_version: str,
        correlation_id: str,
        code: str,
        message: str,
    ) -> dict[str, Any]:
        from src.main import CopilotHybridResultPayload

        summary_map = {
            "JSON_PARSE_FAILED": "Model response could not be parsed. Serving constrained fallback guidance.",
            "SCHEMA_VALIDATION_FAILED": "Model response did not match expected format. Serving constrained fallback guidance.",
            "LLM_TIMEOUT": "Model response timed out. Serving constrained fallback guidance.",
            "PROVIDER_ERROR": "Model provider encountered an error. Serving constrained fallback guidance.",
        }
        summary = summary_map.get(code, "Temporary model degradation. Serving constrained fallback guidance.")

        answer_map = {
            "JSON_PARSE_FAILED": "Model output could not be parsed. Retry shortly.",
            "SCHEMA_VALIDATION_FAILED": "Model output format was invalid. Retry shortly.",
            "LLM_TIMEOUT": "Model response timed out. Retry shortly.",
            "PROVIDER_ERROR": "Model provider error. Retry shortly.",
        }
        answer = answer_map.get(code, "Model output unavailable. Retry shortly.")

        fallback = {
            "schema_version": schema_version,
            "correlation_id": correlation_id,
            "core": {
                "summary": summary,
                "confidence": 0.0,
            },
            "recommended_transfers": [],
            "ask_copilot": {
                "answer": answer,
                "rationale": [message],
                "confidence": 0.0,
            },
            "degraded_mode": {
                "is_degraded": True,
                "code": code,
                "message": message,
                "fallback_used": True,
            },
        }
        validated = CopilotHybridResultPayload.model_validate(fallback)
        return validated.model_dump(by_alias=True)

    def generate_hybrid_payload(
        self,
        *,
        schema_version: str,
        correlation_id: str,
        model_context: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = self._build_prompt(model_context)
        attempts = self.config.max_retries + 1
        last_error_message = "OpenRouter response unavailable"
        degraded_code = "PROVIDER_ERROR"

        for attempt_idx in range(attempts):
            try:
                raw_text = self._invoke_model(prompt)
                json_text = self._extract_json(raw_text)
                parsed = json.loads(json_text)
                if not isinstance(parsed, dict):
                    raise ValueError("OpenRouter response must be a JSON object")
                return self._validate_hybrid_payload(
                    parsed,
                    schema_version=schema_version,
                    correlation_id=correlation_id,
                )
            except requests.Timeout:
                degraded_code = "LLM_TIMEOUT"
                last_error_message = f"OpenRouter timed out after {self.config.timeout_seconds}s"
                if attempt_idx >= self.config.max_retries:
                    break
            except requests.RequestException as exc:
                degraded_code = "PROVIDER_ERROR"
                last_error_message = f"OpenRouter provider error: {exc}"
                if attempt_idx >= self.config.max_retries:
                    break
            except ValidationError as exc:
                degraded_code = "SCHEMA_VALIDATION_FAILED"
                last_error_message = f"OpenRouter output did not match expected schema: {exc}"
                if attempt_idx >= self.config.max_retries:
                    break
            except ValueError as exc:
                degraded_code = "SCHEMA_VALIDATION_FAILED"
                last_error_message = f"OpenRouter output could not be parsed as JSON: {exc}"
                if attempt_idx >= self.config.max_retries:
                    break
            except Exception as exc:
                degraded_code = "PROVIDER_ERROR"
                last_error_message = f"OpenRouter provider error: {exc}"
                if attempt_idx >= self.config.max_retries:
                    break

            if attempt_idx < self.config.max_retries:
                if attempt_idx < len(self.config.retry_backoff_seconds):
                    delay_seconds = self.config.retry_backoff_seconds[attempt_idx]
                else:
                    delay_seconds = self.config.retry_backoff_seconds[-1]
                self.sleep_fn(delay_seconds)

        return self._build_degraded_payload(
            schema_version=schema_version,
            correlation_id=correlation_id,
            code=degraded_code,
            message=last_error_message,
        )
