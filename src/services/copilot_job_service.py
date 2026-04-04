from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Mapping, Protocol
from uuid import uuid4

from src.services.copilot_blend_fallback import CopilotBlendFallback
from src.services.copilot_elo_llm_assembler import CopilotEloLlmAssembler
from src.services.copilot_gemini_adapter import CopilotGeminiAdapter
from src.services.copilot_blend_snapshot import write_blend_snapshot
from src.services.copilot_job_repository import CopilotJobRepository
from src.services.copilot_openrouter_adapter import CopilotOpenRouterAdapter


class _BlendAssembler(Protocol):
    def assemble_model_context(
        self,
        *,
        source_weights: Mapping[str, float],
        player_name_contains: str | None = None,
        gameweek: int | None = None,
        bank: float | None = None,
        free_transfers: int | None = None,
        current_squad: list[dict[str, Any]] | None = None,
        fpl_team_id: int | None = None,
    ) -> dict[str, Any]:
        ...


class _LlmAdapter(Protocol):
    def generate_hybrid_payload(
        self,
        *,
        schema_version: str,
        correlation_id: str,
        model_context: dict[str, Any],
    ) -> dict[str, Any]:
        ...


def _stable_json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _build_fpl_id_lookup(model_context: Mapping[str, Any]) -> dict[int, int]:
    lookup: dict[int, int] = {}
    for player in model_context.get("blended_players", []):
        try:
            player_id = int(player["player_id"])
        except (KeyError, TypeError, ValueError):
            continue
        fpl_api_id = player.get("fpl_api_id")
        if fpl_api_id is None:
            continue
        try:
            lookup[player_id] = int(fpl_api_id)
        except (TypeError, ValueError):
            continue
    return lookup


def _apply_airsenal_transfer_deltas(
    hybrid_payload: dict[str, Any],
    *,
    model_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Set projected_points_delta from blended_players AIrsenal xPts (IN − OUT)."""
    blended = model_context.get("blended_players") or []
    xp_by_pid: dict[int, float] = {}
    for p in blended:
        try:
            pid = int(p["player_id"])
        except (KeyError, TypeError, ValueError):
            continue
        xp_by_pid[pid] = float(p.get("airsenal_predicted_points", 0.0) or 0.0)

    for transfer in hybrid_payload.get("recommended_transfers", []):
        out_ref = transfer.get("out") or {}
        in_ref = transfer.get("in") or {}
        out_id = out_ref.get("player_id")
        in_id = in_ref.get("player_id")
        if not isinstance(out_id, int) or not isinstance(in_id, int):
            continue
        delta = xp_by_pid.get(in_id, 0.0) - xp_by_pid.get(out_id, 0.0)
        transfer["projected_points_delta"] = round(delta, 3)
    return hybrid_payload


def _attach_fpl_api_ids(
    hybrid_payload: dict[str, Any],
    *,
    model_context: Mapping[str, Any],
) -> dict[str, Any]:
    lookup = _build_fpl_id_lookup(model_context)
    if not lookup:
        return hybrid_payload

    for transfer in hybrid_payload.get("recommended_transfers", []):
        for side in ("out", "in"):
            player_ref = transfer.get(side)
            if not isinstance(player_ref, dict):
                continue
            player_id = player_ref.get("player_id")
            if not isinstance(player_id, int):
                continue
            fpl_api_id = lookup.get(player_id)
            if fpl_api_id is not None:
                player_ref["fpl_api_id"] = fpl_api_id
    return hybrid_payload


class CopilotJobService:
    def __init__(
        self,
        *,
        repository: CopilotJobRepository,
        assembler: _BlendAssembler,
        gemini_adapter: _LlmAdapter,
        fallback: CopilotBlendFallback | None = None,
    ) -> None:
        self.repository = repository
        self.assembler = assembler
        self.gemini_adapter = gemini_adapter
        self.fallback = fallback or CopilotBlendFallback()

    @classmethod
    def from_dependencies(
        cls,
        *,
        db_path: str,
        repository: CopilotJobRepository | None = None,
        assembler: CopilotEloLlmAssembler | None = None,
        gemini_adapter: _LlmAdapter | None = None,
        fallback: CopilotBlendFallback | None = None,
    ) -> "CopilotJobService":
        job_repository = repository or CopilotJobRepository(db_path)
        blend_assembler = assembler or CopilotEloLlmAssembler(db_path)
        adapter = gemini_adapter
        if adapter is None:
            provider = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()
            if provider == "gemini":
                adapter = CopilotGeminiAdapter()
            elif provider == "openrouter":
                adapter = CopilotOpenRouterAdapter()
            else:
                raise ValueError(
                    "Unsupported LLM_PROVIDER. Expected 'gemini' or 'openrouter'."
                )
        return cls(
            repository=job_repository,
            assembler=blend_assembler,
            gemini_adapter=adapter,
            fallback=fallback,
        )

    def submit_job(
        self, submit_payload: Mapping[str, Any], *, force_refresh: bool = False
    ) -> dict[str, Any]:
        schema_version = str(submit_payload["schema_version"])
        correlation_id = str(submit_payload["correlation_id"])
        job_type = str(submit_payload.get("task", "hybrid"))

        normalized_input = json.loads(_stable_json_dumps(dict(submit_payload)))
        input_hash = hashlib.sha256(_stable_json_dumps(normalized_input).encode("utf-8")).hexdigest()

        if not force_refresh:
            existing_jobs = self.repository.find_jobs_by_input_hash(input_hash)
            for existing in existing_jobs:
                existing_status = existing["status"]
                if existing_status in ("queued", "running"):
                    return {
                        "schema_version": str(existing["schema_version"]),
                        "correlation_id": str(existing["correlation_id"]),
                        "job_id": existing["job_id"],
                        "status": existing_status,
                    }
                if existing_status == "completed":
                    return {
                        "schema_version": str(existing["schema_version"]),
                        "correlation_id": str(existing["correlation_id"]),
                        "job_id": existing["job_id"],
                        "status": "completed",
                        "result_json": existing["result_json"],
                    }

        job_id = f"job-{uuid4().hex}"

        self.repository.create_job(
            job_id=job_id,
            status="queued",
            job_type=job_type,
            input_payload=normalized_input,
            input_hash=input_hash,
            schema_version=schema_version,
            correlation_id=correlation_id,
        )

        return {
            "schema_version": schema_version,
            "correlation_id": correlation_id,
            "job_id": job_id,
            "status": "queued",
        }

    def execute_next_queued_job(self) -> dict[str, Any] | None:
        job = self.repository.claim_next_queued_job()
        if job is None:
            return None

        job_id = str(job["job_id"])
        payload = job["input_json"] or {}

        model_context: dict[str, Any] | None = None

        try:
            source_weights = payload["source_weights"]
            player_name_contains = payload.get("player_name_contains")
            gameweek = payload.get("gameweek")
            bank = payload.get("bank")
            free_transfers = payload.get("free_transfers")
            current_squad = payload.get("current_squad")
            fpl_team_id = payload.get("fpl_team_id")

            model_context = self.assembler.assemble_model_context(
                source_weights=source_weights,
                player_name_contains=player_name_contains,
                gameweek=gameweek,
                bank=bank,
                free_transfers=free_transfers,
                current_squad=current_squad,
                fpl_team_id=fpl_team_id,
            )

            hybrid_payload = self.gemini_adapter.generate_hybrid_payload(
                schema_version=str(job["schema_version"]),
                correlation_id=str(job["correlation_id"]),
                model_context=model_context,
            )
            hybrid_payload = _attach_fpl_api_ids(
                hybrid_payload,
                model_context=model_context,
            )
            hybrid_payload = _apply_airsenal_transfer_deltas(
                hybrid_payload,
                model_context=model_context,
            )

            write_blend_snapshot(
                job_id=job_id,
                schema_version=str(job["schema_version"]),
                correlation_id=str(job["correlation_id"]),
                input_payload=payload,
                result_payload=hybrid_payload,
            )

            self.repository.update_job_status(
                job_id=job_id,
                new_status="completed",
                result_payload=hybrid_payload,
            )
        except Exception as exc:
            if model_context is not None:
                try:
                    hybrid_payload = self.fallback.build_fallback_payload(
                        schema_version=str(job["schema_version"]),
                        correlation_id=str(job["correlation_id"]),
                        model_context=model_context,
                    )
                    hybrid_payload = _attach_fpl_api_ids(
                        hybrid_payload,
                        model_context=model_context,
                    )
                    hybrid_payload = _apply_airsenal_transfer_deltas(
                        hybrid_payload,
                        model_context=model_context,
                    )
                    write_blend_snapshot(
                        job_id=job_id,
                        schema_version=str(job["schema_version"]),
                        correlation_id=str(job["correlation_id"]),
                        input_payload=payload,
                        result_payload=hybrid_payload,
                    )
                    self.repository.update_job_status(
                        job_id=job_id,
                        new_status="completed",
                        result_payload=hybrid_payload,
                    )
                    return self.get_job_status(job_id)
                except Exception:
                    pass

            self.repository.update_job_status(
                job_id=job_id,
                new_status="failed",
                error_payload=self._build_error_payload(job=job, exc=exc),
            )

        return self.get_job_status(job_id)

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        return self.repository.get_job(job_id)

    def _build_error_payload(self, *, job: dict[str, Any], exc: Exception) -> dict[str, Any]:
        code = "JOB_FAILED"
        retryable = False
        if isinstance(exc, TimeoutError):
            code = "LLM_TIMEOUT"
            retryable = True
        elif isinstance(exc, ValueError):
            code = "VALIDATION_ERROR"

        return {
            "schema_version": str(job["schema_version"]),
            "correlation_id": str(job["correlation_id"]),
            "error": {
                "code": code,
                "message": str(exc),
                "retryable": retryable,
                "field_errors": [],
            },
        }
