from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Protocol
from uuid import uuid4

from src.services.copilot_blend_fallback import CopilotBlendFallback
from src.services.copilot_elo_llm_assembler import CopilotEloLlmAssembler
from src.services.copilot_gemini_adapter import CopilotGeminiAdapter
from src.services.copilot_job_repository import CopilotJobRepository


class _BlendAssembler(Protocol):
    def assemble_model_context(
        self,
        *,
        source_weights: Mapping[str, float],
        player_name_contains: str | None = None,
        gameweek: int | None = None,
    ) -> dict[str, Any]:
        ...


class _GeminiAdapter(Protocol):
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


class CopilotJobService:
    def __init__(
        self,
        *,
        repository: CopilotJobRepository,
        assembler: _BlendAssembler,
        gemini_adapter: _GeminiAdapter,
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
        gemini_adapter: CopilotGeminiAdapter | None = None,
        fallback: CopilotBlendFallback | None = None,
    ) -> "CopilotJobService":
        job_repository = repository or CopilotJobRepository(db_path)
        blend_assembler = assembler or CopilotEloLlmAssembler(db_path)
        adapter = gemini_adapter or CopilotGeminiAdapter()
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

            model_context = self.assembler.assemble_model_context(
                source_weights=source_weights,
                player_name_contains=player_name_contains,
                gameweek=gameweek,
            )

            hybrid_payload = self.gemini_adapter.generate_hybrid_payload(
                schema_version=str(job["schema_version"]),
                correlation_id=str(job["correlation_id"]),
                model_context=model_context,
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
