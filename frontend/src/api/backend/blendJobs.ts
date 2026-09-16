import { backendFetch, isApiError, type ApiError } from "./client";

export type CopilotBlendJobStatus = "queued" | "running" | "completed" | "failed";

export interface CopilotSourceWeights {
  elo: number;
  airsenal: number;
}

export interface CopilotBlendSubmitRequest {
  schema_version: string;
  correlation_id: string;
  source_weights: CopilotSourceWeights;
  gameweek?: number;
  bank?: number;
  free_transfers?: number;
  current_squad?: CopilotCurrentSquadPlayer[];
  /** FPL entry / team id (same as gw_*_optimization_<id>.json suffix) */
  fpl_team_id?: number;
  task: "hybrid";
  force_refresh?: boolean;
}

export interface CopilotCurrentSquadPlayer {
  fpl_api_id: number;
  player_name: string;
  team: string;
  position: "GK" | "DEF" | "MID" | "FWD";
  price: number;
  x_pts: number;
}

export interface CopilotBlendSubmitAcceptedResponse {
  schema_version: string;
  correlation_id: string;
  job_id: string;
  status: "queued";
}

export interface CopilotHybridCore {
  summary: string;
  confidence: number;
}

export interface CopilotTransferPlayerRef {
  player_id: number;
  fpl_api_id?: number;
  player_name: string;
}

export interface CopilotRecommendedTransfer {
  transfer_id: string;
  out: CopilotTransferPlayerRef;
  in: CopilotTransferPlayerRef;
  reason: string;
  projected_points_delta: number;
}

export interface CopilotAskCopilotResponse {
  answer: string;
  rationale: string[];
  confidence: number;
}

export interface CopilotDegradedMode {
  is_degraded: boolean;
  code?: "LLM_TIMEOUT" | "SCHEMA_VALIDATION_FAILED" | "PROVIDER_ERROR" | "FALLBACK";
  message?: string;
  fallback_used: boolean;
}

export interface CopilotHybridResultPayload {
  schema_version: string;
  correlation_id: string;
  core: CopilotHybridCore;
  recommended_transfers: CopilotRecommendedTransfer[];
  ask_copilot: CopilotAskCopilotResponse;
  degraded_mode: CopilotDegradedMode;
}

export interface CopilotErrorField {
  field: string;
  message: string;
}

export interface CopilotErrorDetail {
  code:
    | "VALIDATION_ERROR"
    | "NOT_FOUND"
    | "JOB_FAILED"
    | "LLM_TIMEOUT"
    | "SCHEMA_VALIDATION_FAILED";
  message: string;
  retryable: boolean;
  field_errors: CopilotErrorField[];
}

export interface CopilotErrorResponse {
  schema_version: string;
  correlation_id: string;
  error: CopilotErrorDetail;
}

export interface CopilotBlendJobStatusResponse {
  schema_version: string;
  correlation_id: string;
  job_id: string;
  status: CopilotBlendJobStatus;
  result: CopilotHybridResultPayload | null;
  error: CopilotErrorResponse | null;
}

function normalizeApiError(error: unknown, method: string, path: string): ApiError {
  if (isApiError(error)) {
    return error;
  }

  return {
    name: "ApiError",
    message: error instanceof Error ? error.message : "Backend request failed",
    status: 0,
    statusText: "NETWORK_ERROR",
    method,
    path,
    details: error,
  };
}

export async function submitCopilotBlendJob(
  body: CopilotBlendSubmitRequest,
): Promise<CopilotBlendSubmitAcceptedResponse> {
  const path = "/api/copilot/blend-jobs";
  try {
    return await backendFetch<CopilotBlendSubmitAcceptedResponse>(path, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });
  } catch (error) {
    throw normalizeApiError(error, "POST", path);
  }
}

export async function getCopilotBlendJobStatus(
  jobId: string,
): Promise<CopilotBlendJobStatusResponse> {
  const path = `/api/copilot/blend-jobs/${encodeURIComponent(jobId)}`;
  try {
    return await backendFetch<CopilotBlendJobStatusResponse>(path);
  } catch (error) {
    throw normalizeApiError(error, "GET", path);
  }
}

export const pollCopilotBlendJob = getCopilotBlendJobStatus;

export async function getCopilotBlendJobResult(
  jobId: string,
): Promise<CopilotHybridResultPayload | null> {
  const status = await getCopilotBlendJobStatus(jobId);
  return status.status === "completed" ? status.result : null;
}

/** Persisted by backend to data/api after each successful blend; served via GET /api/files/… */
export interface CopilotBlendSnapshot {
  saved_at: string;
  job_id: string;
  correlation_id: string;
  schema_version: string;
  input: CopilotBlendSubmitRequest & Record<string, unknown>;
  result: CopilotHybridResultPayload;
}

/** File on disk: ``data/api/gw_<gw>_copilot_blend_<fplTeamId>.json`` */
export function copilotBlendSnapshotPath(gameweek: number, fplTeamId: number): string {
  return `/api/files/gw_${gameweek}_copilot_blend_${fplTeamId}`;
}

/** When no ``fpl_team_id`` was sent: ``gw_<gw>_copilot_blend.json`` */
export function copilotBlendSnapshotGlobalPath(gameweek: number): string {
  return `/api/files/gw_${gameweek}_copilot_blend`;
}

/**
 * Load the last saved blend for this gameweek + FPL entry team id.
 * Returns null if no file exists (404).
 */
export async function getCopilotBlendSnapshot(
  gameweek: number,
  fplTeamId: number,
): Promise<CopilotBlendSnapshot | null> {
  const path = copilotBlendSnapshotPath(gameweek, fplTeamId);
  try {
    return await backendFetch<CopilotBlendSnapshot>(path);
  } catch (error) {
    if (isApiError(error) && error.status === 404) {
      return null;
    }
    throw normalizeApiError(error, "GET", path);
  }
}

/** Fallback when the job had no fpl_team_id: ``gw_<gw>_copilot_blend.json`` */
export async function getCopilotBlendSnapshotGlobal(
  gameweek: number,
): Promise<CopilotBlendSnapshot | null> {
  const path = copilotBlendSnapshotGlobalPath(gameweek);
  try {
    return await backendFetch<CopilotBlendSnapshot>(path);
  } catch (error) {
    if (isApiError(error) && error.status === 404) {
      return null;
    }
    throw normalizeApiError(error, "GET", path);
  }
}

export interface CopilotChatTurn {
  role: "user" | "assistant";
  content: string;
}

export interface CopilotChatRequest {
  schema_version: string;
  correlation_id: string;
  message: string;
  messages: CopilotChatTurn[];
  blend_input: CopilotBlendSubmitRequest;
  blend_result: CopilotHybridResultPayload;
}

export interface CopilotChatResponse {
  schema_version: string;
  correlation_id: string;
  answer: string;
}

export async function postCopilotChat(
  body: CopilotChatRequest,
): Promise<CopilotChatResponse> {
  const path = "/api/copilot/chat";
  try {
    return await backendFetch<CopilotChatResponse>(path, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });
  } catch (error) {
    throw normalizeApiError(error, "POST", path);
  }
}
