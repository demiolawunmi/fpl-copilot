/**
 * AIrsenal API endpoints - predictions, form data, and fixtures
 */
import { backendFetch, extractArrayPayload } from "./client";

// ── Form Last 4 Types ──

export interface FormLast4Player {
  player_id?: number;
  element?: number;
  code?: number;
  player_name?: string;
  name?: string;
  web_name?: string;
  team?: string;
  team_short_name?: string;
  team_name?: string;
  last4_points?: number;
  last_4_points?: number;
  points_last4?: number;
  last4_minutes?: number;
  last_4_minutes?: number;
  minutes_last4?: number;
  last4_xgi?: number;
  last4_xgc?: number;
  xG?: number;
  xg?: number;
  expected_goals?: number;
  xA?: number;
  xa?: number;
  expected_assists?: number;
  photo_url?: string;
}

export type FormLast4Response = FormLast4Player[];

// ── Predictions Types ──

export interface PredictionPlayer {
  player_id?: number;
  id?: number;
  element?: number;
  player_name?: string;
  name?: string;
  web_name?: string;
  team?: string;
  team_short_name?: string;
  team_name?: string;
  position?: string;
  xp?: number; // expected points
  expected_points?: number;
}

export type PredictionsResponse = PredictionPlayer[];

// ── Fixtures by Player Types ──

export interface PlayerFixture {
  player_id: number;
  player_name: string;
  team: string;
  fixtures: Array<{
    opponent: string;
    opponent_short: string;
    is_home: boolean;
    difficulty?: number; // 1-5 scale (if available)
    kickoff_time?: string;
  }>;
}

export type FixturesByPlayerResponse = PlayerFixture[];

// ── API Calls ──

/**
 * Get form data for last 4 gameweeks
 */
export async function getFormLast4(): Promise<FormLast4Response> {
  const payload = await backendFetch<unknown>("/api/airsenal/form_last4");
  return extractArrayPayload<FormLast4Player>(payload, ["data", "items", "results", "players", "form_last4"]);
}

/**
 * Get AIrsenal predictions for a specific gameweek
 */
export async function getPredictions(gw: number): Promise<PredictionsResponse> {
  const payload = await backendFetch<unknown>(`/api/airsenal/gw/${gw}/predictions`);
  return extractArrayPayload<PredictionPlayer>(payload, ["data", "items", "results", "predictions", "players"]);
}

/**
 * Get fixtures by player for a specific gameweek
 */
export async function getFixturesByPlayer(gw: number): Promise<FixturesByPlayerResponse> {
  const payload = await backendFetch<unknown>(`/api/airsenal/gw/${gw}/fixtures_by_player`);
  return extractArrayPayload<PlayerFixture>(payload, ["data", "items", "results", "fixtures_by_player", "players"]);
}

// ── AIrsenal CLI run (POST) — long-running; server default timeout 3600s ──

export type AirsenalRunAction = "update_db" | "predict" | "optimize" | "export" | "pipeline";

export interface AirsenalRunRequest {
  action: AirsenalRunAction;
  /** 1–38 */
  weeks_ahead?: number;
  gameweek?: string | number;
  fpl_team_id?: number | null;
}

export interface AirsenalRunResponse {
  ok: boolean;
  action: string;
  steps: unknown[];
}

const defaultRunPath = "/api/airsenal/run";

function airsenalRunTimeoutMs(): number {
  const raw = import.meta.env.VITE_AIRSENAL_RUN_TIMEOUT_SEC as string | undefined;
  const sec = raw != null && raw !== "" ? Number(raw) : 3600;
  return Math.max(1, Number.isFinite(sec) ? sec : 3600) * 1000;
}

/**
 * Run an AIrsenal CLI action on the backend (same env as scripts/airsenal.sh).
 * Use a generous client timeout (default 3600s); override with VITE_AIRSENAL_RUN_TIMEOUT_SEC.
 * When the backend expects a key, set VITE_AIRSENAL_RUN_KEY (sent as X-Airsenal-Run-Key).
 * Override path with VITE_AIRSENAL_RUN_PATH if your API route differs.
 */
export async function runAirsenal(body: AirsenalRunRequest): Promise<AirsenalRunResponse> {
  const path =
    (import.meta.env.VITE_AIRSENAL_RUN_PATH as string | undefined)?.trim() || defaultRunPath;
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  const key = (import.meta.env.VITE_AIRSENAL_RUN_KEY as string | undefined)?.trim();
  if (key) {
    headers["X-Airsenal-Run-Key"] = key;
  }

  const ms = airsenalRunTimeoutMs();
  const signal =
    typeof AbortSignal !== "undefined" && typeof AbortSignal.timeout === "function"
      ? AbortSignal.timeout(ms)
      : (() => {
          const c = new AbortController();
          setTimeout(() => c.abort(), ms);
          return c.signal;
        })();

  return backendFetch<AirsenalRunResponse>(path, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
    signal,
  });
}
