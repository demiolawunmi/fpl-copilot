/**
 * Season status API - compares the live FPL season against local data snapshots.
 */
import { backendFetch } from "./client";

// ── Types ──

export interface SeasonStatus {
  /** Season per the live FPL feed, e.g. "2627" for 2026/27. */
  fpl_season: string | null;
  /** Season of locally exported data (teams.json / AIrsenal DB). */
  data_season: string | null;
  /** True when local snapshots match the live season. */
  is_current: boolean;
  checked_at: string;
}

// ── API Call ──

export async function getSeasonStatus(): Promise<SeasonStatus> {
  return backendFetch<SeasonStatus>("/api/season/status");
}
