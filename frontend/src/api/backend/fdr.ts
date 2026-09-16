import { backendFetch } from "./client";

/** Single fixture block from GET /api/fdr/team/{team} (matches FastAPI models). */
export interface FdrFixtureSaturated {
  fixture_id?: number | null;
  gameweek?: number | null;
  official_fpl_fdr?: number | null;
  official_fpl_event?: number | null;
  official_fpl_source?: string | null;
}

export interface FdrFixtureMetrics {
  overall_fdr?: number;
  attack_fdr?: number;
  defence_fdr?: number;
}

export interface TeamFdrFixtureItem {
  saturated: FdrFixtureSaturated & Record<string, unknown>;
  fdr: FdrFixtureMetrics & Record<string, unknown>;
}

export interface FdrEloRatingRow {
  team_id: number;
  team: string;
  elo: number;
}

export interface FdrEloSnapshot {
  snapshot_date: string;
  ratings: FdrEloRatingRow[];
}

/**
 * ClubElo snapshot for current PL teams (GET /api/fdr/elo).
 */
export async function getFdrEloSnapshot(snapshotDate?: string): Promise<FdrEloSnapshot> {
  const q = snapshotDate ? `?snapshot_date=${encodeURIComponent(snapshotDate)}` : "";
  return backendFetch<FdrEloSnapshot>(`/api/fdr/elo${q}`);
}

/**
 * Next N fixtures with official FPL + custom FDR (GET /api/fdr/team/{team_name}).
 * Backend accepts numeric FPL team id or readable name.
 */
export async function getTeamFdrFixtures(
  teamRef: string | number,
  next: number,
  snapshotDate?: string,
): Promise<TeamFdrFixtureItem[]> {
  const ref = typeof teamRef === "number" ? String(teamRef) : teamRef;
  const params = new URLSearchParams({ next: String(next) });
  if (snapshotDate) params.set("snapshot_date", snapshotDate);
  const path = `/api/fdr/team/${encodeURIComponent(ref)}?${params.toString()}`;
  return backendFetch<TeamFdrFixtureItem[]>(path);
}
