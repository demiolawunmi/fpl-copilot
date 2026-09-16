/**
 * Payload shape for backend fixture difficulty + ELO ratings.
 * Backend can populate gradually; nulls render as placeholders.
 */
export interface TeamFixtureRatingsRow {
  shortName: string;
  /** Team Elo (or similar strength metric). */
  elo: number | null;
  /** Optional single-number custom FDR from your model (when not using per-GW cells). */
  eloFdrSummary?: number | null;
  /** Official FPL fixture difficulty (1–5) per gameweek column, aligned with gameweekIds. */
  officialFdr: (number | null)[];
  /** Custom FDR derived from Elo matchups (1–5), aligned with gameweekIds. */
  eloBasedFdr: (number | null)[];
}

export interface FixturesRatingsResponse {
  gameweekIds: number[];
  teams: TeamFixtureRatingsRow[];
}
