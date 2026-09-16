import { backendFetch } from "./client";
import type { FixturesRatingsResponse } from "../../types/fixturesRatings";

/**
 * Fixture difficulty matrices + per-team Elo / Elo-FDR.
 * Query params are optional hints for the backend cache.
 */
export async function getFixturesRatings(params?: {
  fromGameweek?: number;
  weeks?: number;
}): Promise<FixturesRatingsResponse> {
  const search = new URLSearchParams();
  if (params?.fromGameweek != null) {
    search.set("from_gw", String(params.fromGameweek));
  }
  if (params?.weeks != null) {
    search.set("weeks", String(params.weeks));
  }
  const q = search.toString();
  const path = `/api/fixtures/ratings${q ? `?${q}` : ""}`;
  return backendFetch<FixturesRatingsResponse>(path);
}
