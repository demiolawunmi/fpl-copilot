import { getCdnSeasonSegment } from '../../utils/cdnSeason';

/**
 * FPL JSON API base path.
 * - Dev: `/fpl-api` is proxied by Vite → `https://fantasy.premierleague.com/api` (avoids CORS during local dev).
 * - Prod / preview: use the real origin so `/fpl-api/...` is not requested from your static host (which 404s).
 * Override with `VITE_FPL_API_BASE` if you front the API with your own proxy.
 */
const FPL_API_BASE = (() => {
  const env = import.meta.env.VITE_FPL_API_BASE as string | undefined;
  if (env != null && env.trim() !== "") {
    return env.replace(/\/$/, "");
  }
  if (import.meta.env.DEV) {
    return "/fpl-api";
  }
  return "https://fantasy.premierleague.com/api";
})();

const BASE = FPL_API_BASE;

export type PlayerPhotoSize = "110x140" | "250x250" | "60x60";

// Season segment resolved at startup by `initCdnSeasonSegment()` (see utils/cdnSeason.ts).
const plResources = () =>
    `https://resources.premierleague.com/premierleague${getCdnSeasonSegment()}/photos/players`;
const plBadges = () =>
    `https://resources.premierleague.com/premierleague${getCdnSeasonSegment()}/badges`;

export const fplEndpoints = {
    // Core API
    bootstrap: () => `${BASE}/bootstrap-static/`,
    entry: (teamId: string | number) => `${BASE}/entry/${teamId}/`,
    entryHistory: (teamId: string | number) => `${BASE}/entry/${teamId}/history/`,
    entryPicks: (teamId: string | number, gw: number) => `${BASE}/entry/${teamId}/event/${gw}/picks/`,
    fixtures: (gw?: number) => (gw ? `${BASE}/fixtures/?event=${gw}` : `${BASE}/fixtures/`),
    elementSummary: (playerId: string | number) => `${BASE}/element-summary/${playerId}/`,
    liveEvent: (gw: number) => `${BASE}/event/${gw}/live/`,

    // Player photos (use `code`, NOT `id`)
    playerPhoto: (code: number | string, size: PlayerPhotoSize = "110x140") =>
        `${plResources()}/${size}/${code}.png`,

    /** Grey-silhouette avatar shown when a player photo is missing (404). */
    playerPlaceholder: (size: PlayerPhotoSize = "110x140") =>
        `${plResources()}/${size}/placeholder.png`,

    // Team badges (use team `code`, NOT `id`) — PL CDN uses `/badges/{code}.svg` (no size segment).
    teamBadge: (teamCode: number | string) => `${plBadges()}/${teamCode}.svg`,
};
