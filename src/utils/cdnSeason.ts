/**
 * Season-aware resolver for the Premier League asset CDN segment.
 *
 * Player photos / team badges live at
 *   https://resources.premierleague.com/premierleague{SEG}/...
 * where SEG lags the season label and rolls over when PL migrates assets
 * (e.g. "25" during 2025/26 AND at the start of 2026/27, later "26").
 *
 * Resolution order:
 *   1. VITE_FPL_CDN_SEASON env override (pinned, never probed)
 *   2. Fresh cached value in localStorage (probed within TTL)
 *   3. Probe candidate segments newest-first against a known-stable asset
 *      (`placeholder.png`), fall back to the default on any failure.
 *
 * Call `initCdnSeasonSegment()` once before mounting the app; URL builders
 * then read the resolved segment synchronously via `getCdnSeasonSegment()`.
 */

const PROBE_TIMEOUT_MS = 4_000;
const CACHE_TTL_MS = 12 * 60 * 60 * 1000;
const STORAGE_KEY = "fpl_cdn_season_segment";

/** Segment verified live as of Aug 2026 (official FPL site uses it too). */
const DEFAULT_SEGMENT = "25";
/** Segments probed newest-first when no override/fresh cache exists. */
const CANDIDATE_SEGMENTS = ["27", "26", "25"];

let activeSegment: string = DEFAULT_SEGMENT;

const envOverride = (): string | null => {
    const raw = import.meta.env.VITE_FPL_CDN_SEASON as string | undefined;
    const trimmed = raw?.trim();
    return trimmed ? trimmed : null;
};

export const getCdnSeasonSegment = (): string => activeSegment;

export const applyCdnSeasonSegment = (segment: string): void => {
    if (/^\d{2}$/.test(segment)) {
        activeSegment = segment;
    }
};

type CachedEntry = { segment: string; resolvedAt: number };

const readCache = (): string | null => {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return null;
        const entry = JSON.parse(raw) as CachedEntry;
        if (
            typeof entry?.segment === "string" &&
            typeof entry?.resolvedAt === "number" &&
            Date.now() - entry.resolvedAt < CACHE_TTL_MS
        ) {
            return entry.segment;
        }
    } catch {
        // corrupted cache – ignore and re-probe
    }
    return null;
};

const writeCache = (segment: string): void => {
    try {
        const entry: CachedEntry = { segment, resolvedAt: Date.now() };
        localStorage.setItem(STORAGE_KEY, JSON.stringify(entry));
    } catch {
        // storage unavailable (private mode etc.) – non-fatal
    }
};

const probeSegment = async (segment: string): Promise<boolean> => {
    const url =
        `https://resources.premierleague.com/premierleague${segment}` +
        `/photos/players/110x140/placeholder.png`;
    try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), PROBE_TIMEOUT_MS);
        const res = await fetch(url, { method: "HEAD", signal: controller.signal });
        clearTimeout(timer);
        return res.ok;
    } catch {
        return false;
    }
};

/**
 * Resolve + activate the best CDN segment. Never throws; on total failure the
 * default segment stays active. Resolves quickly when an override or fresh
 * cached value exists (no network).
 */
export const initCdnSeasonSegment = async (): Promise<string> => {
    const override = envOverride();
    if (override) {
        applyCdnSeasonSegment(override);
        return activeSegment;
    }

    const cached = readCache();
    if (cached) {
        applyCdnSeasonSegment(cached);
        return activeSegment;
    }

    for (const candidate of CANDIDATE_SEGMENTS) {
        if (await probeSegment(candidate)) {
            applyCdnSeasonSegment(candidate);
            writeCache(candidate);
            return activeSegment;
        }
    }
    return activeSegment;
};
