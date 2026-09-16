import { fetchJson } from "./client";
import { fplEndpoints } from "./endpoints";
import { cached } from "../../lib/cache";

// -------------------------
// Minimal FPL response types
// -------------------------

export type FplEntry = {
    name: string;
    player_first_name: string;
    player_last_name: string;
};

export type FplBootstrapElement = {
    id: number;
    code: number;
    web_name: string;
    first_name?: string;
    second_name?: string;
    element_type: number;
    team: number;
    status?: string;
    now_cost?: number;
    total_points?: number;
    minutes?: number;
    starts?: number;
    form?: string;
    points_per_game?: string;
    selected_by_percent?: string;
    goals_scored?: number;
    assists?: number;
    clean_sheets?: number;
    goals_conceded?: number;
    own_goals?: number;
    penalties_saved?: number;
    penalties_missed?: number;
    yellow_cards?: number;
    red_cards?: number;
    saves?: number;
    bonus?: number;
    bps?: number;
    expected_goals?: string;
    expected_assists?: string;
    expected_goal_involvements?: string;
    expected_goals_conceded?: string;
    goals_scored_per_90?: string;
    assists_per_90?: string;
    expected_goals_per_90?: string;
    expected_assists_per_90?: string;
    expected_goal_involvements_per_90?: string;
    expected_goals_conceded_per_90?: string;
    clean_sheets_per_90?: string;
    saves_per_90?: string;
    influence?: string;
    creativity?: string;
    threat?: string;
    ict_index?: string;
    chance_of_playing_next_round?: number | null;
    chance_of_playing_this_round?: number | null;
    news?: string;
    news_added?: string | null;
    ep_this?: string | null;
    ep_next?: string | null;
    transfers_in?: number;
    transfers_out?: number;
    transfers_in_event?: number;
    transfers_out_event?: number;
    value_form?: string;
    value_season?: string;
    cost_change_start?: number;
    cost_change_event?: number;
    dreamteam_count?: number;
    in_dreamteam?: boolean;
};

export type FplBootstrapElementType = {
    id: number;
    singular_name_short: "GKP" | "DEF" | "MID" | "FWD";
    singular_name?: string;
    plural_name?: string;
};

export type FplBootstrap = {
    elements: FplBootstrapElement[];
    teams: Array<{
        id: number;
        code: number;
        name: string;
        short_name: string;
    }>;
    element_types: FplBootstrapElementType[];
};

export type FplPicks = {
    picks: Array<{
        element: number;
        position: number;
        multiplier: number;
        is_captain: boolean;
        is_vice_captain: boolean;
    }>;
};

// -------------------------
// Element Summary types
// -------------------------

export type ElementSummaryFixture = {
    id: number;
    code: number;
    team_h: number;
    team_a: number;
    event: number | null;
    finished: boolean;
    minutes: number;
    provisional_start_time: boolean;
    kickoff_time: string | null;
    event_name: string | null;
    is_home: boolean;
    difficulty: number;
    /** Often omitted on live API; derive from `team_h` / `team_a` and the player’s `team` id. */
    opponent_team?: number;
};

export type ElementSummaryHistory = {
    element: number;
    fixture: number;
    opponent_team: number;
    total_points: number;
    was_home: boolean;
    kickoff_time: string | null;
    team_h_score: number | null;
    team_a_score: number | null;
    round: number;
    minutes: number;
    goals_scored: number;
    assists: number;
    clean_sheets: number;
    goals_conceded: number;
    own_goals: number;
    penalties_saved: number;
    penalties_missed: number;
    yellow_cards: number;
    red_cards: number;
    saves: number;
    bonus: number;
    bps: number;
    influence: string;
    creativity: string;
    threat: string;
    ict_index: string;
    starts: number;
    expected_goals: string;
    expected_assists: string;
    expected_goal_involvements: string;
    expected_goals_conceded: string;
    value: number;
    transfers_balance: number;
    selected: number;
    transfers_in: number;
    transfers_out: number;
};

export type ElementSummaryHistoryPast = {
    season_name: string;
    element_code: number;
    start_cost: number;
    end_cost: number;
    total_points: number;
    minutes: number;
    goals_scored: number;
    assists: number;
    clean_sheets: number;
    goals_conceded: number;
    own_goals: number;
    penalties_saved: number;
    penalties_missed: number;
    yellow_cards: number;
    red_cards: number;
    saves: number;
    bonus: number;
    bps: number;
    influence: string;
    creativity: string;
    threat: string;
    ict_index: string;
    starts: number;
    expected_goals: string;
    expected_assists: string;
    expected_goal_involvements: string;
    expected_goals_conceded: string;
};

export type ElementSummaryResponse = {
    fixtures: ElementSummaryFixture[];
    history: ElementSummaryHistory[];
    history_past: ElementSummaryHistoryPast[];
};

// -------------------------
// API calls
// -------------------------

export async function getEntry(teamId: string) {
    return cached(`fpl:entry:${teamId}`, CACHE_TTL_MS, () =>
        fetchJson<FplEntry>(fplEndpoints.entry(teamId)),
    );
}

const CACHE_TTL_MS = 5 * 60 * 1000;

export async function getBootstrap() {
    return cached("fpl:bootstrap", CACHE_TTL_MS, () =>
        fetchJson<FplBootstrap>(fplEndpoints.bootstrap()),
    );
}

export type FplFixtureApi = {
    id?: number;
    event: number | null;
    team_h: number;
    team_a: number;
    kickoff_time: string | null;
    finished: boolean;
    team_h_score: number | null;
    team_a_score: number | null;
    team_h_difficulty?: number | null;
    team_a_difficulty?: number | null;
};

export async function getFixtures(): Promise<FplFixtureApi[]> {
    return cached("fpl:fixtures", CACHE_TTL_MS, () =>
        fetchJson<FplFixtureApi[]>(fplEndpoints.fixtures()),
    );
}

export async function getPicks(teamId: string, gw: number) {
    return fetchJson<FplPicks>(fplEndpoints.entryPicks(teamId, gw));
}

export async function getElementSummary(playerId: string | number) {
    return fetchJson<ElementSummaryResponse>(fplEndpoints.elementSummary(playerId));
}

// -------------------------
// Helpers
// -------------------------

export function elementTypeToPosition(t: number): "GK" | "DEF" | "MID" | "FWD" {
    if (t === 1) return "GK";
    if (t === 2) return "DEF";
    if (t === 3) return "MID";
    return "FWD";
}

export function getPlayerPhotoUrl(code: number | string, size: "110x140" | "250x250" | "60x60" = "110x140") {
    return fplEndpoints.playerPhoto(code, size);
}

// -------------------------
// Optional convenience: build squad for your UI
// -------------------------

export type UiPlayer = {
    id: number;
    code: number;
    name: string;
    position: "GK" | "DEF" | "MID" | "FWD";
    isCaptain?: boolean;
    isViceCaptain?: boolean;
    isBench?: boolean;
    multiplier?: number;
    photoUrl?: string;
    teamId?: number;
    teamAbbr?: string;
};

export function buildUiSquadFromPicks(
    picks: FplPicks,
    bootstrap: FplBootstrap
): UiPlayer[] {
    const byId = new Map(bootstrap.elements.map((e) => [e.id, e]));
    const teamsById = new Map(bootstrap.teams.map((t) => [t.id, t]));

    return picks.picks
        .slice()
        .sort((a, b) => a.position - b.position)
        .map((p) => {
            const el = byId.get(p.element);

            const id = el?.id ?? p.element;
            const code = el?.code ?? 0;
            const playerTeamId = el?.team;
            const team = playerTeamId ? teamsById.get(playerTeamId) : undefined;

            return {
                id,
                code,
                name: el?.web_name ?? `#${p.element}`,
                position: el ? elementTypeToPosition(el.element_type) : "MID",
                isCaptain: p.is_captain,
                isViceCaptain: p.is_vice_captain,
                isBench: p.position > 11,
                multiplier: p.multiplier,
                photoUrl: code ? getPlayerPhotoUrl(code, "110x140") : undefined,
                teamId: playerTeamId,
                teamAbbr: team?.short_name,
            };
        });
}
