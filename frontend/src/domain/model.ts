import type { FplBootstrap, FplBootstrapElement } from '../api/fpl/fpl';
import type { PredictionPlayer } from '../api/backend/airsenal';
import type { FormLast4Player } from '../api/backend/airsenal';
import type { BandwagonPlayer } from '../api/backend/bandwagons';
import type { InjuryNewsPlayer } from '../api/backend/injuryNews';
import type {
  Club,
  FixtureLite,
  Outlook,
  Player,
  PlayerStatus,
  Position,
  TeamGwFixture,
} from './types';
import { clubColor } from './clubs';
import { norm, round1, toNum } from '../lib/format';

/* ------------------------------------------------------------------ raw types */

export interface RawFplFixture {
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
}

export interface PlayerSources {
  predictions?: PredictionPlayer[];
  form?: FormLast4Player[];
  bandwagons?: BandwagonPlayer[];
  injuries?: InjuryNewsPlayer[];
}

/* --------------------------------------------------------------------- clubs */

export function buildClubs(bootstrap: FplBootstrap): Club[] {
  return bootstrap.teams.map((t) => ({
    id: t.id,
    code: t.code,
    name: t.name,
    short: t.short_name,
    color: clubColor(t.short_name),
  }));
}

export function clubsById(clubs: Club[]): Map<number, Club> {
  return new Map(clubs.map((c) => [c.id, c]));
}

/* ------------------------------------------------------------------- helpers */

const toPosition = (elementType: number): Position => {
  if (elementType === 1) return 'GK';
  if (elementType === 2) return 'DEF';
  if (elementType === 3) return 'MID';
  return 'FWD';
};

const toStatus = (raw: string | undefined): PlayerStatus => {
  if (raw === 'd') return 'd';
  if (raw === 'i' || raw === 'u' || raw === 'n') return 'i';
  if (raw === 's') return 's';
  return 'a';
};

const deriveStatus = (
  el: FplBootstrapElement,
  injury: InjuryNewsPlayer | undefined,
): PlayerStatus => {
  const base = toStatus(el.status);
  if (!injury) return base;
  const absence = injury.absence_type ?? injury.status ?? '';
  if (absence === 'suspension' || injury.status === 'suspended') return 's';
  if (absence === 'doubtful' || absence === 'questionable' || absence === 'major_doubt') return 'd';
  if (absence === 'injury' || absence === 'loan_out' || absence === 'transfer_out' || injury.status === 'out') {
    return 'i';
  }
  return base;
};

const per90 = (total: number, minutes: number): number =>
  minutes > 0 ? round1((total / minutes) * 90) : 0;

const firstBy = <T,>(items: T[] | undefined, key: (item: T) => string): Map<string, T> => {
  const map = new Map<string, T>();
  for (const item of items ?? []) {
    const k = key(item);
    if (k && !map.has(k)) map.set(k, item);
  }
  return map;
};

const nameTeamKey = (name: string | undefined, team: string | undefined): string =>
  `${norm(name)}|${norm(team)}`;

/* ------------------------------------------------------------------- players */

export function buildPlayers(
  bootstrap: FplBootstrap,
  clubs: Club[],
  sources: PlayerSources = {},
): Player[] {
  const clubById = clubsById(clubs);
  const clubByShort = new Map(clubs.map((c) => [norm(c.short), c]));

  const predByKey = firstBy(sources.predictions, (p) =>
    nameTeamKey(p.player_name ?? p.name, p.team ?? p.team_short_name),
  );
  const formByKey = firstBy(sources.form, (p) =>
    nameTeamKey(p.player_name ?? p.name ?? p.web_name, p.team ?? p.team_short_name),
  );
  const formByName = firstBy(sources.form, (p) =>
    norm(p.player_name ?? p.name ?? p.web_name),
  );
  const bwByKey = firstBy(sources.bandwagons, (p) =>
    nameTeamKey(p.player_name ?? p.name ?? p.web_name, p.team ?? p.team_short_name),
  );
  const injuryByKey = firstBy(sources.injuries, (p) =>
    nameTeamKey(p.name ?? p.player_name ?? p.web_name, p.team ?? p.team_short_name),
  );
  const injuryByName = firstBy(sources.injuries, (p) =>
    norm(p.name ?? p.player_name ?? p.web_name),
  );

  return bootstrap.elements.map((el: FplBootstrapElement) => {
    const club = clubById.get(el.team) ?? {
      id: el.team,
      code: 0,
      name: `Team ${el.team}`,
      short: '—',
      color: '#64748F',
    };
    const clubFromShort = clubByShort.get(norm(club.short));
    const resolvedClub = clubFromShort ?? club;

    const name = el.web_name;
    const teamShort = resolvedClub.short;
    const key = nameTeamKey(name, teamShort);

    const pred = predByKey.get(key);
    const form = formByKey.get(key) ?? formByName.get(norm(name));
    const bw = bwByKey.get(key);
    const injury = injuryByKey.get(key) ?? injuryByName.get(norm(name));

    const mins = toNum(el.minutes);
    const pts = toNum(el.total_points);
    const g = toNum(el.goals_scored);
    const a = toNum(el.assists);
    const xg = toNum(el.expected_goals);
    const xa = toNum(el.expected_assists);
    const xgi = toNum(el.expected_goal_involvements);
    const xgc = toNum(el.expected_goals_conceded);
    const saves = toNum(el.saves);
    const goalsConceded = toNum(el.goals_conceded);
    const saveRate = saves + goalsConceded > 0
      ? Math.round((saves / (saves + goalsConceded)) * 100)
      : 0;

    const status = deriveStatus(el, injury);
    const chance =
      injury?.prob_available != null
        ? Math.round(toNum(injury.prob_available) * 100)
        : injury?.chance_of_playing_next_round ?? injury?.chance_next_round ?? (
            el.chance_of_playing_next_round == null
              ? status === 'a' ? 100 : status === 'd' ? 50 : 0
              : toNum(el.chance_of_playing_next_round)
          );
    const news = injury?.source_news || injury?.news || el.news || '';
    const absenceType = (injury?.absence_type as string | undefined) ?? null;

    const epNext = toNum(el.ep_next);
    const epThis = toNum(el.ep_this);
    let xpts = 0;
    let xptsSource: Player['xptsSource'] = 'estimate';
    if (pred) {
      xpts = toNum(pred.xp ?? pred.expected_points);
      xptsSource = 'airsenal';
    } else if (epNext > 0) {
      xpts = epNext;
      xptsSource = 'fpl';
    } else if (epThis > 0) {
      xpts = epThis;
      xptsSource = 'fpl';
    } else {
      xpts = round1(Math.max(0, toNum(el.points_per_game) * 0.9));
      xptsSource = 'estimate';
    }
    if (status !== 'a') xpts = round1(xpts * (chance / 100));

    const transfersIn = bw ? toNum(bw.transfers_in) : toNum(el.transfers_in_event);
    const transfersOut = bw ? toNum(bw.transfers_out) : toNum(el.transfers_out_event);

    return {
      id: el.id,
      code: el.code,
      name,
      fullName: [el.first_name, el.second_name].filter(Boolean).join(' ') || name,
      pos: toPosition(el.element_type),
      teamId: resolvedClub.id,
      club: resolvedClub,
      price: toNum(el.now_cost) / 10,
      own: toNum(el.selected_by_percent),
      mins,
      pts,
      form: toNum(el.form),
      status,
      chance,
      news,
      absenceType,
      g,
      a,
      xg,
      xa,
      xgi,
      cs: toNum(el.clean_sheets),
      saves,
      saves90: toNum(el.saves_per_90, per90(saves, mins)),
      gc: goalsConceded,
      penaltiesSaved: toNum(el.penalties_saved),
      bonus: toNum(el.bonus),
      threat: toNum(el.threat),
      creativity: toNum(el.creativity),
      ict: toNum(el.ict_index),
      xgc,
      saveRate,
      ppg: toNum(el.points_per_game),
      pts90: per90(pts, mins),
      g90: per90(g, mins),
      a90: per90(a, mins),
      xg90: toNum(el.expected_goals_per_90, per90(xg, mins)),
      xa90: toNum(el.expected_assists_per_90, per90(xa, mins)),
      xgi90: toNum(el.expected_goal_involvements_per_90, per90(xgi, mins)),
      threat90: per90(toNum(el.threat), mins),
      creat90: per90(toNum(el.creativity), mins),
      xpts,
      xptsSource,
      last4: form
        ? {
            points: toNum(form.last4_points ?? form.last_4_points),
            minutes: toNum(form.last4_minutes ?? form.last_4_minutes),
            xgi: round1(toNum(form.last4_xgi)),
            xgc: round1(toNum(form.last4_xgc)),
          }
        : null,
      transfersIn,
      transfersOut,
      transfersNet: transfersIn - transfersOut,
    };
  });
}

/* ------------------------------------------------------------------ fixtures */

export interface FixtureIndex {
  byGw: Map<number, FixtureLite[]>;
  teamGw: Map<number, Map<number, TeamGwFixture>>;
}

export function buildFixtureIndex(raw: RawFplFixture[], clubs: Club[]): FixtureIndex {
  const clubById = clubsById(clubs);
  const byGw = new Map<number, FixtureLite[]>();
  const teamGw = new Map<number, Map<number, TeamGwFixture>>();

  const add = (teamId: number, gw: number, entry: TeamGwFixture) => {
    let m = teamGw.get(teamId);
    if (!m) {
      m = new Map();
      teamGw.set(teamId, m);
    }
    if (!m.has(gw)) m.set(gw, entry);
  };

  for (const f of raw) {
    if (f.event == null) continue;
    const fixture: FixtureLite = {
      id: f.id ?? 0,
      event: f.event,
      teamH: f.team_h,
      teamA: f.team_a,
      kickoff: f.kickoff_time,
      finished: Boolean(f.finished),
      homeScore: f.team_h_score,
      awayScore: f.team_a_score,
      difficultyH: f.team_h_difficulty ?? null,
      difficultyA: f.team_a_difficulty ?? null,
    };
    const list = byGw.get(f.event) ?? [];
    list.push(fixture);
    byGw.set(f.event, list);

    const homeClub = clubById.get(f.team_h);
    const awayClub = clubById.get(f.team_a);
    if (homeClub && awayClub) {
      add(f.team_h, f.event, {
        gw: f.event,
        opp: awayClub,
        home: true,
        officialFdr: f.team_h_difficulty ?? 3,
        kickoff: f.kickoff_time,
        finished: Boolean(f.finished),
      });
      add(f.team_a, f.event, {
        gw: f.event,
        opp: homeClub,
        home: false,
        officialFdr: f.team_a_difficulty ?? 3,
        kickoff: f.kickoff_time,
        finished: Boolean(f.finished),
      });
    }
  }

  return { byGw, teamGw };
}

export function nextFixtures(
  index: FixtureIndex,
  teamId: number,
  fromGw: number,
  count: number,
): TeamGwFixture[] {
  const m = index.teamGw.get(teamId);
  if (!m) return [];
  const out: TeamGwFixture[] = [];
  const gws = [...m.keys()].sort((a, b) => a - b);
  for (const gw of gws) {
    if (gw < fromGw) continue;
    out.push(m.get(gw)!);
    if (out.length >= count) break;
  }
  return out;
}

export function teamGwDifficulty(
  index: FixtureIndex,
  teamId: number,
  fromGw: number,
  count: number,
): number {
  const list = nextFixtures(index, teamId, fromGw, count);
  if (!list.length) return 3;
  return round1(list.reduce((s, f) => s + f.officialFdr, 0) / list.length);
}

export function outlookLabel(avg: number): Outlook {
  if (avg < 2.4) return { label: 'Favorable', tone: 'pos' };
  if (avg < 3.1) return { label: 'Balanced', tone: 'info' };
  if (avg < 3.8) return { label: 'Challenging', tone: 'warn' };
  return { label: 'Very tough', tone: 'neg' };
}

/* ----------------------------------------------------------------- lookups */

/** Best-effort player lookup by display name (blend payloads use AIrsenal names). */
export function findPlayerIdByName(
  players: Player[],
  name: string,
): number | null {
  const target = norm(name);
  if (!target) return null;
  const exact = players.find((p) => norm(p.name) === target || norm(p.fullName) === target);
  if (exact) return exact.id;
  const contains = players.find(
    (p) => norm(p.name).includes(target) || target.includes(norm(p.name)),
  );
  return contains ? contains.id : null;
}

/* ----------------------------------------------------------------- gameweeks */
export interface RawEvent {
  id: number;
  name: string;
  deadline_time: string | null;
  finished: boolean;
  is_current: boolean;
  is_next: boolean;
  average_entry_score: number | null;
  highest_score: number | null;
}

export function buildGameweeks(events: RawEvent[] | undefined) {
  const list = (events ?? []).map((e) => ({
    id: e.id,
    name: e.name,
    deadline: e.deadline_time,
    finished: e.finished,
    isCurrent: e.is_current,
    isNext: e.is_next,
    average: toNum(e.average_entry_score),
    highest: toNum(e.highest_score),
  }));
  const current = list.find((e) => e.isCurrent);
  const next = list.find((e) => e.isNext);
  const lastFinished = [...list].reverse().find((e) => e.finished);
  return { list, current, next, lastFinished };
}
