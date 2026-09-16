import { useEffect, useState } from 'react';
import {
  getBootstrap,
  getEntry,
  getFixtures,
  type FplBootstrap,
} from '../api/fpl/fpl';
import { fetchJson } from '../api/fpl/client';
import { fplEndpoints } from '../api/fpl/endpoints';
import {
  getMyTeam,
  getPredictions,
  getFormLast4,
  getBandwagons,
  getInjuryNews,
  getSeasonStatus,
  getTransfersLatest,
  getCopilotBlendSnapshot,
  getCopilotBlendSnapshotGlobal,
  type MyTeamResponse,
  type SeasonStatus,
  type LatestTransferRow,
  type CopilotBlendSubmitRequest,
  type CopilotHybridResultPayload,
  type CopilotRecommendedTransfer,
} from '../api/backend';
import {
  buildClubs,
  buildPlayers,
  buildFixtureIndex,
  buildGameweeks,
  type FixtureIndex,
  type RawEvent,
} from '../domain/model';
import type { Club, GameweekMeta, ManagerTeam, Player, SquadPick } from '../domain/types';
import { clearCache } from '../lib/cache';

type BootstrapFull = FplBootstrap & { events?: RawEvent[] };

type EntryHistory = {
  current: Array<{
    event: number;
    points: number;
    rank: number;
    overall_rank: number;
    event_transfers?: number;
    event_transfers_cost?: number;
  }>;
};

export interface GWHistoryRow {
  event: number;
  points: number;
  rank: number;
  overall_rank: number;
  transfers: number;
  hit: number;
}

export interface CoreData {
  loading: boolean;
  error: string | null;
  bootstrap: FplBootstrap | null;
  clubs: Club[];
  players: Player[];
  playersById: Map<number, Player>;
  fixtureIndex: FixtureIndex | null;
  gameweeks: GameweekMeta[];
  currentGW: number;
  nextGW: number;
  nextDeadline: string | null;
  entry: { name: string; manager: string } | null;
  manager: ManagerTeam | null;
  myTeam: MyTeamResponse | null;
  xi: number[];
  bench: number[];
  captainId: number | null;
  viceId: number | null;
  seasonStatus: SeasonStatus | null;
  history: GWHistoryRow[];
  transfersLatest: LatestTransferRow[];
  blendInput: CopilotBlendSubmitRequest | null;
  blendResult: CopilotHybridResultPayload | null;
  recommendedTransfers: CopilotRecommendedTransfer[];
  refresh: () => void;
}

const CHIP_META: Record<string, { key: ManagerTeam['chips'][number]['key']; name: string; note: string }> = {
  wildcard: { key: 'WC', name: 'Wildcard', note: 'Unlimited transfers, no hits. One per half-season.' },
  freehit: { key: 'FH', name: 'Free Hit', note: 'Change your whole squad for one gameweek only.' },
  bboost: { key: 'BB', name: 'Bench Boost', note: 'All 15 squad players score points for one GW.' },
  '3xc': { key: 'TC', name: 'Triple Captain', note: 'Captain scores triple instead of double.' },
};

function buildManager(
  myTeam: MyTeamResponse | null,
  entry: { name: string; manager: string } | null,
  teamId: string,
  history: EntryHistory | null,
  currentGW: number,
): ManagerTeam | null {
  if (!myTeam) return null;
  const current = history?.current?.find((h) => h.event === currentGW);
  const prev = history?.current?.find((h) => h.event === currentGW - 1);

  const picks: SquadPick[] = myTeam.picks.map((p) => ({
    element: p.element,
    position: p.position,
    multiplier: p.multiplier,
    isCaptain: p.is_captain,
    isViceCaptain: p.is_vice_captain,
    elementType: p.element_type,
    sellingPrice: p.selling_price / 10,
    purchasePrice: p.purchase_price / 10,
  }));

  const chips = myTeam.chips.map((c) => {
    const meta = CHIP_META[c.name] ?? { key: 'WC' as const, name: c.name, note: '' };
    const used = c.status_for_entry === 'played' || c.played_by_entry.length > 0;
    return {
      key: meta.key,
      name: meta.name,
      note: meta.note,
      used,
      usedGw: c.played_by_entry.length ? c.played_by_entry[c.played_by_entry.length - 1] : undefined,
    };
  });

  const overallRank = current?.overall_rank ?? prev?.overall_rank ?? 0;
  const prevRank = prev?.overall_rank ?? overallRank;

  return {
    name: entry?.name ?? 'My Team',
    manager: entry?.manager ?? 'Manager',
    teamId,
    overallRank,
    overallRankDelta: prevRank - overallRank,
    gwPoints: current?.points ?? 0,
    gwRank: current?.rank ?? 0,
    average: 0,
    highest: 0,
    bank: myTeam.transfers.bank / 10,
    squadValue: myTeam.transfers.value / 10,
    freeTransfers: Math.max(0, myTeam.transfers.limit - myTeam.transfers.made),
    chips,
    picks,
    lastUpdated: myTeam.picks_last_updated ?? null,
  };
}

const emptyMaps = () => ({
  clubs: [] as Club[],
  players: [] as Player[],
  playersById: new Map<number, Player>(),
});

export function useCoreData(teamId: string | null): CoreData {
  const [reloadKey, setReloadKey] = useState(0);
  const [state, setState] = useState<Omit<CoreData, 'refresh'>>({
    loading: true,
    error: null,
    bootstrap: null,
    ...emptyMaps(),
    fixtureIndex: null,
    gameweeks: [],
    currentGW: 0,
    nextGW: 0,
    nextDeadline: null,
    entry: null,
    manager: null,
    myTeam: null,
    xi: [],
    bench: [],
    captainId: null,
    viceId: null,
    seasonStatus: null,
    history: [],
    transfersLatest: [],
    blendInput: null,
    blendResult: null,
    recommendedTransfers: [],
  });

  useEffect(() => {
    if (!teamId) {
      setState((s) => ({ ...s, loading: false, error: 'No team ID' }));
      return;
    }
    let cancelled = false;
    const id = teamId;

    async function load() {
      setState((s) => ({ ...s, loading: true, error: null }));
      try {
        const bootstrap = (await getBootstrap()) as BootstrapFull;
        const clubs = buildClubs(bootstrap);
        const gw = buildGameweeks(bootstrap.events);
        const currentGW = gw.current?.id ?? gw.lastFinished?.id ?? 1;
        const nextGW = gw.next?.id ?? currentGW + 1;

        const [entryRes, fixturesRes, myTeamRes, historyRes, predsRes, formRes, bwRes, injRes, seasonRes, transfersRes] =
          await Promise.allSettled([
            getEntry(id),
            getFixtures(),
            getMyTeam(),
            fetchJson<EntryHistory>(fplEndpoints.entryHistory(id)),
            getPredictions(nextGW),
            getFormLast4(),
            getBandwagons(),
            getInjuryNews(),
            getSeasonStatus(),
            getTransfersLatest(),
          ]);

        const entry = entryRes.status === 'fulfilled'
          ? { name: entryRes.value.name, manager: `${entryRes.value.player_first_name} ${entryRes.value.player_last_name}`.trim() }
          : null;
        const rawFixtures = fixturesRes.status === 'fulfilled' ? fixturesRes.value : [];
        const myTeam = myTeamRes.status === 'fulfilled' ? myTeamRes.value : null;
        const history = historyRes.status === 'fulfilled' ? historyRes.value : null;
        const predictions = predsRes.status === 'fulfilled' ? predsRes.value : [];
        const form = formRes.status === 'fulfilled' ? formRes.value : [];
        const bandwagons = bwRes.status === 'fulfilled' ? bwRes.value : [];
        const injuries = injRes.status === 'fulfilled' ? injRes.value : [];
        const seasonStatus = seasonRes.status === 'fulfilled' ? seasonRes.value : null;
        const transfersLatest = transfersRes.status === 'fulfilled' ? transfersRes.value : [];

        const players = buildPlayers(bootstrap, clubs, { predictions, form, bandwagons, injuries });
        const playersById = new Map(players.map((p) => [p.id, p]));
        const fixtureIndex = buildFixtureIndex(rawFixtures, clubs);

        let blendResult: CopilotHybridResultPayload | null = null;
        let blendInput: CopilotBlendSubmitRequest | null = null;
        try {
          const numericId = Number(id);
          let snap = Number.isFinite(numericId)
            ? await getCopilotBlendSnapshot(nextGW, numericId)
            : null;
          if (!snap) snap = await getCopilotBlendSnapshotGlobal(nextGW);
          if (snap) {
            blendResult = snap.result;
            blendInput = snap.input;
          }
        } catch {
          /* blend snapshot is optional */
        }

        const picks = myTeam?.picks ?? [];
        const xi = picks
          .filter((p) => p.position <= 11)
          .sort((a, b) => a.position - b.position)
          .map((p) => p.element);
        const bench = picks
          .filter((p) => p.position > 11)
          .sort((a, b) => a.position - b.position)
          .map((p) => p.element);
        const captainId = picks.find((p) => p.is_captain)?.element ?? xi[0] ?? null;
        const viceId = picks.find((p) => p.is_vice_captain)?.element ?? xi[1] ?? null;

        const manager = buildManager(myTeam, entry, id, history, currentGW);
        if (manager) {
          manager.average = gw.current?.average ?? 0;
          manager.highest = gw.current?.highest ?? 0;
        }

        if (cancelled) return;
        setState({
          loading: false,
          error: null,
          bootstrap,
          clubs,
          players,
          playersById,
          fixtureIndex,
          gameweeks: gw.list,
          currentGW,
          nextGW,
          nextDeadline: gw.next?.deadline ?? null,
          entry,
          manager,
          myTeam,
          xi,
          bench,
          captainId,
          viceId,
          seasonStatus,
          history: (history?.current ?? []).map((h) => ({
            event: h.event,
            points: h.points,
            rank: h.rank,
            overall_rank: h.overall_rank,
            transfers: h.event_transfers ?? 0,
            hit: h.event_transfers_cost ?? 0,
          })),
          transfersLatest,
          blendInput,
          blendResult,
          recommendedTransfers: blendResult?.recommended_transfers ?? [],
        });
      } catch (err) {
        if (!cancelled) {
          setState((s) => ({
            ...s,
            loading: false,
            error: err instanceof Error ? err.message : 'Failed to load FPL data',
          }));
        }
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, [teamId, reloadKey]);

  return {
    ...state,
    refresh: () => {
      clearCache('fpl:');
      setReloadKey((k) => k + 1);
    },
  };
}
