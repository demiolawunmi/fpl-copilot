import { useEffect, useState } from 'react';
import { getPicks } from '../api/fpl/fpl';
import { fetchJson } from '../api/fpl/client';
import { fplEndpoints } from '../api/fpl/endpoints';

type FplLiveEvent = {
  elements: Array<{ id: number; stats: { total_points: number } }>;
};

export interface GameweekPicks {
  loading: boolean;
  /** True when the FPL API returned a squad for this gameweek. */
  hasPicks: boolean;
  xi: number[];
  bench: number[];
  captainId: number | null;
  viceId: number | null;
  pointsById: Map<number, number>;
  multiplierById: Map<number, number>;
}

const EMPTY: GameweekPicks = {
  loading: false,
  hasPicks: false,
  xi: [],
  bench: [],
  captainId: null,
  viceId: null,
  pointsById: new Map(),
  multiplierById: new Map(),
};

/**
 * The actual picks for a specific gameweek (`/entry/{id}/event/{gw}/picks/`)
 * plus that gameweek's live points (`/event/{gw}/live/`). Falls back to
 * `hasPicks: false` when the gameweek has not been played yet.
 */
export function useGameweekPicks(teamId: string | null, gw: number): GameweekPicks {
  const [state, setState] = useState<GameweekPicks>(EMPTY);

  useEffect(() => {
    if (!teamId || !gw) {
      return;
    }
    let cancelled = false;

    (async () => {
      setState((s) => ({ ...s, loading: true }));
      try {
        const picks = await getPicks(teamId, gw);
        const live = await fetchJson<FplLiveEvent>(fplEndpoints.liveEvent(gw)).catch(() => null);

        const sorted = picks.picks.slice().sort((a, b) => a.position - b.position);
        const xi = sorted.filter((p) => p.position <= 11).map((p) => p.element);
        const bench = sorted.filter((p) => p.position > 11).map((p) => p.element);

        const pointsById = new Map<number, number>();
        const multiplierById = new Map<number, number>();
        for (const p of sorted) multiplierById.set(p.element, p.multiplier);
        for (const el of live?.elements ?? []) pointsById.set(el.id, el.stats.total_points);

        if (cancelled) return;
        setState({
          loading: false,
          hasPicks: sorted.length > 0,
          xi,
          bench,
          captainId: sorted.find((p) => p.is_captain)?.element ?? null,
          viceId: sorted.find((p) => p.is_vice_captain)?.element ?? null,
          pointsById,
          multiplierById,
        });
      } catch {
        if (!cancelled) setState({ ...EMPTY });
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [teamId, gw]);

  if (!teamId || !gw) return EMPTY;
  return state;
}
