import type { FixtureIndex } from './model';
import { teamGwDifficulty } from './model';
import type { Player } from './types';
import { clamp, round1 } from '../lib/format';

/** Fixture-aware per-gameweek multiplier used by sandbox projections. */
export function gwFactor(index: FixtureIndex | null, teamId: number, gw: number): number {
  if (!index) return 1;
  const diff = teamGwDifficulty(index, teamId, gw, 1);
  return clamp(1.18 - (diff - 3) * 0.09, 0.72, 1.3);
}

export function xiXpts(
  ids: number[],
  playersById: Map<number, Player>,
  captainId: number | null,
): number {
  return round1(
    ids.reduce((sum, id) => {
      const p = playersById.get(id);
      if (!p) return sum;
      const v = p.xpts * (id === captainId ? 2 : 1);
      return sum + v;
    }, 0),
  );
}

export function futureXpts(
  ids: number[],
  playersById: Map<number, Player>,
  captainId: number | null,
  index: FixtureIndex | null,
  fromGw: number,
  count: number,
  gwCount: number,
): number {
  return round1(
    ids.reduce((sum, id) => {
      const p = playersById.get(id);
      if (!p) return sum;
      let v = 0;
      for (let g = fromGw; g < fromGw + count && g <= gwCount; g++) {
        v += p.xpts * gwFactor(index, p.teamId, g);
      }
      if (id === captainId) v *= 2;
      return sum + v;
    }, 0),
  );
}
