import { useState } from 'react';
import type { Player, TeamGwFixture } from '../../domain/types';
import { clamp, surname } from '../../lib/format';
import { fplEndpoints } from '../../api/fpl/endpoints';

/* ------------------------------------------------------------------- pitch */

export interface StatContext {
  isCaptain?: boolean;
  isVice?: boolean;
  bench?: boolean;
}

export type StatOf = (
  player: Player,
  ctx: StatContext,
) => { value: number; suffix: string; decimals?: number };

export function PlayerChip({
  player,
  xpts,
  isCaptain,
  isVice,
  bench,
  selected,
  swapSrc,
  fixture,
  statOf,
  onClick,
}: {
  player: Player;
  xpts?: number;
  isCaptain?: boolean;
  isVice?: boolean;
  bench?: boolean;
  selected?: boolean;
  swapSrc?: boolean;
  fixture?: TeamGwFixture | null;
  statOf?: StatOf;
  onClick?: (player: Player) => void;
}) {
  const [failed, setFailed] = useState(false);
  const stat = statOf
    ? statOf(player, { isCaptain, isVice, bench })
    : { value: xpts ?? (isCaptain ? player.xpts * 2 : player.xpts), suffix: 'xP', decimals: 1 };
  const photo = player.code > 0 && !failed ? fplEndpoints.playerPhoto(player.code, '110x140') : null;
  const fdr = fixture?.officialFdr;
  const fdrKey = clamp(Math.round(fdr ?? 3), 1, 5);
  const barBackground = fdr != null
    ? `var(--fdr-${fdrKey})`
    : 'color-mix(in srgb, var(--accent) 22%, transparent)';
  const barColor = fdr != null && fdrKey >= 4 ? '#fff' : '#08131F';
  const ring = selected
    ? '0 0 0 2px var(--accent)'
    : swapSrc
      ? '0 0 0 2px var(--accent-2)'
      : undefined;

  return (
    <button
      className={`group relative w-[60px] transition-transform duration-150 sm:w-[78px] ${onClick ? 'cursor-pointer hover:-translate-y-0.5' : ''} ${bench ? 'opacity-95' : ''}`}
      type="button"
      onClick={onClick ? () => onClick(player) : undefined}
      aria-label={`${player.name}, ${stat.value.toFixed(stat.decimals ?? 0)} ${stat.suffix}`}
      style={{ boxShadow: ring, borderRadius: 'var(--r-sm)' }}
    >
      <div
        className="relative overflow-hidden rounded-t-[var(--r-sm)] border"
        style={{
          aspectRatio: '11 / 8',
          borderColor: selected ? 'var(--accent)' : swapSrc ? 'var(--accent-2)' : 'var(--border-strong)',
          background: 'color-mix(in srgb, var(--surface-2) 82%, transparent)',
        }}
      >
        {photo ? (
          <img
            src={photo}
            alt={player.name}
            loading="lazy"
            className="absolute inset-0 h-full w-full object-contain object-bottom"
            onError={() => setFailed(true)}
          />
        ) : (
          <span
            className="absolute inset-0 grid place-items-center text-[11px] font-bold"
            style={{ color: 'var(--muted)', fontFamily: 'var(--font-display)' }}
          >
            {player.name.slice(0, 3).toUpperCase()}
          </span>
        )}

        {isCaptain ? <span className="pc-tag pc-tag-c">C</span> : null}
        {isVice ? <span className="pc-tag pc-tag-v">V</span> : null}
      </div>
      {player.status !== 'a' ? (
        <span className="flag" title={player.news}>
          {player.status === 'i' ? '✕' : '!'}
        </span>
      ) : null}

      <div
        className="border-x px-1 py-0.5"
        style={{ borderColor: 'var(--border-strong)', background: 'color-mix(in srgb, var(--bg) 78%, transparent)' }}
      >
        <p className="line-clamp-1 text-center text-[10px] font-semibold leading-tight" style={{ color: 'var(--text)' }}>
          {surname(player.name)}
        </p>
      </div>

      <div className="rounded-b-[var(--r-sm)] border border-t-0 px-1 py-0.5" style={{ borderColor: 'var(--border-strong)', background: barBackground, color: barColor }}>
        <p className="text-center text-[9px] font-bold leading-tight">
          {stat.value.toFixed(stat.decimals ?? 0)} {stat.suffix}
        </p>
      </div>
    </button>
  );
}

function groupByPos(players: Player[]) {
  const by: Record<string, Player[]> = { GK: [], DEF: [], MID: [], FWD: [] };
  for (const p of players) by[p.pos]?.push(p);
  return by;
}

const ORDER: ('GK' | 'DEF' | 'MID' | 'FWD')[] = ['GK', 'DEF', 'MID', 'FWD'];

function PitchLines() {
  const line = '1px solid var(--pitch-line)';
  return (
    <div className="pointer-events-none absolute inset-3 sm:inset-4" aria-hidden="true">
      <div className="absolute inset-0 rounded-[var(--r-sm)]" style={{ border: line }} />
      <div className="absolute left-0 right-0 top-1/2 h-px" style={{ background: 'var(--pitch-line)' }} />
      <div className="absolute left-1/2 top-1/2 h-[110px] w-[110px] -translate-x-1/2 -translate-y-1/2 rounded-full" style={{ border: line }} />
      <div className="absolute left-1/2 top-1/2 h-1.5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full" style={{ background: 'var(--pitch-line)' }} />
      <div className="absolute left-1/2 top-0 h-[20%] w-[44%] -translate-x-1/2 border-x border-b" style={{ borderColor: 'var(--pitch-line)' }} />
      <div className="absolute bottom-0 left-1/2 h-[20%] w-[44%] -translate-x-1/2 border-x border-t" style={{ borderColor: 'var(--pitch-line)' }} />
    </div>
  );
}

function PitchRow({
  players,
  captainId,
  viceId,
  selectedId,
  swapId,
  fixtureOf,
  statOf,
  onChipClick,
  bench,
}: {
  players: Player[];
  captainId?: number | null;
  viceId?: number | null;
  selectedId?: number | null;
  swapId?: number | null;
  fixtureOf?: (player: Player) => TeamGwFixture | null;
  statOf?: StatOf;
  onChipClick?: (player: Player) => void;
  bench?: boolean;
}) {
  if (!players.length) return null;
  return (
    <div
      className="grid w-full place-items-center gap-2 sm:gap-3"
      style={{ gridTemplateColumns: `repeat(${players.length}, minmax(0, 1fr))` }}
    >
      {players.map((p) => (
        <PlayerChip
          key={p.id}
          player={p}
          bench={bench}
          isCaptain={captainId === p.id}
          isVice={viceId === p.id}
          selected={selectedId === p.id}
          swapSrc={swapId === p.id}
          fixture={fixtureOf ? fixtureOf(p) : null}
          statOf={statOf}
          onClick={onChipClick}
        />
      ))}
    </div>
  );
}

export function Pitch({
  ids,
  byId,
  captainId,
  viceId,
  selectedId,
  swapId,
  fixtureOf,
  statOf,
  onChipClick,
}: {
  ids: number[];
  byId: Map<number, Player>;
  captainId?: number | null;
  viceId?: number | null;
  selectedId?: number | null;
  swapId?: number | null;
  fixtureOf?: (player: Player) => TeamGwFixture | null;
  statOf?: StatOf;
  onChipClick?: (player: Player) => void;
}) {
  const players = ids.map((id) => byId.get(id)).filter((p): p is Player => Boolean(p));
  const by = groupByPos(players);
  return (
    <div
      className="relative overflow-hidden rounded-[var(--r-md)] px-2 py-4 sm:px-4 sm:py-5"
      style={{
        background:
          'repeating-linear-gradient(180deg, var(--pitch-1) 0px, var(--pitch-1) 56px, var(--pitch-2) 56px, var(--pitch-2) 112px)',
      }}
    >
      <PitchLines />
      <div className="relative flex flex-col items-center gap-3 sm:gap-4">
        {ORDER.map((pos) => (
          <PitchRow
            key={pos}
            players={by[pos]}
            captainId={captainId}
            viceId={viceId}
            selectedId={selectedId}
            swapId={swapId}
            fixtureOf={fixtureOf}
            statOf={statOf}
            onChipClick={onChipClick}
          />
        ))}
      </div>
    </div>
  );
}

export function BenchRow({
  ids,
  byId,
  captainId,
  viceId,
  selectedId,
  swapId,
  fixtureOf,
  statOf,
  onChipClick,
}: {
  ids: number[];
  byId: Map<number, Player>;
  captainId?: number | null;
  viceId?: number | null;
  selectedId?: number | null;
  swapId?: number | null;
  fixtureOf?: (player: Player) => TeamGwFixture | null;
  statOf?: StatOf;
  onChipClick?: (player: Player) => void;
}) {
  const players = ids.map((id) => byId.get(id)).filter((p): p is Player => Boolean(p));
  if (!players.length) return null;
  return (
    <div
      className="mt-4 rounded-[var(--r-md)] px-3 py-3"
      style={{ background: 'color-mix(in srgb, var(--bg) 70%, transparent)' }}
    >
      <p className="mb-2 text-center text-[10px] font-semibold uppercase tracking-[.16em]" style={{ color: 'var(--muted)' }}>
        Bench
      </p>
      <div
        className="grid w-full place-items-center gap-2 sm:gap-3"
        style={{ gridTemplateColumns: `repeat(${players.length}, minmax(0, 1fr))` }}
      >
        {players.map((p) => (
          <PlayerChip
            key={p.id}
            player={p}
            bench
            isCaptain={captainId === p.id}
            isVice={viceId === p.id}
            selected={selectedId === p.id}
            swapSrc={swapId === p.id}
            fixture={fixtureOf ? fixtureOf(p) : null}
            statOf={statOf}
            onClick={onChipClick}
          />
        ))}
      </div>
    </div>
  );
}
