import { useState } from 'react';
import type { Player } from '../../data/gwOverviewMocks';
import { getDifficultyColor } from '../../utils/difficulty';
import { fplEndpoints } from '../../api/fpl/endpoints';
import { Button } from '@/components/ui/button';
import { DashboardCard } from '@/components/ui/primitives';

interface Props {
  squad: Player[];
  /** When provided, player chips become clickable for swap / selection. */
  onPlayerClick?: (player: Player) => void;
  /** ID of the currently selected player (highlighted). */
  selectedPlayerId?: number | null;
  /** Hint text shown when a player is selected. */
  swapHint?: string;
  /** Set a starter as captain (single-tap the C badge). */
  onSetCaptain?: (player: Player) => void;
  /** Set a starter as vice-captain (single-tap the V badge). */
  onSetViceCaptain?: (player: Player) => void;
}

const CHIP: {
  imgAspect: string;
  imgScale: number;
  imgAnchor: 'top' | 'center' | 'bottom';
  imgYOffset: number;
  imgFit: 'cover' | 'contain';
} = {
  imgAspect: '11 / 8',
  imgScale: 140,
  imgAnchor: 'bottom',
  imgYOffset: 35,
  imgFit: 'contain',
};

const PlayerChip = ({
  player,
  onClick,
  isSelected,
  onCaptainClick,
  onViceCaptainClick,
}: {
  player: Player;
  onClick?: () => void;
  isSelected?: boolean;
  onCaptainClick?: () => void;
  onViceCaptainClick?: () => void;
}) => {
  const difficultyStyles = (() => {
    if (player.chipDifficulty !== undefined) {
      const color = getDifficultyColor(player.chipDifficulty);
      if (color === 'emerald') {
        return {
          bg: 'bg-[rgba(16,185,129,0.2)]',
          borderColor: 'border-[rgba(16,185,129,0.25)]',
          color: 'text-emerald-400',
        };
      }
      if (color === 'yellow') {
        return {
          bg: 'bg-[rgba(250,204,21,0.2)]',
          borderColor: 'border-[rgba(250,204,21,0.25)]',
          color: 'text-yellow-300',
        };
      }
      return {
        bg: 'bg-[rgba(244,63,94,0.2)]',
        borderColor: 'border-[rgba(244,63,94,0.25)]',
        color: 'text-red-300',
      };
    }
    return {
      bg: 'bg-[rgba(16,185,129,0.2)]',
      borderColor: 'border-[rgba(16,185,129,0.25)]',
      color: 'text-emerald-400',
    };
  })();

  const objectPos = CHIP.imgAnchor === 'top' ? 'top' : CHIP.imgAnchor === 'center' ? 'center' : 'bottom';
  const wrapperAlign =
    CHIP.imgAnchor === 'top' ? 'items-start' : CHIP.imgAnchor === 'center' ? 'items-center' : 'items-end';

  return (
    <div
      className={`relative w-[60px] rounded-lg transition-[transform,box-shadow] duration-150 focus-visible:outline-2 focus-visible:outline-blue-400 focus-visible:outline-offset-2 sm:w-[81px] ${onClick ? 'cursor-pointer' : ''} ${isSelected ? 'scale-[1.08] shadow-[0_0_0_2px_rgba(59,130,246,0.7),0_0_12px_rgba(59,130,246,0.35)]' : ''} ${onClick ? (isSelected ? 'hover:scale-[1.08]' : 'hover:scale-[1.04]') : ''}`}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={(e) => {
        if (onClick && (e.key === 'Enter' || e.key === ' ')) {
          e.preventDefault();
          onClick();
        }
      }}
    >
      <div
        className={`relative overflow-hidden rounded-t-lg border bg-[rgba(51,65,85,0.4)] ${isSelected ? 'border-blue-400' : 'border-white/12'}`}
        style={{ aspectRatio: CHIP.imgAspect }}
      >
        {player.photoUrl ? (
          CHIP.imgFit === 'cover' ? (
            <img
              src={player.photoUrl}
              alt={player.name}
              className="absolute inset-0 h-full w-full object-cover"
              style={{
                objectPosition: objectPos,
                transform: `translateY(${CHIP.imgYOffset}%)`,
              }}
              loading="lazy"
              onError={(e) => {
                const img = e.currentTarget;
                if (img.dataset.photoFallback) return;
                img.dataset.photoFallback = '1';
                img.src = fplEndpoints.playerPlaceholder();
              }}
            />
          ) : (
            <div className={`flex h-full w-full justify-center ${wrapperAlign}`}>
              <img
                src={player.photoUrl}
                alt={player.name}
                className="max-h-[140%] w-auto object-contain"
                style={{
                  objectPosition: objectPos,
                  transform: `translateY(${CHIP.imgYOffset}%)`,
                }}
                loading="lazy"
                onError={(e) => {
                  const img = e.currentTarget;
                  if (img.dataset.photoFallback) return;
                  img.dataset.photoFallback = '1';
                  img.src = fplEndpoints.playerPlaceholder();
                }}
              />
            </div>
          )
        ) : (
          <div className="flex h-full w-full items-center justify-center text-[10px] font-bold uppercase text-white">
            {player.name.slice(0, 3)}
          </div>
        )}

      </div>

      {/* Captain / Vice-Captain badges */}
      {player.isCaptain ? (
        <div className="absolute top-[2px] right-[2px] z-[3] flex size-4 items-center justify-center rounded-full bg-yellow-400 text-[8px] font-bold text-black">
          C
        </div>
      ) : player.isViceCaptain ? (
        <div className="absolute top-[2px] right-[2px] z-[3] flex size-4 items-center justify-center rounded-full bg-slate-400 text-[8px] font-bold text-black">
          V
        </div>
      ) : !player.isBench && (onCaptainClick || onViceCaptainClick) ? (
        <div
          onClick={(e) => e.stopPropagation()}
          onMouseDown={(e) => e.stopPropagation()}
          style={{ position: 'absolute', top: 2, right: 2, display: 'flex', flexDirection: 'column', gap: 2, zIndex: 10 }}
        >
          {onCaptainClick && (
            <button
              type="button"
              onClick={onCaptainClick}
              style={{
                width: 18, height: 18, borderRadius: '50%', border: 'none',
                background: 'rgba(255,255,255,0.3)', color: 'rgba(255,255,255,0.8)',
                fontSize: 8, fontWeight: 700, cursor: 'pointer', display: 'grid', placeItems: 'center',
                transition: 'all 0.15s',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.background = '#facc15'; e.currentTarget.style.color = '#000'; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.3)'; e.currentTarget.style.color = 'rgba(255,255,255,0.8)'; }}
            >
              C
            </button>
          )}
          {onViceCaptainClick && (
            <button
              type="button"
              onClick={onViceCaptainClick}
              style={{
                width: 18, height: 18, borderRadius: '50%', border: 'none',
                background: 'rgba(255,255,255,0.3)', color: 'rgba(255,255,255,0.8)',
                fontSize: 8, fontWeight: 700, cursor: 'pointer', display: 'grid', placeItems: 'center',
                transition: 'all 0.15s',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.background = '#94a3b8'; e.currentTarget.style.color = '#000'; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.3)'; e.currentTarget.style.color = 'rgba(255,255,255,0.8)'; }}
            >
              V
            </button>
          )}
        </div>
      ) : null}

      <div className="border-x border-white/12 bg-[rgba(15,23,42,0.8)] px-1 py-0.5">
        <p className="line-clamp-1 text-center text-[10px] leading-tight font-medium text-white">
          {player.name}
        </p>
      </div>

      <div className={`rounded-b-md border px-3 py-0.5 ${difficultyStyles.borderColor} ${difficultyStyles.bg} ${difficultyStyles.color}`}>
        {player.chipLabel ? (
          <p className="text-center text-[7px] leading-tight font-semibold opacity-80">
            {player.chipLabel}
          </p>
        ) : null}
        <p className="text-center text-[9px] font-bold">
          {player.chipLabel != null || player.chipDifficulty != null
            ? `${player.isCaptain ? (player.points * 2).toFixed(1) : player.points.toFixed(1)} xP`
            : player.isCaptain
              ? player.points * 2
              : player.points}
        </p>
      </div>
    </div>
  );
};

const PitchRow = ({
  players,
  onPlayerClick,
  selectedPlayerId,
  onSetCaptain,
  onSetViceCaptain,
}: {
  players: Player[];
  onPlayerClick?: (p: Player) => void;
  selectedPlayerId?: number | null;
  onSetCaptain?: (p: Player) => void;
  onSetViceCaptain?: (p: Player) => void;
}) => (
  <div
    className="grid w-full place-items-center gap-x-2 gap-y-2 sm:gap-x-4 sm:gap-y-3"
    style={{ gridTemplateColumns: `repeat(${Math.max(players.length, 1)}, minmax(0, 1fr))` }}
  >
    {players.map((p) => (
      <PlayerChip
        key={p.name}
        player={p}
        onClick={onPlayerClick ? () => onPlayerClick(p) : undefined}
        isSelected={selectedPlayerId != null && p.id === selectedPlayerId}
        onCaptainClick={onSetCaptain ? () => onSetCaptain(p) : undefined}
        onViceCaptainClick={onSetViceCaptain ? () => onSetViceCaptain(p) : undefined}
      />
    ))}
  </div>
);

const BenchRow = ({
  players,
  onPlayerClick,
  selectedPlayerId,
}: {
  players: Player[];
  onPlayerClick?: (p: Player) => void;
  selectedPlayerId?: number | null;
}) => (
  <div
    className="grid w-full place-items-center gap-x-2 gap-y-2 sm:gap-x-4 sm:gap-y-3"
    style={{ gridTemplateColumns: `repeat(${Math.max(players.length, 1)}, minmax(0, 1fr))` }}
  >
    {players.map((p) => (
      <PlayerChip
        key={p.name}
        player={p}
        onClick={onPlayerClick ? () => onPlayerClick(p) : undefined}
        isSelected={selectedPlayerId != null && p.id === selectedPlayerId}
      />
    ))}
  </div>
);

const PitchCard = ({ squad, onPlayerClick, selectedPlayerId, swapHint, onSetCaptain, onSetViceCaptain }: Props) => {
  const [tab, setTab] = useState<'pitch' | 'table'>('pitch');

  const starters = squad.filter((p) => !p.isBench);
  const bench = squad.filter((p) => p.isBench);

  const gk = starters.filter((p) => p.position === 'GK');
  const def = starters.filter((p) => p.position === 'DEF');
  const mid = starters.filter((p) => p.position === 'MID');
  const fwd = starters.filter((p) => p.position === 'FWD');

  return (
    <DashboardCard>
      <div className="flex border-b border-white/6">
        {(['pitch', 'table'] as const).map((t) => (
          <Button
            key={t}
            onClick={() => setTab(t)}
            variant="ghost"
            className={`h-auto flex-1 rounded-none border-b-2 py-3 text-sm font-semibold capitalize hover:bg-transparent hover:text-white ${tab === t ? 'border-emerald-400 text-emerald-400' : 'border-transparent text-slate-400'}`}
          >
            {t}
          </Button>
        ))}
      </div>

      {tab === 'pitch' ? (
        <div
          className="relative px-3 py-4 sm:px-6 sm:py-6 lg:px-8"
          style={{ background: 'repeating-linear-gradient(180deg, #1a3d1a 0px, #1a3d1a 60px, #1f4a1f 60px, #1f4a1f 120px)' }}
        >
          <PitchLines />
          <div className="relative flex flex-col items-center gap-5">
            {swapHint && selectedPlayerId != null && (
              <p className="rounded-md bg-[rgba(59,130,246,0.1)] px-3 py-1 text-center text-xs font-medium text-blue-300">
                {swapHint}
              </p>
            )}
            <PitchRow players={gk} onPlayerClick={onPlayerClick} selectedPlayerId={selectedPlayerId} onSetCaptain={onSetCaptain} onSetViceCaptain={onSetViceCaptain} />
            <PitchRow players={def} onPlayerClick={onPlayerClick} selectedPlayerId={selectedPlayerId} onSetCaptain={onSetCaptain} onSetViceCaptain={onSetViceCaptain} />
            <PitchRow players={mid} onPlayerClick={onPlayerClick} selectedPlayerId={selectedPlayerId} onSetCaptain={onSetCaptain} onSetViceCaptain={onSetViceCaptain} />
            <PitchRow players={fwd} onPlayerClick={onPlayerClick} selectedPlayerId={selectedPlayerId} onSetCaptain={onSetCaptain} onSetViceCaptain={onSetViceCaptain} />
            <div className="mt-2 w-full rounded-xl bg-[rgba(15,23,42,0.8)] px-4 py-3">
              <p className="mb-2 text-center text-[10px] font-semibold tracking-widest uppercase text-slate-500">
                Bench
              </p>
              <BenchRow players={bench} onPlayerClick={onPlayerClick} selectedPlayerId={selectedPlayerId} />
            </div>
          </div>
        </div>
      ) : (
        <div className="flex items-center justify-center py-20 text-sm text-slate-500">
          Table view coming soon
        </div>
      )}
    </DashboardCard>
  );
};

function PitchLines() {
  return (
    <div className="pointer-events-none absolute inset-3 sm:inset-4">
      <div className="absolute inset-0 rounded-lg border border-white/12" />
      <div className="absolute top-1/2 right-0 left-0 h-px bg-white/8" />
      <div className="absolute top-1/2 left-1/2 h-[112px] w-[112px] -translate-x-1/2 -translate-y-1/2 rounded-full border border-white/8" />
      <div className="absolute top-1/2 left-1/2 h-1.5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white/12" />
      <div className="absolute top-[22%] left-1/2 h-[8%] w-[24%] -translate-x-1/2 rounded-b-full border border-t-0 border-white/8" />
      <div className="absolute bottom-[22%] left-1/2 h-[8%] w-[24%] -translate-x-1/2 rounded-t-full border border-b-0 border-white/8" />
      <div className="absolute top-0 left-1/2 h-[22%] w-[44%] -translate-x-1/2 border-x border-b border-white/8" />
      <div className="absolute top-0 left-1/2 h-[12%] w-[24%] -translate-x-1/2 border-x border-b border-white/8" />
      <div className="absolute top-[16%] left-1/2 h-1.5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white/12" />
      <div className="absolute bottom-0 left-1/2 h-[22%] w-[44%] -translate-x-1/2 border-x border-t border-white/8" />
      <div className="absolute bottom-0 left-1/2 h-[12%] w-[24%] -translate-x-1/2 border-x border-t border-white/8" />
      <div className="absolute bottom-[16%] left-1/2 h-1.5 w-1.5 -translate-x-1/2 rounded-full bg-white/12" />
    </div>
  );
}

export default PitchCard;
