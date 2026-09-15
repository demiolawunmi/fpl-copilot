import { useNavigate, useLocation } from 'react-router-dom';
import type { EnhancedPlayer } from '../../data/commandCenterMocks';
import { DashboardCard } from '@/components/ui/primitives';
import { Button } from '@/components/ui/button';

interface Props {
  squad: EnhancedPlayer[];
  onSetCaptain: (playerId: number) => void;
}

const CommandCenterPitch = ({ squad, onSetCaptain }: Props) => {
  const navigate = useNavigate();
  const location = useLocation();
  const starters = squad.filter((p) => !p.isBench);
  const bench = squad.filter((p) => p.isBench);

  const gk = starters.filter((p) => p.position === 'GK');
  const def = starters.filter((p) => p.position === 'DEF');
  const mid = starters.filter((p) => p.position === 'MID');
  const fwd = starters.filter((p) => p.position === 'FWD');

  const handlePlayerClick = (player: EnhancedPlayer) => {
    const id = player?.id;
    if (id == null || typeof id !== 'number' || Number.isNaN(id) || id <= 0) return;
    navigate(`/players/${id}`, { state: { from: location.pathname } });
  };

  return (
    <DashboardCard>
      <div className="flex border-b border-white/6">
        <Button variant="ghost" className="flex-1 rounded-none border-b-2 border-b-emerald-400 py-3 text-sm font-semibold text-emerald-400 hover:bg-transparent">
          Pitch View
        </Button>
      </div>

      <div
        className="relative px-3 py-4 sm:px-6 sm:py-6 lg:px-8"
        style={{ background: 'repeating-linear-gradient(180deg, #1a3d1a 0px, #1a3d1a 60px, #1f4a1f 60px, #1f4a1f 120px)' }}
      >
        <PitchLines />
        <div className="relative flex flex-col items-center gap-5">
          <PitchRow players={gk} onSetCaptain={onSetCaptain} onPlayerClick={handlePlayerClick} />
          <PitchRow players={def} onSetCaptain={onSetCaptain} onPlayerClick={handlePlayerClick} />
          <PitchRow players={mid} onSetCaptain={onSetCaptain} onPlayerClick={handlePlayerClick} />
          <PitchRow players={fwd} onSetCaptain={onSetCaptain} onPlayerClick={handlePlayerClick} />
          <div className="mt-2 w-full rounded-xl bg-[rgba(15,23,42,0.8)] px-4 py-3">
            <p className="mb-2 text-center text-[10px] font-semibold uppercase tracking-widest text-slate-500">Bench</p>
            <BenchRow players={bench} onPlayerClick={handlePlayerClick} />
          </div>
        </div>
      </div>
    </DashboardCard>
  );
};

const PitchRow = ({ players, onSetCaptain, onPlayerClick }: { players: EnhancedPlayer[]; onSetCaptain: (id: number) => void; onPlayerClick?: (player: EnhancedPlayer) => void }) => (
  <div
    className="grid w-full place-items-center gap-x-3 gap-y-3 sm:gap-x-6 sm:gap-y-4"
    style={{ gridTemplateColumns: `repeat(${Math.max(players.length, 1)}, minmax(0, 1fr))` }}
  >
    {players.map((p) => (
      <PlayerChip key={p.id} player={p} onSetCaptain={onSetCaptain} onPlayerClick={onPlayerClick} />
    ))}
  </div>
);

const BenchRow = ({ players, onPlayerClick }: { players: EnhancedPlayer[]; onPlayerClick?: (player: EnhancedPlayer) => void }) => (
  <div
    className="grid w-full place-items-center gap-x-3 gap-y-3 sm:gap-x-6 sm:gap-y-4"
    style={{ gridTemplateColumns: `repeat(${Math.max(players.length, 1)}, minmax(0, 1fr))` }}
  >
    {players.map((p) => (
      <PlayerChip key={p.id} player={p} onPlayerClick={onPlayerClick} />
    ))}
  </div>
);

const PlayerChip = ({ player, onSetCaptain, onPlayerClick }: { player: EnhancedPlayer; onSetCaptain?: (id: number) => void; onPlayerClick?: (player: EnhancedPlayer) => void }) => {
  const minutesColor = player.minutesRisk === 'Safe' ? 'text-emerald-400' : player.minutesRisk === 'Risk' ? 'text-orange-300' : 'text-slate-400';
  const injuryBadge = player.injuryStatus !== 'Available';
  const chipBg = player.injuryStatus === 'Injured'
    ? 'bg-red-800'
    : player.injuryStatus === 'Doubtful'
      ? 'bg-orange-700'
      : player.injuryStatus === 'Suspended'
        ? 'bg-yellow-700'
        : 'bg-slate-700';
  const chipBorder = player.injuryStatus === 'Injured'
    ? 'border-red-500'
    : player.injuryStatus === 'Doubtful'
      ? 'border-orange-400'
      : player.injuryStatus === 'Suspended'
        ? 'border-yellow-400'
        : 'border-white/12';
  const handleClick = () => {
    if (onPlayerClick) onPlayerClick(player);
    if (onSetCaptain) onSetCaptain(player.id);
  };
  const isClickable = Boolean(onPlayerClick || onSetCaptain);

  return (
    <div className="flex w-[84px] flex-col items-center gap-1 sm:w-24">
      <div className="relative">
        <div
          className={`flex size-10 items-center justify-center rounded-full border-2 text-[10px] font-bold uppercase text-white transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 focus-visible:ring-offset-2 focus-visible:ring-offset-gray-900 ${chipBorder} ${chipBg} ${isClickable ? 'cursor-pointer' : 'cursor-default'}`}
          onClick={isClickable ? handleClick : undefined}
          role={isClickable ? 'button' : undefined}
          tabIndex={isClickable ? 0 : undefined}
          onKeyDown={isClickable ? (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              handleClick();
            }
          } : undefined}
        >
          {player.name.slice(0, 3)}
        </div>
        {player.isCaptain ? <div className="absolute -top-1 -right-1 flex size-4 items-center justify-center rounded-full bg-yellow-400 text-[8px] font-bold text-black">C</div> : null}
        {player.isViceCaptain ? <div className="absolute -top-1 -right-1 flex size-4 items-center justify-center rounded-full bg-slate-400 text-[8px] font-bold text-black">V</div> : null}
        {injuryBadge ? <div className="absolute -bottom-1 -right-1 flex size-3 items-center justify-center rounded-full bg-red-500 text-[8px] font-bold text-white">!</div> : null}
      </div>
      <span className="w-full line-clamp-1 text-center text-[11px] font-medium leading-tight text-white">{player.name}</span>
      <div className="flex items-center gap-1">
        <div className="rounded-md bg-[rgba(16,185,129,0.2)] px-1.5 py-0.5 text-[10px] font-bold text-emerald-400">
          {player.xPts.toFixed(1)}
        </div>
        <span className={`text-[9px] font-medium ${minutesColor}`} title={`Minutes risk: ${player.minutesRisk}`}>
          {player.minutesRisk === 'Safe' ? '✓' : player.minutesRisk === 'Risk' ? '⚠' : '?'}
        </span>
      </div>
    </div>
  );
};

function PitchLines() {
  return (
    <div className="pointer-events-none absolute inset-3 sm:inset-4">
      <div className="absolute inset-0 rounded-lg border border-white/12" />
      <div className="absolute left-0 right-0 top-1/2 h-px bg-white/8" />
      <div className="absolute left-1/2 top-1/2 size-[112px] -translate-x-1/2 -translate-y-1/2 rounded-full border border-white/8" />
      <div className="absolute left-1/2 top-1/2 size-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white/12" />
      <div className="absolute left-1/2 top-0 h-[22%] w-[44%] -translate-x-1/2 border-b border-l border-r border-white/8" />
      <div className="absolute left-1/2 top-0 h-[12%] w-[24%] -translate-x-1/2 border-b border-l border-r border-white/8" />
      <div className="absolute left-1/2 top-[16%] size-1.5 -translate-x-1/2 rounded-full bg-white/12" />
      <div className="absolute bottom-0 left-1/2 h-[22%] w-[44%] -translate-x-1/2 border-t border-l border-r border-white/8" />
      <div className="absolute bottom-0 left-1/2 h-[12%] w-[24%] -translate-x-1/2 border-t border-l border-r border-white/8" />
      <div className="absolute bottom-[16%] left-1/2 size-1.5 -translate-x-1/2 rounded-full bg-white/12" />
    </div>
  );
}

export default CommandCenterPitch;
