import { FiArrowDown, FiArrowUp, FiChevronRight } from 'react-icons/fi';
import type { RecommendedTransfer, RecommendedTransferPlayer } from '../../data/gwOverviewMocks';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { DashboardCard, DashboardHeader } from '@/components/ui/primitives';

interface Props {
  transfers: RecommendedTransfer[];
  heightPx?: number;
}

const PlayerInfo = ({
  player,
  direction,
}: {
  player: RecommendedTransferPlayer;
  direction: 'in' | 'out';
}) => (
  <div className="flex min-w-0 items-center gap-2.5">
    <div
      className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full ${direction === 'in' ? 'bg-[rgba(16,185,129,0.12)] text-emerald-400' : 'bg-[rgba(248,113,113,0.12)] text-red-300'}`}
    >
      {direction === 'in' ? <FiArrowUp size={14} /> : <FiArrowDown size={14} />}
    </div>
    <div className="min-w-0">
      <p className="line-clamp-1 text-sm font-medium text-white">
        {player.name}
      </p>
      <p className="text-[11px] text-slate-500">
        {player.team} · {player.position} · {player.price}
      </p>
    </div>
  </div>
);

const RecommendedTransfersCard = ({ transfers, heightPx }: Props) => (
  <DashboardCard
    className="flex flex-col"
    style={heightPx ? { height: `${heightPx}px` } : undefined}
  >
    <DashboardHeader title="Recommended Transfers" />

    <div className="card-scroll flex flex-1 flex-col overflow-auto">
      {transfers.map((t, i) => (
        <div
          key={i}
          className={`px-5 py-4 hover:bg-white/4 ${i === transfers.length - 1 ? '' : 'border-b border-white/6'}`}
        >
          <div className="flex items-center justify-between gap-4">
            <div className="flex min-w-0 flex-1 flex-col gap-2.5">
              <PlayerInfo player={t.playerIn} direction="in" />
              <PlayerInfo player={t.playerOut} direction="out" />
            </div>

            <div className="flex shrink-0 flex-col items-center gap-1">
              <span className="text-[10px] uppercase text-slate-500">
                xPts
              </span>
              <Badge className="rounded-lg border border-[rgba(16,185,129,0.22)] bg-[rgba(16,185,129,0.12)] px-2.5 py-1 normal-case text-emerald-400">
                +{t.xPointsDiff.toFixed(1)}
              </Badge>
            </div>
          </div>

          <div className="mt-3 flex items-center justify-between gap-3">
            <span className="line-clamp-1 flex-1 text-[11px] leading-snug text-slate-500">
              {t.rationale}
            </span>

            <div className="flex shrink-0 items-center gap-2">
              <Button
                size="sm"
                variant="outline"
                className="h-6 border-[rgba(16,185,129,0.22)] px-2 text-xs text-emerald-400 hover:bg-[rgba(16,185,129,0.12)]"
              >
                Apply
              </Button>
              <Button size="sm" variant="ghost" disabled className="h-6 px-2 text-xs text-slate-600">
                <FiChevronRight size={16} />
              </Button>
            </div>
          </div>
        </div>
      ))}
    </div>
  </DashboardCard>
);

export default RecommendedTransfersCard;
