import type { PlayerDetailHistory } from '../../hooks/usePlayerDetail';
import { DashboardCard, DashboardHeader } from '@/components/ui/primitives';

type PlayerFormTrendProps = {
  history: PlayerDetailHistory[];
  maxItems?: number;
};

type TrendRow = {
  id: string;
  roundLabel: string;
  points: number;
};

const DEFAULT_MAX_ITEMS = 5;

const PlayerFormTrend = ({ history, maxItems = DEFAULT_MAX_ITEMS }: PlayerFormTrendProps) => {
  const boundedCount = Math.max(1, maxItems);
  const latestSlice = history.slice(-Math.min(boundedCount, history.length));
  const trendRows: TrendRow[] = latestSlice.map((entry) => ({
    id: `${entry.fixture}-${entry.round}`,
    roundLabel: `GW ${entry.round}`,
    points: entry.total_points,
  }));

  const maxAbsolutePoints = trendRows.reduce(
    (max, row) => Math.max(max, Math.abs(row.points)),
    0,
  );
  const scaleBase = maxAbsolutePoints > 0 ? maxAbsolutePoints : 1;

  return (
    <DashboardCard>
      <DashboardHeader
        title="Form Trend"
        description="Last up to 5 gameweeks by points (ordered oldest to latest)."
      />

      <div className="flex flex-col gap-3 px-5 py-4">
        {trendRows.length === 0 ? (
          <p className="text-sm text-slate-400">
            No recent gameweek history is available for this player yet.
          </p>
        ) : (
          trendRows.map((row) => {
            const widthPercent = Math.max((Math.abs(row.points) / scaleBase) * 100, 6);
            const isPositive = row.points >= 0;

            return (
              <div key={row.id} className="flex flex-col gap-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-semibold text-slate-300">
                    {row.roundLabel}
                  </span>
                  <span
                    className={`text-sm font-bold ${isPositive ? 'text-green-300' : 'text-red-300'}`}
                  >
                    {row.points} pts
                  </span>
                </div>
                <div className="h-9 overflow-hidden rounded-lg border border-white/6 bg-[rgba(15,23,42,0.72)]">
                  <div
                    className={`h-full border-r transition-[width] duration-200 ease-[ease] ${
                      isPositive
                        ? 'border-r-green-300 bg-[rgba(16,185,129,0.42)]'
                        : 'border-r-red-300 bg-[rgba(248,113,113,0.38)]'
                    }`}
                    style={{ width: `${widthPercent}%` }}
                  />
                </div>
              </div>
            );
          })
        )}
      </div>
    </DashboardCard>
  );
};

export default PlayerFormTrend;
