import type { EnhancedPlayer } from '../../data/commandCenterMocks';
import { DashboardCard, DashboardHeader } from '@/components/ui/primitives';

interface Props {
  squad: EnhancedPlayer[];
}

const SandboxCharts = ({ squad }: Props) => {
  const starters = squad.filter((p) => !p.isBench);
  const byPosition = {
    DEF: starters.filter((p) => p.position === 'DEF').reduce((sum, p) => sum + p.xPts, 0),
    MID: starters.filter((p) => p.position === 'MID').reduce((sum, p) => sum + p.xPts, 0),
    FWD: starters.filter((p) => p.position === 'FWD').reduce((sum, p) => sum + p.xPts, 0),
  };

  const totalXPts = Object.values(byPosition).reduce((sum, val) => sum + val, 0);

  return (
    <DashboardCard>
      <DashboardHeader title="Charts & Analytics" />
      <div className="flex flex-col gap-4 px-5 py-4">
        <div>
          <p className="mb-2 text-xs uppercase tracking-wide text-slate-400">
            Team xPts by Position
          </p>
          <div className="flex flex-col gap-2">
            {(Object.entries(byPosition) as [string, number][]).map(([pos, xPts]) => {
              const pct = totalXPts > 0 ? (xPts / totalXPts) * 100 : 0;
              return (
                <div key={pos} className="flex items-center gap-3">
                  <span className="w-8 text-xs text-slate-300">{pos}</span>
                  <div className="relative h-6 flex-1 overflow-hidden rounded-md bg-white/6">
                    <div
                      className="h-full border-r-2 border-emerald-400 bg-[rgba(16,185,129,0.3)]"
                      style={{ width: `${pct}%` }}
                    />
                    <span className="absolute inset-0 flex items-center justify-center text-xs font-semibold text-white">
                      {xPts.toFixed(1)} xPts
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="border-t border-white/6 pt-4">
          <p className="text-center text-xs italic text-slate-500">
            More charts coming soon: xPts trends, fixture difficulty, transfer impact
          </p>
        </div>
      </div>
    </DashboardCard>
  );
};

export default SandboxCharts;
