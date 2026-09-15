import type { GWStats } from '../../data/gwOverviewMocks';
import { DashboardCard } from '@/components/ui/primitives';

interface Props {
  stats: GWStats;
}

const format = (n: number) => n.toLocaleString();

const StatsStrip = ({ stats }: Props) => {
  const items = [
    { label: 'Average', value: format(stats.average) },
    { label: 'Highest', value: format(stats.highest) },
    { label: 'GW Points', value: format(stats.gwPoints), highlight: true },
    { label: 'GW Rank', value: format(stats.gwRank) },
    { label: 'Overall Rank', value: format(stats.overallRank) },
  ];

  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
      {items.map((item) => (
        <DashboardCard key={item.label} className="px-4 py-3">
          <div className="text-center">
            <p className="text-xs tracking-wide uppercase text-slate-400">
              {item.label}
            </p>
            <p className={`mt-1 text-lg ${item.highlight ? 'text-emerald-400' : 'text-white'}`}>
              {item.value}
            </p>
          </div>
        </DashboardCard>
      ))}
    </div>
  );
};

export default StatsStrip;
