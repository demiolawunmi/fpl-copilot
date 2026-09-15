import type { FixtureItem } from '../../data/commandCenterMocks';
import { DashboardCard, DashboardHeader } from '@/components/ui/primitives';
import { Badge } from '@/components/ui/badge';

interface Props {
  fixtures: FixtureItem[];
}

const getDifficultyStyle = (difficulty: number) => {
  if (difficulty === 1) return 'bg-[rgba(16,185,129,0.12)] text-emerald-400 border-[rgba(16,185,129,0.22)]';
  if (difficulty === 2) return 'bg-[rgba(34,197,94,0.12)] text-green-300 border-[rgba(34,197,94,0.22)]';
  if (difficulty === 3) return 'bg-[rgba(100,116,139,0.12)] text-slate-300 border-[rgba(100,116,139,0.22)]';
  if (difficulty === 4) return 'bg-[rgba(251,146,60,0.12)] text-orange-300 border-[rgba(251,146,60,0.22)]';
  return 'bg-[rgba(248,113,113,0.12)] text-red-300 border-[rgba(248,113,113,0.22)]';
};

const FixturesSnapshot = ({ fixtures }: Props) => {
  return (
    <DashboardCard>
      <DashboardHeader title="Fixtures Snapshot" description="Upcoming fixtures for your squad" />
      <div className="card-scroll flex max-h-80 flex-col gap-2 overflow-y-auto px-5 py-4">
        {fixtures.map((fixture, idx) => {
          const style = getDifficultyStyle(fixture.difficulty);
          return (
            <div
              key={idx}
              className="flex items-center justify-between gap-3 rounded-lg bg-[rgba(30,41,59,0.3)] px-3 py-2 hover:bg-[rgba(30,41,59,0.6)]"
            >
              <div className="flex min-w-0 flex-1 items-center gap-2">
                <Badge className={`rounded-md px-2 py-1 text-[10px] normal-case ${style}`}>
                  {fixture.difficulty}
                </Badge>
                <span className="text-xs text-slate-400">
                  GW{fixture.gameweek}
                </span>
                <span className="line-clamp-1 text-sm font-medium text-white">
                  {fixture.home ? 'vs' : '@'} {fixture.opponentAbbr}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </DashboardCard>
  );
};

export default FixturesSnapshot;
