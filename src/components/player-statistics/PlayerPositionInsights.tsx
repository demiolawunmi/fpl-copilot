import { elementTypeToPosition, type FplBootstrapElement } from '../../api/fpl/fpl';
import { parseStatNumber } from '../../utils/playerStatsFormat';
import { DashboardCard, DashboardHeader } from '@/components/ui/primitives';

type PositionInsightsElement = Pick<
  FplBootstrapElement,
  | 'element_type'
  | 'minutes'
  | 'goals_scored'
  | 'clean_sheets'
  | 'saves'
  | 'bonus'
  | 'expected_goals'
  | 'expected_goal_involvements'
  | 'threat'
  | 'form'
>;

type PlayerPositionInsightsProps = {
  element: PositionInsightsElement;
};

type InsightStat = {
  label: string;
  value: string;
};

const PlayerPositionInsights = ({ element }: PlayerPositionInsightsProps) => {
  const position = elementTypeToPosition(element.element_type);
  const insightStats = buildInsightStats(position, element);

  return (
    <DashboardCard>
      <DashboardHeader
        title="Position Insights"
        description={`${position}-focused season indicators from official FPL stats.`}
      />
      <div className="grid grid-cols-1 gap-3 px-5 py-4 md:grid-cols-2">
        {insightStats.map((stat) => (
          <div
            key={stat.label}
            className="flex flex-col gap-1 rounded-lg border border-white/6 bg-[rgba(15,23,42,0.65)] px-3.5 py-3"
          >
            <span className="text-xs uppercase tracking-wide text-slate-500">
              {stat.label}
            </span>
            <span className="text-lg font-bold text-white">
              {stat.value}
            </span>
          </div>
        ))}
      </div>
    </DashboardCard>
  );
};

function buildInsightStats(position: 'GK' | 'DEF' | 'MID' | 'FWD', element: PositionInsightsElement): InsightStat[] {
  const minutes = parseStatNumber(element.minutes);
  const goals = parseStatNumber(element.goals_scored);
  const cleanSheets = parseStatNumber(element.clean_sheets);
  const saves = parseStatNumber(element.saves);
  const bonus = parseStatNumber(element.bonus);
  const xg = parseStatNumber(element.expected_goals);
  const xgi = parseStatNumber(element.expected_goal_involvements);
  const threat = parseStatNumber(element.threat);
  const form = parseStatNumber(element.form);

  if (position === 'GK') {
    return [
      { label: 'Saves', value: String(saves) },
      { label: 'Clean sheets', value: String(cleanSheets) },
      { label: 'Save rate', value: formatRate(saves, minutes) },
      { label: 'Minutes', value: String(minutes) },
    ];
  }

  if (position === 'DEF') {
    return [
      { label: 'Clean sheets', value: String(cleanSheets) },
      { label: 'Bonus', value: String(bonus) },
      { label: 'xGI', value: xgi.toFixed(2) },
      { label: 'Attacking xGI/90', value: formatRate(xgi, minutes) },
    ];
  }

  return [
    { label: 'Goals', value: String(goals) },
    { label: 'xG', value: xg.toFixed(2) },
    { label: 'Threat', value: threat.toFixed(0) },
    { label: 'Form', value: form.toFixed(1) },
  ];
}

function formatRate(stat: number, minutes: number): string {
  if (minutes <= 0) {
    return '0.00 / 90';
  }

  return `${((stat / minutes) * 90).toFixed(2)} / 90`;
}

export default PlayerPositionInsights;
