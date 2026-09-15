import type { FplBootstrapElement } from '../../api/fpl/fpl';
import {
  getTopPlayersByMetric,
  parseStatNumber,
  type PlayerLeaderboardRow,
  type TeamAbbreviationMap,
} from '../../utils/playerStatsFormat';
import { DashboardCard, DashboardHeader } from '@/components/ui/primitives';
import PlayerHeadshot from './PlayerHeadshot';

export type LeaderboardCardKey = 'goals' | 'assists' | 'xg' | 'xgi' | 'cleanSheets';

export type PlayerLeaderboardCardEntry = PlayerLeaderboardRow & {
  valueLabel: string;
};

export type PlayerLeaderboardCardData = {
  key: LeaderboardCardKey;
  title: string;
  metricLabel: string;
  leaders: PlayerLeaderboardCardEntry[];
  contextText?: string;
};

export type PlayerLeaderboardCardsProps = {
  cards: PlayerLeaderboardCardData[];
};

export function buildPlayerLeaderboardCardsData(input: {
  elements: FplBootstrapElement[];
  teamMap?: TeamAbbreviationMap;
  topN?: number;
}): PlayerLeaderboardCardData[] {
  const topN = input.topN ?? 3;
  const outfieldElements = input.elements.filter((element) => element.element_type !== 1);
  const gkElements = input.elements.filter((element) => element.element_type === 1);

  const goalsLeaders = toCardEntries(
    getTopPlayersByMetric(input.elements, {
      metric: (element) => element.goals_scored,
      teamMap: input.teamMap,
      topN,
    }),
    (value) => formatIntegerMetric(value)
  );

  const assistsLeaders = toCardEntries(
    getTopPlayersByMetric(input.elements, {
      metric: (element) => element.assists,
      teamMap: input.teamMap,
      topN,
    }),
    (value) => formatIntegerMetric(value)
  );

  const xgLeaders = toCardEntries(
    getTopPlayersByMetric(input.elements, {
      metric: (element) => element.expected_goals,
      teamMap: input.teamMap,
      topN,
    }),
    (value) => value.toFixed(2)
  );

  const xgiLeaders = toCardEntries(
    getTopPlayersByMetric(input.elements, {
      metric: (element) => element.expected_goal_involvements,
      teamMap: input.teamMap,
      topN,
    }),
    (value) => value.toFixed(2)
  );

  const cleanSheetLeaders = toCardEntries(
    getTopPlayersByMetric(outfieldElements, {
      metric: (element) => element.clean_sheets,
      teamMap: input.teamMap,
      topN,
    }),
    (value) => formatIntegerMetric(value)
  );

  const topGkBySaves = getTopPlayersByMetric(gkElements, {
    metric: (element) => element.saves,
    teamMap: input.teamMap,
    topN: 1,
  })[0];

  const gkSavesContext = topGkBySaves
    ? `GK saves leader: ${topGkBySaves.name} (${topGkBySaves.teamAbbr}) ${formatIntegerMetric(topGkBySaves.metric)}`
    : 'GK saves leader unavailable';

  return [
    {
      key: 'goals',
      title: 'Goals',
      metricLabel: 'Season goals',
      leaders: goalsLeaders,
    },
    {
      key: 'assists',
      title: 'Assists',
      metricLabel: 'Season assists',
      leaders: assistsLeaders,
    },
    {
      key: 'xg',
      title: 'xG',
      metricLabel: 'Expected goals',
      leaders: xgLeaders,
    },
    {
      key: 'xgi',
      title: 'xGI',
      metricLabel: 'Expected goal involvement',
      leaders: xgiLeaders,
    },
    {
      key: 'cleanSheets',
      title: 'Clean sheets',
      metricLabel: 'Outfield clean sheets',
      leaders: cleanSheetLeaders,
      contextText: gkSavesContext,
    },
  ];
}

const PlayerLeaderboardCards = ({ cards }: PlayerLeaderboardCardsProps) => {
  return (
    <div className="card-scroll overflow-x-auto pb-2">
      <div className="flex min-w-max items-stretch gap-4">
        {cards.map((card) => (
          <DashboardCard key={card.key} className="w-[260px] shrink-0 md:w-[280px]">
            <DashboardHeader title={card.title} description={card.metricLabel} />
            <div className="flex flex-col gap-2.5 px-5 py-4">
              {card.leaders.map((leader, index) => {
                const rank = index + 1;
                const isTopRank = rank === 1;

                return (
                  <div
                    key={`${card.key}-${leader.id}-${rank}`}
                    className={`flex items-center justify-between gap-3 rounded-lg border px-3 py-2.5 ${
                      isTopRank
                        ? 'border-emerald-400 bg-[rgba(56,189,248,0.09)]'
                        : 'border-white/8 bg-[rgba(30,41,59,0.36)]'
                    }`}
                  >
                    <div className="flex min-w-0 items-center gap-2.5">
                      <span
                        className={`rounded-md px-2 py-0.5 text-[10px] font-medium uppercase ${
                          isTopRank ? 'bg-emerald-500 text-slate-950' : 'bg-white/12 text-slate-200'
                        }`}
                      >
                        #{rank}
                      </span>
                      <PlayerHeadshot
                        code={leader.photoCode}
                        name={leader.name}
                        size={isTopRank ? 'md' : 'sm'}
                      />
                      <div className="min-w-0">
                        <p
                          className={`truncate text-white ${
                            isTopRank ? 'text-sm font-bold' : 'text-xs font-semibold'
                          }`}
                        >
                          {leader.name}
                        </p>
                        <p className="text-xs text-slate-400">
                          {leader.teamAbbr}
                        </p>
                      </div>
                    </div>
                    <span
                      className={`${isTopRank ? 'text-base font-extrabold text-emerald-300' : 'text-sm font-bold text-slate-200'}`}
                    >
                      {leader.valueLabel}
                    </span>
                  </div>
                );
              })}
              {card.contextText ? (
                <p className="pt-1 text-xs text-slate-400">
                  {card.contextText}
                </p>
              ) : null}
            </div>
          </DashboardCard>
        ))}
      </div>
    </div>
  );
};

function toCardEntries(
  rows: PlayerLeaderboardRow[],
  valueFormatter: (value: number) => string
): PlayerLeaderboardCardEntry[] {
  return rows.map((row) => {
    const metric = parseStatNumber(row.metric);
    return {
      ...row,
      metric,
      valueLabel: valueFormatter(metric),
    };
  });
}

function formatIntegerMetric(value: number): string {
  return String(Math.round(value));
}

export default PlayerLeaderboardCards;
