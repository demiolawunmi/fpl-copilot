import { useState } from 'react';
import {
  elementTypeToPosition,
  type FplBootstrapElement,
} from '../../api/fpl/fpl';
import { fplEndpoints } from '../../api/fpl/endpoints';
import PlayerHeadshot from './PlayerHeadshot';
import {
  formatOwnershipPercent,
  formatPriceFromNowCost,
  parseStatNumber,
} from '../../utils/playerStatsFormat';
import type { PlayerDetailFixture } from '../../hooks/usePlayerDetail';
import { DashboardCard } from '@/components/ui/primitives';

/** Matches `PlayerHeadshot` hero size so photo and team crest read as a pair. */
const HERO_VISUAL_SIZE = 'size-24';

type PlayerHeroElement = Pick<
  FplBootstrapElement,
  | 'id'
  | 'code'
  | 'web_name'
  | 'element_type'
  | 'status'
  | 'now_cost'
  | 'selected_by_percent'
  | 'total_points'
  | 'chance_of_playing_this_round'
  | 'chance_of_playing_next_round'
  | 'form'
  | 'minutes'
  | 'goals_scored'
  | 'assists'
  | 'expected_goal_involvements'
  | 'bonus'
  | 'ep_this'
  | 'ep_next'
  | 'expected_goals'
  | 'clean_sheets'
  | 'saves'
  | 'threat'
> & {
  teamName: string;
  teamShortName: string;
  teamCode?: number | null;
};

export type PlayerHeroCardProps = {
  element: PlayerHeroElement;
  teamBadgeLabel?: string;
  expectedPoints?: number | null;
  fixtures?: PlayerDetailFixture[];
};

function TeamBadgeSquare({ teamCode, abbr }: { teamCode: number | null | undefined; abbr: string }) {
  const [failed, setFailed] = useState(false);
  const url = teamCode != null ? fplEndpoints.teamBadge(teamCode) : undefined;

  return (
    <div className={`${HERO_VISUAL_SIZE} flex shrink-0 items-center justify-center overflow-hidden rounded-xl bg-white p-2`}>
      {url && !failed ? (
        <img
          src={url}
          alt={`${abbr} badge`}
          className="max-h-full max-w-full h-auto w-auto object-contain"
          loading="lazy"
          onError={() => setFailed(true)}
        />
      ) : (
        <span className="text-sm font-bold uppercase text-slate-700">
          {abbr.slice(0, 3)}
        </span>
      )}
    </div>
  );
}

const PlayerHeroCard = ({ element, teamBadgeLabel, expectedPoints, fixtures = [] }: PlayerHeroCardProps) => {
  const ownership = parseStatNumber(element.selected_by_percent);
  const formValue = parseStatNumber(element.form);
  const availability = resolveAvailability(element);
  const ruleTags = getRuleTags(ownership, formValue);

  const epBootstrap = parseStatNumber(element.ep_next ?? element.ep_this);
  const xPtsDisplay =
    expectedPoints != null && Number.isFinite(expectedPoints) ? expectedPoints : epBootstrap;

  const overviewStats = [
    { label: 'Total points', value: String(element.total_points ?? 0) },
    { label: 'xPts', value: xPtsDisplay.toFixed(1) },
    { label: 'Minutes', value: String(element.minutes ?? 0) },
    { label: 'Goals', value: String(element.goals_scored ?? 0) },
    { label: 'Assists', value: String(element.assists ?? 0) },
    {
      label: 'xGI',
      value: parseStatNumber(element.expected_goal_involvements).toFixed(2),
    },
    { label: 'Bonus', value: String(element.bonus ?? 0) },
  ];

  const position = elementTypeToPosition(element.element_type);
  const positionStats = buildPositionInsightStats(position, element);
  const insightText = buildInsightText(formValue, fixtures);

  return (
    <div className="flex flex-col gap-4">
      <DashboardCard className="overflow-hidden px-4 py-4 md:px-6 md:py-5">
        <div className="flex flex-col items-stretch gap-6 lg:flex-row lg:items-start lg:gap-8">
          {/* Left: photo + team crest + identity */}
          <div className="flex min-w-0 flex-col items-stretch gap-4 sm:flex-row sm:items-start sm:gap-5 lg:flex-[1.15]">
            <div className="flex shrink-0 flex-col items-center gap-3">
              <div className="translate-x-1 pl-1 pr-0.5">
                <PlayerHeadshot code={element.code} name={element.web_name} size="hero" />
              </div>
              <TeamBadgeSquare teamCode={element.teamCode} abbr={teamBadgeLabel ?? element.teamShortName} />
            </div>

            <div className="flex min-w-0 flex-1 flex-col gap-3">
              <div className="flex flex-col gap-1.5">
                <span className="truncate text-2xl font-bold text-white md:text-3xl">
                  {element.web_name}
                </span>

                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded-md bg-white/8 px-2.5 py-1 text-xs font-medium text-slate-100 normal-case">
                    {teamBadgeLabel ?? element.teamShortName}
                  </span>
                  <span className="rounded-md bg-white/6 px-2.5 py-1 text-xs font-medium text-slate-200 normal-case">
                    {element.teamName}
                  </span>
                  <span className="rounded-md bg-emerald-500 px-2.5 py-1 text-xs font-medium text-slate-950 normal-case">
                    {position}
                  </span>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-5">
                <MetricText label="Price" value={formatPriceFromNowCost(element.now_cost)} />
                <MetricText label="Ownership" value={formatOwnershipPercent(element.selected_by_percent)} />
                <MetricText label="Total points" value={String(element.total_points ?? 0)} />
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <span
                  className={`rounded-md border px-2.5 py-1 text-xs font-medium normal-case ${availability.bg} ${availability.color} ${availability.borderColor}`}
                >
                  {availability.label}
                </span>
                {availability.chanceText ? (
                  <span className="text-sm text-slate-300">
                    {availability.chanceText}
                  </span>
                ) : null}
              </div>

              {ruleTags.length > 0 ? (
                <div className="flex flex-col gap-2 pt-1">
                  {ruleTags.map((tag) => (
                    <div key={tag.label} className="flex flex-wrap items-center gap-3">
                      <span
                        className={`shrink-0 rounded-md border px-2.5 py-1 text-xs font-medium normal-case ${tag.bg} ${tag.color} ${tag.borderColor}`}
                      >
                        {tag.label}
                      </span>
                      <span className="text-sm leading-[1.45] text-slate-400">
                        {tag.helperText}
                      </span>
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
          </div>

          {/* Right: narrative insight + position stat tiles */}
          <div className="flex min-w-0 flex-col gap-4 border-t border-white/6 pt-4 lg:flex-1 lg:border-l lg:border-t-0 lg:pl-2 lg:pt-0">
            <div className="flex flex-col gap-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                Insight
              </span>
              <p className="text-sm leading-[1.6] text-slate-200">
                {insightText}
              </p>
            </div>

            <div className="flex flex-col gap-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                {position} insights
              </span>
              <div className="grid grid-cols-2 gap-3">
                {positionStats.map((stat) => (
                  <div
                    key={stat.label}
                    className="flex flex-col gap-0.5 rounded-lg border border-white/6 bg-[rgba(15,23,42,0.65)] px-3 py-2.5"
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
            </div>
          </div>
        </div>
      </DashboardCard>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-7">
        {overviewStats.map((stat) => (
          <DashboardCard key={stat.label} className="px-4 py-3">
            <span className="text-xs uppercase tracking-wide text-slate-500">
              {stat.label}
            </span>
            <p className="mt-1 text-xl font-bold text-white">
              {stat.value}
            </p>
          </DashboardCard>
        ))}
      </div>
    </div>
  );
};

type MetricTextProps = { label: string; value: string };

function MetricText({ label, value }: MetricTextProps) {
  return (
    <div>
      <span className="block text-xs uppercase tracking-wide text-slate-500">
        {label}
      </span>
      <span className="text-base font-semibold text-white">
        {value}
      </span>
    </div>
  );
}

type AvailabilityState = {
  label: string;
  chanceText: string | null;
  bg: string;
  color: string;
  borderColor: string;
};

function resolveAvailability(element: PlayerHeroElement): AvailabilityState {
  const status = element.status ?? 'u';
  const chance =
    element.chance_of_playing_this_round ??
    element.chance_of_playing_next_round ??
    null;

  if (status === 'a' && (chance == null || chance >= 100)) {
    return {
      label: 'Available',
      chanceText: chance != null ? `${chance}% chance this GW` : null,
      bg: 'bg-[rgba(16,185,129,0.12)]',
      color: 'text-green-300',
      borderColor: 'border-[rgba(16,185,129,0.22)]',
    };
  }

  if (status === 's') {
    return {
      label: 'Suspended',
      chanceText: chance != null ? `${chance}% chance next GW` : null,
      bg: 'bg-[rgba(251,146,60,0.12)]',
      color: 'text-orange-300',
      borderColor: 'border-[rgba(251,146,60,0.22)]',
    };
  }

  if (status === 'd' || (chance != null && chance > 0 && chance < 100)) {
    return {
      label: 'Doubtful',
      chanceText: chance != null ? `${chance}% chance this GW` : null,
      bg: 'bg-[rgba(250,204,21,0.12)]',
      color: 'text-yellow-300',
      borderColor: 'border-[rgba(250,204,21,0.22)]',
    };
  }

  if (status === 'i' || status === 'u' || status === 'n') {
    return {
      label: 'Unavailable',
      chanceText: chance != null ? `${chance}% chance this GW` : null,
      bg: 'bg-[rgba(248,113,113,0.12)]',
      color: 'text-red-300',
      borderColor: 'border-[rgba(248,113,113,0.22)]',
    };
  }

  return {
    label: 'Status unknown',
    chanceText: chance != null ? `${chance}% chance this GW` : null,
    bg: 'bg-[rgba(148,163,184,0.15)]',
    color: 'text-slate-200',
    borderColor: 'border-[rgba(148,163,184,0.2)]',
  };
}

type RuleTag = {
  label: string;
  helperText: string;
  bg: string;
  color: string;
  borderColor: string;
};

function getRuleTags(ownership: number, formValue: number): RuleTag[] {
  const tags: RuleTag[] = [];

  if (ownership > 30) {
    tags.push({
      label: 'Template',
      helperText: 'High ownership across active managers.',
      bg: 'bg-[rgba(56,189,248,0.12)]',
      color: 'text-cyan-300',
      borderColor: 'border-[rgba(56,189,248,0.24)]',
    });
  }

  if (ownership < 5) {
    tags.push({
      label: 'Differential',
      helperText: 'Low ownership provides an edge to climb ranks.',
      bg: 'bg-[rgba(20,184,166,0.12)]',
      color: 'text-teal-300',
      borderColor: 'border-[rgba(20,184,166,0.24)]',
    });
  }

  if (formValue > 6) {
    tags.push({
      label: 'Hot form',
      helperText: 'Averaging high points in recent matches.',
      bg: 'bg-[rgba(251,146,60,0.12)]',
      color: 'text-orange-300',
      borderColor: 'border-[rgba(251,146,60,0.24)]',
    });
  }

  return tags;
}

type InsightStat = { label: string; value: string };

function buildPositionInsightStats(position: 'GK' | 'DEF' | 'MID' | 'FWD', element: PlayerHeroElement): InsightStat[] {
  const minutes = parseStatNumber(element.minutes);
  const goals = parseStatNumber(element.goals_scored);
  const cleanSheets = parseStatNumber(element.clean_sheets);
  const saves = parseStatNumber(element.saves);
  const bonus = parseStatNumber(element.bonus);
  const xg = parseStatNumber(element.expected_goals);
  const xgi = parseStatNumber(element.expected_goal_involvements);
  const threat = parseStatNumber(element.threat);
  const form = parseStatNumber(element.form);

  const per90 = (stat: number) => (minutes > 0 ? `${((stat / minutes) * 90).toFixed(2)} / 90` : '0.00 / 90');

  if (position === 'GK') {
    return [
      { label: 'Saves', value: String(saves) },
      { label: 'Clean sheets', value: String(cleanSheets) },
      { label: 'Save rate', value: per90(saves) },
      { label: 'Minutes', value: String(minutes) },
    ];
  }

  if (position === 'DEF') {
    return [
      { label: 'Clean sheets', value: String(cleanSheets) },
      { label: 'Bonus', value: String(bonus) },
      { label: 'xGI', value: xgi.toFixed(2) },
      { label: 'Attacking xGI/90', value: per90(xgi) },
    ];
  }

  return [
    { label: 'Goals', value: String(goals) },
    { label: 'xG', value: xg.toFixed(2) },
    { label: 'Threat', value: threat.toFixed(0) },
    { label: 'Form', value: form.toFixed(1) },
  ];
}

function buildInsightText(formValue: number, fixtures: PlayerDetailFixture[]): string {
  const formCue =
    formValue >= 6
      ? 'Recent form is strong.'
      : formValue >= 4
        ? 'Recent form is steady.'
        : 'Recent form is cooling off.';

  const nextFixture = fixtures[0] ?? null;
  if (!nextFixture) {
    return `${formCue} No confirmed next fixture is available yet, so reassess once the schedule updates.`;
  }

  const difficulty = Math.min(5, Math.max(1, nextFixture.difficulty));
  const venue = nextFixture.is_home ? 'at home' : 'away';
  const opponent = nextFixture.opponentTeamShortName || `Team ${nextFixture.opponent_team}`;

  const fixtureCue =
    difficulty <= 2
      ? 'The matchup rates favorable for attacking returns.'
      : difficulty === 3
        ? 'The matchup looks balanced, so expect a moderate ceiling.'
        : 'The matchup is difficult, so expectations should be tempered.';

  return `${formCue} Next up is ${opponent} ${venue} (FDR ${difficulty}). ${fixtureCue}`;
}

export default PlayerHeroCard;
