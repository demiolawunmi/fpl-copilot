import { useState } from 'react';
import type { PlayerDetailFixture } from '../../hooks/usePlayerDetail';
import { DashboardCard, DashboardHeader } from '@/components/ui/primitives';

type PlayerFixturesPanelProps = {
  fixtures: PlayerDetailFixture[];
  maxItems?: number;
};

const DEFAULT_MAX_ITEMS = 8;

const PlayerFixturesPanel = ({ fixtures, maxItems = DEFAULT_MAX_ITEMS }: PlayerFixturesPanelProps) => {
  const visibleFixtures = fixtures.slice(0, Math.max(1, maxItems));
  const nextFive = fixtures.slice(0, 5);
  const averageDifficulty =
    nextFive.length > 0
      ? nextFive.reduce((total, fixture) => total + normalizeDifficulty(fixture.difficulty), 0) / nextFive.length
      : null;

  return (
    <DashboardCard>
      <DashboardHeader
        title="Upcoming Fixtures"
        description="Next opponents, venue, and fixture difficulty."
      />

      <div className="card-scroll flex max-h-96 flex-col gap-3 overflow-y-auto px-5 py-4">
        {averageDifficulty != null ? (
          <p className="text-sm text-slate-300">
            Next 5 outlook: {averageDifficulty.toFixed(1)} average FDR ({difficultyOutlookLabel(averageDifficulty)})
          </p>
        ) : null}

        {visibleFixtures.length === 0 ? (
          <p className="text-sm text-slate-400">
            No upcoming fixtures are available yet.
          </p>
        ) : (
          visibleFixtures.map((fixture) => {
            const style = getDifficultyStyle(normalizeDifficulty(fixture.difficulty));
            const opponentLabel = fixture.opponentTeamShortName || fixture.opponentTeamName || `Team ${fixture.opponent_team}`;
            const kickoffLabel = formatKickoffTime(fixture.kickoff_time);

            return (
              <div
                key={`${fixture.id}-${fixture.event ?? 'e'}-${fixture.kickoff_time ?? ''}`}
                className="flex items-center justify-between gap-3 rounded-lg bg-[rgba(30,41,59,0.3)] px-3 py-2.5 hover:bg-[rgba(30,41,59,0.6)]"
              >
                <div className="flex min-w-0 items-center gap-2">
                  <OpponentBadge
                    abbr={opponentLabel}
                    badgeUrl={fixture.opponentBadgeUrl}
                  />
                  <span className="truncate text-sm font-semibold text-white">
                    {opponentLabel}
                  </span>
                  <span className="rounded-md bg-white/8 px-2 py-0.5 text-[10px] font-medium uppercase text-slate-100">
                    {fixture.is_home ? 'H' : 'A'}
                  </span>
                </div>

                <div className="flex shrink-0 items-center gap-2">
                  {fixture.event != null ? (
                    <span className="whitespace-nowrap text-xs text-slate-500">
                      GW {fixture.event}
                    </span>
                  ) : null}
                  {kickoffLabel ? (
                    <span className="whitespace-nowrap text-xs text-slate-400">
                      {kickoffLabel}
                    </span>
                  ) : null}
                  <span
                    className={`rounded-md border px-2 py-1 text-[10px] font-medium ${style.bg} ${style.color} ${style.borderColor}`}
                  >
                    FDR {normalizeDifficulty(fixture.difficulty)}
                  </span>
                </div>
              </div>
            );
          })
        )}
      </div>
    </DashboardCard>
  );
};

function normalizeDifficulty(difficulty: number): number {
  if (difficulty < 1) return 1;
  if (difficulty > 5) return 5;
  return difficulty;
}

function difficultyOutlookLabel(averageDifficulty: number): string {
  if (averageDifficulty <= 2) return 'favorable';
  if (averageDifficulty <= 3) return 'balanced';
  if (averageDifficulty <= 4) return 'challenging';
  return 'very tough';
}

function formatKickoffTime(kickoffTime: string | null): string | null {
  if (!kickoffTime) {
    return null;
  }

  const parsed = new Date(kickoffTime);
  if (Number.isNaN(parsed.getTime())) {
    return null;
  }

  return new Intl.DateTimeFormat('en-GB', {
    day: '2-digit',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
  }).format(parsed);
}

function getDifficultyStyle(difficulty: number) {
  if (difficulty === 1) {
    return {
      bg: 'bg-[rgba(16,185,129,0.12)]',
      color: 'text-emerald-400',
      borderColor: 'border-[rgba(16,185,129,0.22)]',
    };
  }
  if (difficulty === 2) {
    return {
      bg: 'bg-[rgba(34,197,94,0.12)]',
      color: 'text-green-300',
      borderColor: 'border-[rgba(34,197,94,0.22)]',
    };
  }
  if (difficulty === 3) {
    return {
      bg: 'bg-[rgba(100,116,139,0.12)]',
      color: 'text-slate-300',
      borderColor: 'border-[rgba(100,116,139,0.22)]',
    };
  }
  if (difficulty === 4) {
    return {
      bg: 'bg-[rgba(251,146,60,0.12)]',
      color: 'text-orange-300',
      borderColor: 'border-[rgba(251,146,60,0.22)]',
    };
  }
  return {
    bg: 'bg-[rgba(248,113,113,0.12)]',
    color: 'text-red-300',
    borderColor: 'border-[rgba(248,113,113,0.22)]',
  };
}

function OpponentBadge({ abbr, badgeUrl }: { abbr: string; badgeUrl?: string }) {
  const [failed, setFailed] = useState(false);
  if (badgeUrl && !failed) {
    return (
      <img
        src={badgeUrl}
        alt={abbr}
        className="size-6 shrink-0 rounded-full bg-white/4 object-contain"
        loading="lazy"
        onError={() => setFailed(true)}
      />
    );
  }
  return (
    <div className="flex size-6 shrink-0 items-center justify-center rounded-full bg-slate-600 text-[7px] font-bold text-white">
      {abbr.slice(0, 3)}
    </div>
  );
}

export default PlayerFixturesPanel;
