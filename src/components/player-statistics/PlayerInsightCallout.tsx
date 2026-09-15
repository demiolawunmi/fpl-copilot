import type { PlayerDetailFixture } from '../../hooks/usePlayerDetail';
import { parseStatNumber } from '../../utils/playerStatsFormat';
import { DashboardCard, DashboardHeader } from '@/components/ui/primitives';

type PlayerInsightCalloutProps = {
  form: number | string | null | undefined;
  fixtures: PlayerDetailFixture[];
};

const PlayerInsightCallout = ({ form, fixtures }: PlayerInsightCalloutProps) => {
  const insightText = buildInsightText(form, fixtures);

  return (
    <DashboardCard>
      <DashboardHeader
        title="Insight"
        description="Simple stat-based cue from recent form and immediate fixture run."
      />
      <div className="flex flex-col gap-2 px-5 py-4">
        <p className="text-sm leading-[1.6] text-slate-200">
          {insightText}
        </p>
      </div>
    </DashboardCard>
  );
};

function buildInsightText(
  form: number | string | null | undefined,
  fixtures: PlayerDetailFixture[],
): string {
  const formValue = parseStatNumber(form);
  const nextFixture = fixtures[0] ?? null;

  const formCue =
    formValue >= 6
      ? 'Recent form is strong.'
      : formValue >= 4
      ? 'Recent form is steady.'
      : 'Recent form is cooling off.';

  if (!nextFixture) {
    return `${formCue} No confirmed next fixture is available yet, so reassess once the schedule updates.`;
  }

  const difficulty = clampDifficulty(nextFixture.difficulty);
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

function clampDifficulty(value: number): number {
  if (value < 1) return 1;
  if (value > 5) return 5;
  return value;
}

export default PlayerInsightCallout;
