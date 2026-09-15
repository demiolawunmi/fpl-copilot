import { useMemo } from 'react';
import { Link as RouterLink, useLocation, useParams } from 'react-router-dom';
import PlayerFixturesPanel from '../components/player-statistics/PlayerFixturesPanel';
import PlayerFormTrend from '../components/player-statistics/PlayerFormTrend';
import PlayerHeroCard from '../components/player-statistics/PlayerHeroCard';
import { DashboardCard } from '@/components/ui/primitives';
import { Skeleton } from '@/components/ui/skeleton';
import { usePlayerDetail } from '../hooks/usePlayerDetail';
import { usePredictionsData } from '../hooks/usePredictionsData';

const PlayerDetailPage = () => {
  const location = useLocation();
  const { playerId } = useParams<{ playerId: string }>();
  const safePlayerId = playerId ?? '';
  const { loading, error, element, summary, historySorted, nextGameweekId } = usePlayerDetail(safePlayerId);
  const predictions = usePredictionsData(nextGameweekId);

  const back = useMemo(() => {
    const from = (location.state as { from?: string } | null)?.from;
    if (from === '/gw-overview') return { to: '/gw-overview', label: 'Back to GW overview' };
    if (from === '/command-center') return { to: '/command-center', label: 'Back to command center' };
    if (from === '/players') return { to: '/players', label: 'Back to players' };
    return { to: '/players', label: 'Back to players' };
  }, [location.state]);

  const predictionXp = useMemo(() => {
    if (!element) return null;
    const pred = predictions.lookupPrediction(element.web_name, element.teamShortName);
    if (pred != null && Number.isFinite(pred.xp)) return pred.xp;
    return null;
  }, [element, predictions]);

  return (
    <div className="mx-auto flex w-full max-w-[90rem] flex-1 flex-col px-4 py-6 md:px-6 xl:px-10 xl:py-8">
      <div className="flex flex-col gap-6">
        <RouterLink
          to={back.to}
          className="inline-flex h-8 shrink-0 items-center self-start rounded-lg px-3 text-sm font-semibold text-slate-200 transition-colors hover:bg-white/6 hover:text-white"
        >
          {back.label}
        </RouterLink>

        {loading ? (
          <div className="flex w-full flex-col gap-4">
            <Skeleton className="h-[220px] w-full rounded-2xl" />
            <div className="grid w-full grid-cols-1 gap-4 md:grid-cols-2">
              <Skeleton className="h-[280px] w-full rounded-2xl" />
              <Skeleton className="h-[280px] w-full rounded-2xl" />
            </div>
          </div>
        ) : null}

        {!loading && error ? (
          <div
            role="alert"
            className="rounded-xl border bg-[rgba(234,179,8,0.08)] px-4 py-3"
            style={{ borderColor: 'rgba(234, 179, 8, 0.2)' }}
          >
            <p className="text-sm text-yellow-300">
              Couldn&apos;t load player detail. {error}
            </p>
          </div>
        ) : null}

        {!loading && !error && (!element || !summary) ? (
          <DashboardCard className="px-5 py-4">
            <p className="text-sm text-slate-300">Player detail is unavailable right now.</p>
          </DashboardCard>
        ) : null}

        {!loading && !error && element && summary ? (
          <div className="flex w-full flex-col gap-4">
            <PlayerHeroCard
              key={element.id}
              element={element}
              expectedPoints={predictionXp}
              fixtures={summary.fixtures}
            />

            <div className="grid w-full grid-cols-1 items-stretch gap-4 md:grid-cols-2">
              <PlayerFormTrend history={historySorted} />
              <PlayerFixturesPanel fixtures={summary.fixtures} />
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
};

export default PlayerDetailPage;
