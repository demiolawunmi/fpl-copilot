import { useMemo, useState } from 'react';
import { Loader2 } from 'lucide-react';
import GWHeader from '../components/gw-overview/GWHeader';
import FixturesCard from '../components/gw-overview/FixturesCard';
import FdrMatrixTable from '../components/fixtures-page/FdrMatrixTable';
import TeamEloTable from '../components/fixtures-page/TeamEloTable';
import { mockFixtures } from '../data/gwOverviewMocks';
import { useTeamId } from '../context/TeamIdContext';
import { useFplData } from '../hooks/useFplData';
import { useFixturesRatings } from '../hooks/useFixturesRatings';
import { Alert, AlertDescription } from '@/components/ui/alert';

const FixturesPage = () => {
  const { teamId } = useTeamId();
  const [selectedGW, setSelectedGW] = useState<number | null>(null);
  const fpl = useFplData(teamId, selectedGW ?? undefined);

  const gwInfo = fpl.gwInfo;
  const fixtures =
    fpl.fixtures.length > 0 ? fpl.fixtures : mockFixtures;

  const { minGW, maxGW } = useMemo(() => {
    const ids = fpl.bootstrap?.events?.map((e) => e.id) ?? [];
    if (!ids.length) {
      return { minGW: 1, maxGW: fpl.currentGW || 1 };
    }
    const min = Math.min(...ids);
    const max = fpl.currentGW || Math.max(...ids);
    return { minGW: min, maxGW: max };
  }, [fpl.bootstrap, fpl.currentGW]);

  const currentSelected = selectedGW ?? (fpl.currentGW || minGW);
  const disablePrev = currentSelected <= minGW;
  const disableNext = currentSelected >= maxGW;

  const handlePrev = () => {
    if (disablePrev) return;
    setSelectedGW(currentSelected - 1);
  };

  const handleNext = () => {
    if (disableNext) return;
    setSelectedGW(currentSelected + 1);
  };

  const ratings = useFixturesRatings(fpl.bootstrap, currentSelected);

  return (
    <div className="flex flex-1 flex-col gap-6 px-4 py-6 md:px-6 xl:px-10 xl:py-8">
      {fpl.loading ? (
        <div className="flex flex-col items-center justify-center gap-3 py-24">
          <Loader2 size={32} className="animate-spin text-emerald-400" />
          <p className="text-sm text-slate-400">
            Loading your FPL data…
          </p>
        </div>
      ) : (
        <>
          {fpl.error ? (
            <Alert className="rounded-2xl border border-[rgba(234,179,8,0.2)] bg-[rgba(234,179,8,0.08)]">
              <AlertDescription className="text-sm text-yellow-300">
                ⚠ Couldn&apos;t load live data — showing mock fixtures where needed. ({fpl.error})
              </AlertDescription>
            </Alert>
          ) : null}

          {ratings.error && !ratings.loading ? (
            <Alert className="rounded-2xl border border-[rgba(59,130,246,0.2)] bg-[rgba(59,130,246,0.08)]">
              <AlertDescription className="text-sm text-blue-200">
                {ratings.error}
              </AlertDescription>
            </Alert>
          ) : null}

          {gwInfo ? (
            <GWHeader
              info={gwInfo}
              onPrev={handlePrev}
              onNext={handleNext}
              disablePrev={disablePrev}
              disableNext={disableNext}
            />
          ) : null}

          <div className="grid grid-cols-1 items-start gap-6 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,0.8fr)]">
            <div>
              <FixturesCard
                fixtures={fixtures}
                isCurrentGw={currentSelected === fpl.currentGW}
              />
            </div>
            <div>
              <div className="flex flex-col gap-3">
                {ratings.loading ? (
                  <div className="flex items-center gap-2 text-slate-500">
                    <Loader2 size={18} className="animate-spin text-emerald-400" />
                    <span className="text-xs">Loading Elo / FDR from API…</span>
                  </div>
                ) : null}
                {ratings.data ? (
                  <TeamEloTable teams={ratings.data.teams} />
                ) : (
                  <div className="rounded-2xl border border-white/8 bg-slate-900 px-6 py-8">
                    <p className="text-sm text-slate-500">
                      Team ratings will appear here once bootstrap data is available.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {ratings.data ? (
            <div className="flex flex-col gap-6">
              <FdrMatrixTable
                title="Official FPL FDR"
                description="Integer difficulty (1–5) from fantasy.premierleague.com via your API /api/fdr/team (official_fpl_fdr)."
                gameweekIds={ratings.data.gameweekIds}
                teams={ratings.data.teams}
                mode="official"
              />
              <FdrMatrixTable
                title="Copilot Elo FDR"
                description="Custom overall FDR (1–5 scale) from ClubElo + injuries + squad changes — overall_fdr from /api/fdr/team."
                gameweekIds={ratings.data.gameweekIds}
                teams={ratings.data.teams}
                mode="elo"
              />
            </div>
          ) : null}
        </>
      )}
    </div>
  );
};

export default FixturesPage;
