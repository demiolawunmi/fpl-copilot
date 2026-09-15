import { useRef, useState, useEffect, useMemo } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Loader2 } from 'lucide-react';
import GWHeader from '../components/gw-overview/GWHeader';
import StatsStrip from '../components/gw-overview/StatsStrip';
import PitchCard from '../components/gw-overview/PitchCard';
import FixturesCard from '../components/gw-overview/FixturesCard';
import InjuriesTable from '../components/gw-overview/InjuriesTable';
import TransfersTable from '../components/gw-overview/TransfersTable';
import RecommendedTransfersCard from '../components/gw-overview/RecommendedTransfersCard';
import AISummaryCard from '../components/gw-overview/AISummaryCard';
import {
  mockGWInfo,
  mockStats,
  mockSquad,
  mockFixtures,
  mockInjuries,
  mockTransfers,
  mockRecommendedTransfers,
  mockAISummary,
} from '../data/gwOverviewMocks';
import { useTeamId } from '../context/TeamIdContext';
import { useFplData } from '../hooks/useFplData';
import { Alert, AlertDescription } from '@/components/ui/alert';

const GWOverviewPage = () => {
  const { teamId } = useTeamId();
  const navigate = useNavigate();
  const location = useLocation();
  const [selectedGW, setSelectedGW] = useState<number | null>(null);
  const fpl = useFplData(teamId, selectedGW ?? undefined);

  const gwInfo = fpl.gwInfo ?? mockGWInfo;
  const stats = fpl.stats ?? mockStats;
  const squad = fpl.squad.length > 0 ? fpl.squad : mockSquad;
  const fixtures = fpl.fixtures.length > 0 ? fpl.fixtures : mockFixtures;

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

  const pitchRef = useRef<HTMLDivElement | null>(null);
  const fixturesTopRef = useRef<HTMLDivElement | null>(null);
  const aiSummaryRef = useRef<HTMLDivElement | null>(null);
  const [fixturesHeight, setFixturesHeight] = useState<number | undefined>(undefined);
  const [recommendedHeight, setRecommendedHeight] = useState<number | undefined>(undefined);

  useEffect(() => {
    const update = () => {
      if (!pitchRef.current || !fixturesTopRef.current) return;
      const pitchRect = pitchRef.current.getBoundingClientRect();
      const fixturesRect = fixturesTopRef.current.getBoundingClientRect();
      const height = Math.max(0, Math.round(pitchRect.bottom - fixturesRect.top));
      setFixturesHeight(height || undefined);
    };

    const raf = requestAnimationFrame(update);
    window.addEventListener('resize', update);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', update);
    };
  }, [gwInfo.gameweek, stats?.gwPoints, squad.length, fixtures.length]);

  useEffect(() => {
    const update = () => {
      if (!aiSummaryRef.current) return;
      const rect = aiSummaryRef.current.getBoundingClientRect();
      setRecommendedHeight(Math.round(rect.height) || undefined);
    };

    const raf = requestAnimationFrame(update);
    window.addEventListener('resize', update);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', update);
    };
  }, [gwInfo.gameweek, stats?.gwPoints, squad.length]);

  return (
    <div className="flex flex-1 flex-col gap-6 px-4 py-6 md:px-6 xl:px-10 xl:py-8">
      {fpl.loading ? (
        <div className="flex flex-col items-center justify-center gap-3 py-12">
          <Loader2 size={32} className="animate-spin text-emerald-400" />
          <p className="text-sm text-slate-400">
            Loading your FPL data…
          </p>
        </div>
      ) : null}

      {fpl.error && !fpl.loading ? (
        <Alert className="rounded-2xl border border-[rgba(234,179,8,0.2)] bg-[rgba(234,179,8,0.08)]">
          <AlertDescription className="text-sm text-yellow-300">
            ⚠ Couldn't load live data — showing mock data. ({fpl.error})
          </AlertDescription>
        </Alert>
      ) : null}

      <GWHeader
        info={gwInfo}
        onPrev={handlePrev}
        onNext={handleNext}
        disablePrev={disablePrev}
        disableNext={disableNext}
      />

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        <div className="col-span-1 xl:col-span-2">
          <div className="flex flex-col gap-6">
            <StatsStrip stats={stats} />
            <div ref={pitchRef}>
              <PitchCard
                squad={squad}
                onPlayerClick={(player) => {
                  // Guard: only navigate when a valid numeric id is present
                  const id = player?.id;
                  if (id == null || typeof id !== 'number' || Number.isNaN(id) || id <= 0) return;
                  navigate(`/players/${id}`, { state: { from: location.pathname } });
                }}
              />
            </div>
            <div ref={aiSummaryRef}>
              <AISummaryCard gwInfo={gwInfo} summary={mockAISummary} />
            </div>
          </div>
        </div>

        <div className="col-span-1">
          <div className="flex flex-col gap-6">
            <div ref={fixturesTopRef}>
              <FixturesCard
                fixtures={fixtures}
                heightPx={fixturesHeight}
                isCurrentGw={currentSelected === fpl.currentGW}
              />
            </div>
            <RecommendedTransfersCard
              transfers={mockRecommendedTransfers}
              heightPx={recommendedHeight}
            />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <InjuriesTable injuries={mockInjuries} />
        <TransfersTable transfers={mockTransfers} />
      </div>
    </div>
  );
};

export default GWOverviewPage;
