import { Loader2 } from 'lucide-react';
import { useTeamId } from '../context/TeamIdContext';
import { useFplData } from '../hooks/useFplData';
import { Badge } from '@/components/ui/badge';
import { DashboardCard } from '@/components/ui/primitives';

const HomePage = () => {
  const { teamId } = useTeamId();
  const fpl = useFplData(teamId);

  return (
    <div className="flex flex-1 items-center justify-center px-8 py-12">
      <div className="flex w-full max-w-2xl flex-col items-center gap-6 text-center">
        <h1 className="text-4xl font-bold leading-[1.33]">Welcome to FPL Copilot</h1>

        {fpl.loading ? (
          <div className="flex items-center gap-3 text-slate-400">
            <Loader2 size={24} className="animate-spin text-emerald-400" />
            <span className="text-sm">Loading team info…</span>
          </div>
        ) : null}

        {!fpl.loading && fpl.gwInfo ? (
          <DashboardCard className="w-full max-w-xl px-10 py-8">
            <div className="flex flex-col items-center gap-4">
              <p className="text-lg font-semibold text-white">
                {fpl.gwInfo.teamName}
              </p>
              <p className="text-sm text-slate-400">
                {fpl.gwInfo.manager}
              </p>
              <Badge className="rounded-full border border-[rgba(16,185,129,0.22)] bg-[rgba(16,185,129,0.12)] px-4 py-1.5 font-mono text-sm normal-case text-emerald-300">
                Team ID: {fpl.gwInfo.teamId}
              </Badge>
              <p className="text-xs text-slate-500">
                Current Gameweek: {fpl.gwInfo.gameweek}
              </p>
            </div>
          </DashboardCard>
        ) : null}

        {!fpl.loading && fpl.error ? (
          <div className="flex flex-col items-center gap-3">
            <p className="text-slate-400">Team ID:</p>
            <Badge className="rounded-2xl border border-[rgba(16,185,129,0.22)] bg-[rgba(16,185,129,0.12)] px-6 py-3 font-mono text-2xl normal-case text-emerald-300">
              {teamId}
            </Badge>
            <p className="text-xs text-yellow-300">
              ⚠ Couldn't reach FPL API — {fpl.error}
            </p>
          </div>
        ) : null}
      </div>
    </div>
  );
};

export default HomePage;
