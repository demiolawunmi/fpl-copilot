import { Routes, Route } from 'react-router-dom';
import Navbar from './components/NavBar';
import ProtectedRoute from './components/ProtectedRoute';
import LoginPage from './pages/LoginPage';
import HomePage from './pages/HomePage';
import GWOverviewPage from './pages/GWOverviewPage';
import PlayersPage from './pages/PlayersPage';
import PlayerDetailPage from './pages/PlayerDetailPage';
import FixturesPage from './pages/FixturesPage';
import CommandCenterPage from './pages/CommandCenterPage';
import { Toaster } from '@/components/ui/sonner';
import { TooltipProvider } from '@/components/ui/tooltip';
import { useTeamId } from './context/TeamIdContext';
import { getEntry } from './api/fpl/fpl';
import { useEffect, useState } from 'react';

function App() {
  const { teamId } = useTeamId();
  const [teamName, setTeamName] = useState<string | null>(null);
  const visibleTeamName = teamId ? teamName : null;

  useEffect(() => {
    if (!teamId) {
      return;
    }

    void getEntry(teamId).then((entry) => setTeamName(entry.name));
  }, [teamId]);

  return (
    <TooltipProvider>
      <div className="flex min-h-screen flex-col bg-slate-950">
        {teamId && <Navbar teamName={visibleTeamName} />}

        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route element={<ProtectedRoute />}>
            <Route path="/" element={<HomePage />} />
            <Route path="/gw-overview" element={<GWOverviewPage />} />
            <Route path="/command-center" element={<CommandCenterPage />} />
            <Route path="/players" element={<PlayersPage />} />
            <Route path="/players/:playerId" element={<PlayerDetailPage />} />
            <Route path="/fixtures" element={<FixturesPage />} />
          </Route>
        </Routes>
        <Toaster />
      </div>
    </TooltipProvider>
  );
}

export default App;
