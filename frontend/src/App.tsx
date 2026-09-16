import { Routes, Route, Navigate } from 'react-router-dom';
import ProtectedRoute from './components/ProtectedRoute';
import { CoreProvider } from './context/CoreContext';
import { SquadProvider } from './context/SquadContext';
import { AppShell } from './components/layout/AppShell';
import LoginPage from './pages/LoginPage';
import HomePage from './pages/HomePage';
import GameweekPage from './pages/GameweekPage';
import CommandCenterPage from './pages/CommandCenterPage';
import PlayersPage from './pages/PlayersPage';
import PlayerDetailPage from './pages/PlayerDetailPage';
import FixturesPage from './pages/FixturesPage';

function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route element={<ProtectedRoute />}>
        <Route
          element={
            <CoreProvider>
              <SquadProvider>
                <AppShell />
              </SquadProvider>
            </CoreProvider>
          }
        >
          <Route path="/" element={<Navigate to="/home" replace />} />
          <Route path="/home" element={<HomePage />} />
          <Route path="/gw" element={<GameweekPage />} />
          <Route path="/command" element={<CommandCenterPage />} />
          <Route path="/players" element={<PlayersPage />} />
          <Route path="/player/:playerId" element={<PlayerDetailPage />} />
          <Route path="/fixtures" element={<FixturesPage />} />
          <Route path="*" element={<Navigate to="/home" replace />} />
        </Route>
      </Route>
    </Routes>
  );
}

export default App;
