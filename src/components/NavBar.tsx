import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useTeamId } from '../context/TeamIdContext';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';

interface NavbarProps {
  teamName?: string | null;
}

const navLinks = [
  { to: '/', label: 'Home' },
  { to: '/gw-overview', label: 'GW Overview' },
  { to: '/command-center', label: 'Command Center' },
  { to: '/players', label: 'Players' },
  { to: '/fixtures', label: 'Fixtures' },
];

const Navbar = ({ teamName }: NavbarProps) => {
  const location = useLocation();
  const navigate = useNavigate();
  const { teamId, clearTeamId } = useTeamId();

  const handleSignOut = () => {
    clearTeamId();
    navigate('/login', { replace: true });
  };

  return (
    <nav className="border-b border-white/8 bg-[rgba(15,23,42,0.92)] shadow-lg">
      <div className="mx-auto w-full max-w-[90rem] px-4 py-4 md:px-6 xl:px-10">
        <div className="flex flex-wrap items-center justify-between gap-6">
          <div className="flex flex-wrap items-center gap-4 md:gap-8">
            <Link
              to="/"
              className="text-xl font-bold tracking-wide text-emerald-400"
            >
              FPL Copilot
            </Link>

            <div className="flex flex-wrap items-center gap-2">
              {navLinks.map(({ to, label }) => {
                const isPlayersLink = to === '/players';
                const isActive = isPlayersLink
                  ? location.pathname === '/players' || location.pathname.startsWith('/players/')
                  : location.pathname === to;
                return (
                  <Button
                    key={to}
                    asChild
                    size="sm"
                    variant="ghost"
                    className={`hover:bg-white/6 hover:text-white ${isActive ? 'bg-white/8 text-white' : 'text-slate-300'}`}
                  >
                    <Link to={to}>{label}</Link>
                  </Button>
                );
              })}
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-start gap-4 md:justify-end">
            {teamId ? (
              <Badge className="rounded-full border border-[rgba(16,185,129,0.22)] bg-[rgba(16,185,129,0.12)] px-3 py-1.5 font-mono text-xs normal-case text-emerald-300">
                ID: {teamId} {teamName ? `| ${teamName}` : ''}
              </Badge>
            ) : null}
            <Button
              size="sm"
              variant="ghost"
              className="text-red-300 hover:bg-[rgba(248,113,113,0.12)] hover:text-red-200"
              onClick={handleSignOut}
            >
              Sign Out
            </Button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
