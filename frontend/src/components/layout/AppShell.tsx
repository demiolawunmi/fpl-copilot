import { useEffect, useState, type ReactNode } from 'react';
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom';
import { Icon, LogoMark, type IconName } from '../Icon';
import { Countdown } from '../ds/Countdown';
import { useCore } from '../../context/CoreContext';
import { useTheme } from '../../context/ThemeContext';
import { useTeamId } from '../../context/TeamIdContext';
import { useToast } from '../../context/ToastContext';
import { refreshMyTeam } from '../../api/backend';
import { initials, isDeadlineUrgent } from '../../lib/format';
import { animateCounts } from '../../lib/motion';

interface NavItem {
  to: string;
  label: string;
  icon: IconName;
  group: 'Review' | 'Plan' | 'Explore';
}

const NAV: NavItem[] = [
  { to: '/home', label: 'Home', icon: 'home', group: 'Review' },
  { to: '/gw', label: 'Gameweek', icon: 'calendar', group: 'Review' },
  { to: '/command', label: 'Command Center', icon: 'target', group: 'Plan' },
  { to: '/players', label: 'Players', icon: 'users', group: 'Explore' },
  { to: '/fixtures', label: 'Fixtures', icon: 'shield', group: 'Explore' },
];

const TITLES: Record<string, [string, string]> = {
  home: ['Review', 'Dashboard'],
  gw: ['Review', 'Gameweek overview'],
  command: ['Plan', 'Command Center'],
  players: ['Explore', 'Players'],
  player: ['Explore', 'Player'],
  fixtures: ['Explore', 'Fixtures'],
};

function routeKey(pathname: string): string {
  const seg = pathname.split('/').filter(Boolean)[0] ?? 'home';
  return seg in TITLES ? seg : 'home';
}

function Rail() {
  const core = useCore();
  const { clearTeamId } = useTeamId();
  const navigate = useNavigate();
  const toast = useToast();
  const [refreshing, setRefreshing] = useState(false);
  const groups: NavItem['group'][] = ['Review', 'Plan', 'Explore'];
  const manager = core.manager;

  const refresh = async () => {
    if (refreshing) return;
    setRefreshing(true);
    try {
      await refreshMyTeam();
      toast('Team refreshed from the official FPL API.', 'pos');
    } catch {
      toast('Couldn’t re-pull your team from FPL — showing the latest cached data.', 'warn');
    } finally {
      core.refresh();
      window.setTimeout(() => setRefreshing(false), 900);
    }
  };

  return (
    <aside className="rail">
      <NavLink to="/home" className="brand" aria-label="FPL Copilot home" style={{ textDecoration: 'none', color: 'inherit' }}>
        <LogoMark size={40} />
        <span className="word">
          FPL<span>Copilot</span>
        </span>
      </NavLink>
      <nav className="od-stack" style={{ gap: 4 }}>
        {groups.map((g) => (
          <div key={g}>
            <div className="rail-label">{g}</div>
            <div className="rail-group">
              {NAV.filter((n) => n.group === g).map((n) => (
                <NavLink key={n.to} to={n.to} className="navlink">
                  <Icon name={n.icon} />
                  <span>{n.label}</span>
                </NavLink>
              ))}
            </div>
          </div>
        ))}
      </nav>
      <div className="rail-foot">
        <div className="mgr-card">
          <div className="od-row" style={{ gap: 10 }}>
            <span className="avatar sm">{initials(manager?.name ?? 'FC')}</span>
            <div style={{ minWidth: 0 }}>
              <div className="small" style={{ fontWeight: 700, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {manager?.name ?? 'Loading…'}
              </div>
              <div className="tid">ID {manager?.teamId ?? '—'}</div>
            </div>
            <span className="spacer" />
            <button
              className="btn btn-icon btn-ghost btn-sm"
              type="button"
              title="Refresh team data from the official FPL API"
              aria-label="Refresh team data"
              onClick={() => void refresh()}
              disabled={refreshing}
            >
              <span className="spin-host" style={{ display: 'inline-flex', animation: refreshing ? 'spin .8s linear infinite' : undefined }}>
                <Icon name="refresh" size={15} />
              </span>
            </button>
          </div>
          <div className="divider" style={{ margin: '10px 0' }} />
          <div className="tiny muted">{manager?.manager ?? '—'}</div>
        </div>
        <button
          className="navlink"
          type="button"
          onClick={() => {
            clearTeamId();
            navigate('/login');
          }}
        >
          <Icon name="logout" />
          <span>Sign out</span>
        </button>
      </div>
    </aside>
  );
}

function Topbar({ onOpenPalette }: { onOpenPalette: () => void }) {
  const core = useCore();
  const { theme, toggle } = useTheme();
  const location = useLocation();
  const key = routeKey(location.pathname);
  const [eyebrow, title] = TITLES[key];
  const deadline = core.nextDeadline;
  const showDeadline = key === 'command' || key === 'home' || key === 'gw';
  const urgent = isDeadlineUrgent(deadline);

  return (
    <header className="topbar">
      <div className="crumb">
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
      </div>
      <span className="spacer" />
      {showDeadline && deadline ? (
        <div className={`deadline-chip ${urgent ? 'urgent' : ''}`}>
          <span className="dot" />
          <span className="small muted">GW{core.nextGW}</span>
          <span className="mono small" data-act="clock-target" data-ms={new Date(deadline).getTime()}>
            <Countdown target={deadline} />
          </span>
        </div>
      ) : null}
      <button className="btn btn-ghost btn-sm" type="button" onClick={onOpenPalette} aria-label="Open command palette">
        <Icon name="search" size={15} /> <span className="mono tiny">⌘K</span>
      </button>
      <button className="btn btn-icon btn-ghost" type="button" onClick={toggle} aria-label="Toggle colour theme">
        <Icon name={theme === 'dark' ? 'sun' : 'moon'} />
      </button>
    </header>
  );
}

function MobileNav() {
  return (
    <nav className="mobile-nav" aria-label="Primary">
      {NAV.map((n) => (
        <NavLink key={n.to} to={n.to} className="navlink">
          <Icon name={n.icon} size={20} />
          <span>{n.label.replace(' Center', '')}</span>
        </NavLink>
      ))}
    </nav>
  );
}

function CommandPalette({ onClose }: { onClose: () => void }) {
  const core = useCore();
  const navigate = useNavigate();
  const [q, setQ] = useState('');

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const query = q.trim().toLowerCase();
  const navMatches = NAV.filter((n) => !query || n.label.toLowerCase().includes(query));
  const playerMatches = (query ? core.players.filter((p) => p.name.toLowerCase().includes(query)) : [])
    .slice(0, 7);

  const go = (to: string) => {
    onClose();
    navigate(to);
  };

  return (
    <div className="scrim" onClick={onClose}>
      <div
        className="dialog"
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
        style={{ alignSelf: 'start', marginTop: '12vh' }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="dialog-head">
          <Icon name="search" size={18} />
          <input
            className="input"
            style={{ border: 0, background: 'transparent' }}
            placeholder="Search pages and players…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            aria-label="Search"
            autoFocus
          />
        </div>
        <div className="dialog-body od-stack" style={{ gap: 2, maxHeight: '50vh', overflow: 'auto' }}>
          {navMatches.map((n) => (
            <button key={n.to} className="od-row" style={paletteRowStyle} type="button" onClick={() => go(n.to)}>
              <span className="muted">
                <Icon name={n.icon} size={16} />
              </span>
              <span className="od-fill">
                <span className="small" style={{ fontWeight: 600, display: 'block' }}>
                  {n.label}
                </span>
                <span className="tiny faint">{n.group}</span>
              </span>
              <span className="badge">Navigate</span>
            </button>
          ))}
          {playerMatches.map((p) => (
            <button
              key={p.id}
              className="od-row"
              style={paletteRowStyle}
              type="button"
              onClick={() => go(`/player/${p.id}?from=players`)}
            >
              <span className="muted">
                <Icon name="users" size={16} />
              </span>
              <span className="od-fill">
                <span className="small" style={{ fontWeight: 600, display: 'block' }}>
                  {p.name}
                </span>
                <span className="tiny faint">
                  {p.club.short} · {p.pos} · £{p.price.toFixed(1)}m
                </span>
              </span>
              <span className="badge">Player</span>
            </button>
          ))}
          {!navMatches.length && !playerMatches.length ? (
            <div className="empty">
              <Icon name="search" size={32} />
              <span className="small">No matches</span>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

const paletteRowStyle: React.CSSProperties = {
  gap: 12,
  width: '100%',
  textAlign: 'left',
  padding: '10px 12px',
  border: 0,
  borderRadius: 'var(--r-sm)',
  background: 'transparent',
  cursor: 'pointer',
};

export function AppShell() {
  const [paletteOpen, setPaletteOpen] = useState(false);
  const core = useCore();
  const location = useLocation();

  useEffect(() => {
    const view = document.getElementById('view') ?? document;
    const id = window.setTimeout(() => animateCounts(view), 30);
    return () => window.clearTimeout(id);
  }, [location.pathname, core.loading, core.players]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setPaletteOpen(true);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  return (
    <>
      <div className="shell">
        <Rail />
        <div style={{ minWidth: 0 }}>
          <Topbar onOpenPalette={() => setPaletteOpen(true)} />
          <main className="content" id="view">
            <Outlet />
          </main>
        </div>
      </div>
      <MobileNav />
      {paletteOpen ? <CommandPalette onClose={() => setPaletteOpen(false)} /> : null}
    </>
  );
}

export function PageWrap({ children, gap = 24 }: { children: ReactNode; gap?: number }) {
  return (
    <div className="od-stack page-enter" style={{ gap }}>
      {children}
    </div>
  );
}
