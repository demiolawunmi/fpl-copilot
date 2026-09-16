import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Icon } from '../components/Icon';
import { Card } from '../components/ds/Card';
import { Avatar, Badge, Chip, Crest, Empty, PosBadge, Segmented } from '../components/ds/atoms';
import { FixturePills } from '../components/shared';
import { useCore } from '../context/CoreContext';
import { money, num, surname } from '../lib/format';
import type { Player } from '../domain/types';

type ColumnKey =
  | 'price' | 'own' | 'mins' | 'pts' | 'xpts' | 'g' | 'a'
  | 'xg' | 'xa' | 'xgi' | 'pts90' | 'xgi90' | 'form';

interface Column {
  key: ColumnKey;
  label: string;
  on: boolean;
}

const STORAGE_KEY = 'fpl:columns';

const DEFAULT_COLUMNS: Column[] = [
  { key: 'price', label: '£', on: true },
  { key: 'own', label: 'Owned', on: true },
  { key: 'mins', label: 'Mins', on: false },
  { key: 'pts', label: 'Pts', on: true },
  { key: 'xpts', label: 'xPts', on: true },
  { key: 'g', label: 'G', on: true },
  { key: 'a', label: 'A', on: true },
  { key: 'xg', label: 'xG', on: false },
  { key: 'xa', label: 'xA', on: false },
  { key: 'xgi', label: 'xGI', on: true },
  { key: 'pts90', label: 'Pts/90', on: false },
  { key: 'xgi90', label: 'xGI/90', on: false },
  { key: 'form', label: 'Form', on: true },
];

const PRESETS: Record<string, string> = {
  none: 'All players',
  fwdxg90: 'Forwards by xG/90',
  midcreate: 'Midfield creativity',
  budgetdiff: 'Budget differentials',
};

function loadColumns(): Column[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_COLUMNS;
    const saved = JSON.parse(raw) as Column[];
    return DEFAULT_COLUMNS.map((c) => {
      const hit = saved.find((s) => s.key === c.key);
      return hit ? { ...c, on: hit.on } : c;
    });
  } catch {
    return DEFAULT_COLUMNS;
  }
}

export default function PlayersPage() {
  const core = useCore();
  const navigate = useNavigate();
  const [columns, setColumns] = useState<Column[]>(loadColumns);
  const [columnsOpen, setColumnsOpen] = useState(false);
  const [q, setQ] = useState('');
  const [team, setTeam] = useState('all');
  const [pos, setPos] = useState<'all' | 'GK' | 'DEF' | 'MID' | 'FWD'>('all');
  const [preset, setPreset] = useState('none');
  const [sort, setSort] = useState<{ key: ColumnKey; dir: 'asc' | 'desc' }>({ key: 'pts', dir: 'desc' });
  const [page, setPage] = useState(1);

  const perPage = 25;

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(columns));
    } catch {
      /* storage unavailable */
    }
  }, [columns]);

  let list = core.players.slice();
  if (preset === 'budgetdiff') list = list.filter((p) => p.price <= 6 && p.own < 5);
  if (team !== 'all') list = list.filter((p) => p.club.short === team);
  if (pos !== 'all') list = list.filter((p) => p.pos === pos);
  const query = q.trim().toLowerCase();
  if (query) list = list.filter((p) => p.name.toLowerCase().includes(query) || p.club.name.toLowerCase().includes(query));

  const dir = sort.dir === 'asc' ? 1 : -1;
  list.sort((a, b) => ((a[sort.key] as number) - (b[sort.key] as number)) * dir);

  const totalPages = Math.max(1, Math.ceil(list.length / perPage));
  const safePage = Math.min(page, totalPages);
  const pageRows = list.slice((safePage - 1) * perPage, safePage * perPage);
  const activeCols = columns.filter((c) => c.on);

  const applyPreset = (key: string) => {
    setPreset(key);
    setPage(1);
    setQ('');
    if (key === 'fwdxg90') {
      setPos('FWD');
      setSort({ key: 'xg90' as ColumnKey, dir: 'desc' });
    } else if (key === 'midcreate') {
      setPos('MID');
      setSort({ key: 'form', dir: 'desc' });
    } else if (key === 'budgetdiff') {
      setPos('all');
      setTeam('all');
      setSort({ key: 'xpts', dir: 'desc' });
    } else {
      setPos('all');
      setTeam('all');
      setSort({ key: 'pts', dir: 'desc' });
    }
  };

  const toggleSort = (key: ColumnKey) => {
    setSort((s) => (s.key === key ? { key, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { key, dir: 'desc' }));
  };

  const leaders: { k: string; key: keyof Player; fmt: (v: number) => string }[] = [
    { k: 'Goals', key: 'g', fmt: (v) => String(v) },
    { k: 'Assists', key: 'a', fmt: (v) => String(v) },
    { k: 'xG', key: 'xg', fmt: (v) => v.toFixed(1) },
    { k: 'xGI', key: 'xgi', fmt: (v) => v.toFixed(1) },
    { k: 'Clean sheets', key: 'cs', fmt: (v) => String(v) },
  ];

  return (
    <div className="od-stack page-enter" style={{ gap: 24 }}>
      <div className="od-row" style={{ gap: 16, flexWrap: 'wrap' }}>
        <div>
          <h2>Players</h2>
          <p className="muted small">Season stats, expected points and per-90 rates for every player in the game.</p>
        </div>
        <span className="spacer" />
        <button className="btn btn-ghost" type="button" onClick={() => setColumnsOpen(true)}>
          <Icon name="columns" size={16} /> Columns <Badge>{activeCols.length}/{columns.length}</Badge>
        </button>
      </div>

      <div className="od-stack" style={{ gap: 12 }}>
        <div className="section-title">
          <h3>Leaderboards</h3>
          <span className="sub">Top 3 by metric</span>
        </div>
        <div className="od-rail" style={{ ['--od-rail-pad' as string]: '0px', paddingBottom: 4 }}>
          {leaders.map((set) => {
            const top = core.players
              .slice()
              .sort((a, b) => (b[set.key] as number) - (a[set.key] as number))
              .slice(0, 3);
            return (
              <div className="card" key={set.k} style={{ width: 220, flex: 'none' }}>
                <div className="card-body" style={{ padding: 'var(--sp-4)' }}>
                  <div className="od-row" style={{ justifyContent: 'space-between' }}>
                    <span className="tiny faint" style={{ textTransform: 'uppercase', letterSpacing: '.12em' }}>
                      {set.k}
                    </span>
                  </div>
                  <div className="od-stack" style={{ gap: 10, marginTop: 12 }}>
                    {top.map((p, i) => (
                      <button
                        key={p.id}
                        className="od-row"
                        type="button"
                        style={{ gap: 10, background: 'transparent', border: 0, textAlign: 'left', cursor: 'pointer', width: '100%' }}
                        onClick={() => navigate(`/player/${p.id}?from=players`)}
                      >
                        <span className={`mono ${i === 0 ? 'pos' : 'faint'}`} style={{ width: 14 }}>
                          {i + 1}
                        </span>
                        <Avatar player={p} size={30} />
                        <Crest club={p.club} size={20} />
                        <span className="od-fill">
                          <span className="small" style={{ display: 'block', fontWeight: 600 }}>
                            {surname(p.name)}
                          </span>
                          <span className="tiny faint">{p.club.short}</span>
                        </span>
                        <span className="mono" style={{ fontWeight: 700 }}>
                          {set.fmt(p[set.key] as number)}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <Card
        reveal
        title="Explorer"
        actions={<Badge>{list.length} players</Badge>}
        bodyClassName="tight"
      >
        <div className="od-stack" style={{ gap: 12, padding: 'var(--sp-3)' }}>
          <div className="od-row" style={{ gap: 10, flexWrap: 'wrap' }}>
            <label className="od-row od-fill" style={{ gap: 8, minWidth: 200, position: 'relative' }}>
              <span style={{ position: 'absolute', left: 12, color: 'var(--faint)' }}>
                <Icon name="search" size={16} />
              </span>
              <input
                className="input"
                style={{ paddingLeft: 38 }}
                placeholder="Search by name or club…"
                value={q}
                onChange={(e) => {
                  setQ(e.target.value);
                  setPage(1);
                }}
                aria-label="Search players"
              />
            </label>
            <select
              className="select"
              aria-label="Filter by team"
              style={{ width: 'auto' }}
              value={team}
              onChange={(e) => {
                setTeam(e.target.value);
                setPage(1);
              }}
            >
              <option value="all">All teams</option>
              {core.clubs.map((c) => (
                <option key={c.id} value={c.short}>
                  {c.name}
                </option>
              ))}
            </select>
            <Segmented
              ariaLabel="Filter by position"
              value={pos}
              onChange={(v) => {
                setPos(v);
                setPage(1);
              }}
              options={[
                { value: 'all', label: 'All' },
                { value: 'GK', label: 'GK' },
                { value: 'DEF', label: 'DEF' },
                { value: 'MID', label: 'MID' },
                { value: 'FWD', label: 'FWD' },
              ]}
            />
          </div>
          <div className="chips-row">
            <span className="tiny faint" style={{ alignSelf: 'center' }}>
              Presets
            </span>
            {Object.keys(PRESETS).map((k) => (
              <Chip key={k} pressed={preset === k} onClick={() => applyPreset(k)}>
                {PRESETS[k]}
              </Chip>
            ))}
          </div>
        </div>

        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>Player</th>
                {activeCols.map((c) => (
                  <th
                    key={c.key}
                    className="sortable"
                    aria-sort={sort.key === c.key ? (sort.dir === 'asc' ? 'ascending' : 'descending') : undefined}
                    onClick={() => toggleSort(c.key)}
                  >
                    {c.label}
                    {sort.key === c.key ? (sort.dir === 'asc' ? ' ▲' : ' ▼') : ''}
                  </th>
                ))}
                <th>Next</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {pageRows.length ? (
                pageRows.map((p) => (
                  <tr key={p.id} className="clickable" onClick={() => navigate(`/player/${p.id}?from=players`)}>
                    <td>
                      <div className="cell-player">
                        <Avatar player={p} size={30} />
                        <Crest club={p.club} size={20} />
                        <PosBadge pos={p.pos} />
                        <div>
                          <div className="nm">{p.name}</div>
                          <div className="meta">{p.club.name}</div>
                        </div>
                      </div>
                    </td>
                    {activeCols.map((c) => (
                      <td key={c.key} className={`mono ${c.key === 'form' && p.form >= 6 ? 'pos' : ''}`}>
                        {formatCell(p, c.key)}
                      </td>
                    ))}
                    <td>
                      <FixturePills teamId={p.teamId} fromGw={core.nextGW} count={3} />
                    </td>
                    <td>
                      <Link className="btn btn-ghost btn-sm" to={`/player/${p.id}?from=players`}>
                        View
                      </Link>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={activeCols.length + 3}>
                    <Empty title="No players match those filters" text="Clear the search or pick another preset." />
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="od-row" style={{ gap: 12, padding: 'var(--sp-4)' }}>
          <span className="small muted">
            Showing {list.length ? (safePage - 1) * perPage + 1 : 0}–{Math.min(list.length, safePage * perPage)} of {list.length}
          </span>
          <span className="spacer" />
          <button className="btn btn-ghost btn-sm" type="button" disabled={safePage <= 1} onClick={() => setPage((p) => Math.max(1, p - 1))}>
            <Icon name="chevronLeft" size={14} /> Prev
          </button>
          <span className="mono small">
            {safePage} / {totalPages}
          </span>
          <button className="btn btn-ghost btn-sm" type="button" disabled={safePage >= totalPages} onClick={() => setPage((p) => Math.min(totalPages, p + 1))}>
            Next <Icon name="chevronRight" size={14} />
          </button>
        </div>
      </Card>

      {columnsOpen ? (
        <div className="scrim" onClick={() => setColumnsOpen(false)}>
          <div className="dialog" role="dialog" aria-modal="true" aria-label="Customise columns" onClick={(e) => e.stopPropagation()}>
            <div className="dialog-head">
              <h3>Customise columns</h3>
              <span className="spacer" />
              <button className="btn btn-icon btn-ghost" type="button" onClick={() => setColumnsOpen(false)} aria-label="Close">
                <Icon name="x" />
              </button>
            </div>
            <div className="dialog-body od-stack" style={{ gap: 4 }}>
              <p className="small muted" style={{ marginBottom: 8 }}>
                {activeCols.length} of {columns.length} columns visible. Preferences are saved to this browser.
              </p>
              {columns.map((c) => (
                <label
                  key={c.key}
                  className="od-row"
                  style={{ gap: 12, padding: '10px 4px', borderBottom: '1px solid var(--border)', cursor: 'pointer' }}
                >
                  <span className="od-fill small" style={{ fontWeight: 600 }}>
                    {c.label}
                  </span>
                  <span
                    className="switch"
                    role="switch"
                    tabIndex={0}
                    aria-checked={c.on}
                    aria-label={`${c.label} column`}
                    onClick={() => setColumns((cols) => cols.map((x) => (x.key === c.key ? { ...x, on: !x.on } : x)))}
                    onKeyDown={(e) => {
                      if (e.key === ' ' || e.key === 'Enter') {
                        e.preventDefault();
                        setColumns((cols) => cols.map((x) => (x.key === c.key ? { ...x, on: !x.on } : x)));
                      }
                    }}
                  />
                </label>
              ))}
            </div>
            <div className="dialog-foot">
              <button
                className="btn btn-ghost"
                type="button"
                onClick={() => setColumns(DEFAULT_COLUMNS)}
              >
                Reset to default
              </button>
              <button className="btn btn-primary" type="button" onClick={() => setColumnsOpen(false)}>
                Done
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function formatCell(p: Player, key: ColumnKey): string {
  switch (key) {
    case 'price':
      return money(p.price);
    case 'own':
      return `${num(p.own, 1)}%`;
    case 'xpts':
      return p.xpts.toFixed(1);
    case 'xg':
    case 'xa':
    case 'xgi':
    case 'xgi90':
    case 'pts90':
      return (p[key] as number).toFixed(1);
    default:
      return String(p[key]);
  }
}
