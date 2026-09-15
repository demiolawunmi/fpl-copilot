import { useState, useMemo } from 'react';
import { DashboardCard, DashboardHeader } from '@/components/ui/primitives';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { getDifficultyColor } from '../../utils/difficulty';
import { getPlayerPhotoUrl } from '../../api/fpl/fpl';
import type { EnhancedPlayer } from '../../data/commandCenterMocks';
import type { PredictionPlayer, PlayerFixture } from '../../api/backend';

type Position = 'GK' | 'DEF' | 'MID' | 'FWD';

interface BootstrapElement {
  id: number;
  code: number;
  web_name: string;
  element_type: number;
  team: number;
  now_cost?: number;
  total_points?: number;
  form?: string;
  selected_by_percent?: string;
}

interface BootstrapTeam {
  id: number;
  short_name: string;
  name: string;
}

export interface CustomTransferBuilderProps {
  squad: EnhancedPlayer[];
  bootstrapElements: BootstrapElement[];
  bootstrapTeams: BootstrapTeam[];
  lookupPrediction: (name: string, teamAbbr?: string) => PredictionPlayer | undefined;
  fixturesByName: Map<string, PlayerFixture>;
  onTransfer: (playerInId: number, playerOutId: number) => void;
}

// ── helpers ──

const ELEM_POS: Record<number, Position> = { 1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD' };
const ALL_POSITIONS: Position[] = ['GK', 'DEF', 'MID', 'FWD'];
const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
const mkPhoto = (code: number) => getPlayerPhotoUrl(code);
const getInitials = (name: string) =>
  name
    .split(/\s+/)
    .map((w) => w[0])
    .join('')
    .slice(0, 2)
    .toUpperCase();

function avgDiff5(fixtures: PlayerFixture['fixtures'] | undefined): number | null {
  if (!fixtures || fixtures.length === 0) return null;
  const n = Math.min(fixtures.length, 5);
  return fixtures.slice(0, n).reduce((s, f) => s + (f.difficulty ?? 3), 0) / n;
}

function nextFixtureLabel(fixtures: PlayerFixture['fixtures'] | undefined): string | null {
  if (!fixtures || fixtures.length === 0) return null;
  const f = fixtures[0];
  return `${f.is_home ? 'H' : 'A'} ${f.opponent_short}`;
}

interface RowData {
  id: number;
  name: string;
  position: Position;
  teamAbbr: string;
  xPts: number;
  avgDiff: number | null;
  photoUrl: string | undefined;
  price: number;
  form: number;
  ownership: number;
  nextFixture: string | null;
  isBench?: boolean;
}

type SortKey = 'xPts' | 'price' | 'avgDiff' | 'form' | 'ownership';
type SortDir = 'asc' | 'desc';

const SORT_OPTIONS: { key: SortKey; label: string }[] = [
  { key: 'xPts', label: 'xP' },
  { key: 'price', label: 'Price' },
  { key: 'avgDiff', label: 'FDR' },
  { key: 'form', label: 'Form' },
  { key: 'ownership', label: 'Sel%' },
];

function sortRows(rows: RowData[], key: SortKey, dir: SortDir): RowData[] {
  const sorted = [...rows];
  sorted.sort((a, b) => {
    const va = a[key] ?? (dir === 'asc' ? Infinity : -Infinity);
    const vb = b[key] ?? (dir === 'asc' ? Infinity : -Infinity);
    return dir === 'desc' ? (vb as number) - (va as number) : (va as number) - (vb as number);
  });
  return sorted;
}

// ── sub-components ──

function DiffBadge({ value }: { value: number | null }) {
  if (value == null) {
    return <span className="min-w-[30px] text-center text-[10px] text-slate-600">–</span>;
  }
  const colorName = getDifficultyColor(Math.round(value));
  const styles: Record<string, string> = {
    emerald: 'bg-[rgba(16,185,129,0.2)] text-green-300',
    yellow: 'bg-[rgba(250,204,21,0.2)] text-yellow-300',
    rose: 'bg-[rgba(244,63,94,0.2)] text-red-300',
  };
  const s = styles[colorName] ?? styles.yellow;
  return (
    <span
      className={`inline-flex min-w-[30px] items-center justify-center rounded-full px-2 py-0.5 text-[10px] font-bold ${s}`}
    >
      {value.toFixed(1)}
    </span>
  );
}

const POS_COLORS: Record<Position, string> = {
  GK: 'bg-yellow-500/20 text-yellow-300',
  DEF: 'bg-green-500/20 text-green-300',
  MID: 'bg-blue-500/20 text-blue-300',
  FWD: 'bg-red-500/20 text-red-300',
};

function PlayerRow({
  player,
  actionLabel,
  actionColor = 'blue',
  onAction,
  onRowClick,
}: {
  player: RowData;
  actionLabel?: string;
  actionColor?: string;
  onAction?: () => void;
  onRowClick?: () => void;
}) {
  const formColor =
    player.form <= 0 ? 'text-slate-600'
      : player.form >= 6 ? 'text-green-300'
      : player.form >= 3 ? 'text-orange-300'
      : 'text-red-300';
  return (
    <div
      className={`flex items-center gap-2 rounded-md px-3 py-2 transition-colors duration-150 ${onRowClick ? 'cursor-pointer hover:bg-white/4' : ''}`}
      onClick={onRowClick}
    >
      <Avatar size="sm">
        {player.photoUrl ? <AvatarImage src={player.photoUrl} alt={player.name} /> : null}
        <AvatarFallback className="bg-slate-700 text-slate-300">{getInitials(player.name)}</AvatarFallback>
      </Avatar>

      {/* Name + meta subtitle */}
      <div className="min-w-0 flex-1">
        <div className="mb-0.5 flex items-center gap-1.5">
          <span className="truncate text-sm font-semibold text-white">
            {player.name}
          </span>
          <Badge className={`rounded-xs px-1 text-[9px] ${POS_COLORS[player.position]}`}>
            {player.position}
          </Badge>
          {player.isBench && (
            <Badge variant="outline" className="rounded-xs border-gray-500/40 px-1 text-[9px] text-gray-300">
              BENCH
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-1">
          <span className="text-xs text-slate-400">{player.teamAbbr}</span>
          {player.price > 0 && (
            <>
              <span className="text-[9px] text-slate-700">·</span>
              <span className="text-xs text-slate-400">£{player.price.toFixed(1)}m</span>
            </>
          )}
          {player.nextFixture && (
            <>
              <span className="text-[9px] text-slate-700">·</span>
              <span className="text-xs text-slate-500">{player.nextFixture}</span>
            </>
          )}
        </div>
      </div>

      {/* Form — color-coded: green ≥ 6, orange 3–5.9, red < 3 */}
      <span className={`min-w-[28px] shrink-0 text-right text-xs font-bold ${formColor}`}>
        {player.form > 0 ? player.form.toFixed(1) : '–'}
      </span>

      {/* xP */}
      <span className="min-w-[32px] shrink-0 text-right text-sm font-bold text-blue-300">
        {player.xPts > 0 ? player.xPts.toFixed(1) : '–'}
      </span>

      {/* Sel% */}
      <span className="min-w-[34px] shrink-0 text-right text-xs font-medium text-slate-400">
        {player.ownership > 0 ? `${player.ownership.toFixed(1)}%` : '–'}
      </span>

      {/* FDR */}
      <DiffBadge value={player.avgDiff} />

      {/* Action */}
      {actionLabel && onAction ? (
        <Button
          variant={actionLabel === 'In' ? 'solid' : 'outline'}
          onClick={(e) => {
            e.stopPropagation();
            onAction();
          }}
          className={`h-6 min-w-[40px] rounded-md px-2.5 text-[10px] ${
            actionColor === 'green'
              ? actionLabel === 'In'
                ? 'bg-green-500 text-white hover:bg-green-600'
                : 'border-green-500/40 text-green-300 hover:bg-green-500/10'
              : actionLabel === 'In'
                ? 'bg-blue-500 text-white hover:bg-blue-600'
                : 'border-red-500/40 text-red-300 hover:bg-red-500/10'
          }`}
        >
          {actionLabel}
        </Button>
      ) : (
        <div className="w-[40px]" />
      )}
    </div>
  );
}

// ── sort header chip ──

function SortChip({
  label,
  active,
  dir,
  onClick,
}: {
  label: string;
  active: boolean;
  dir: SortDir;
  onClick: () => void;
}) {
  return (
    <Button
      aria-label={`Sort by ${label}`}
      variant={active ? 'solid' : 'ghost'}
      onClick={onClick}
      className={`h-[22px] min-w-0 rounded-md px-2 text-[10px] ${
        active ? 'bg-blue-500 font-bold text-white hover:bg-blue-600' : 'font-medium text-slate-500'
      }`}
    >
      <span className="text-[10px]">
        {label} {active ? (dir === 'desc' ? '↓' : '↑') : ''}
      </span>
    </Button>
  );
}

// ── main component ──

const CustomTransferBuilder = ({
  squad,
  bootstrapElements,
  bootstrapTeams,
  lookupPrediction,
  fixturesByName,
  onTransfer,
}: CustomTransferBuilderProps) => {
  const [selectedOutId, setSelectedOutId] = useState<number | null>(null);
  const [search, setSearch] = useState('');
  const [posFilter, setPosFilter] = useState<Position | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>('xPts');
  const [sortDir, setSortDir] = useState<SortDir>('desc');

  const teamMap = useMemo(
    () => new Map(bootstrapTeams.map((t) => [t.id, t])),
    [bootstrapTeams],
  );

  const squadIds = useMemo(() => new Set(squad.map((p) => p.id)), [squad]);

  const selectedOut = useMemo(
    () => (selectedOutId != null ? squad.find((p) => p.id === selectedOutId) ?? null : null),
    [squad, selectedOutId],
  );

  const bootstrapById = useMemo(
    () => new Map(bootstrapElements.map((el) => [el.id, el])),
    [bootstrapElements],
  );

  const squadRows = useMemo<RowData[]>(
    () =>
      squad.map((p) => {
        const fixture = fixturesByName.get(norm(p.name));
        const bel = bootstrapById.get(p.id);
        return {
          id: p.id,
          name: p.name,
          position: p.position,
          teamAbbr: p.teamAbbr,
          xPts: p.xPts,
          avgDiff: avgDiff5(fixture?.fixtures),
          photoUrl: p.photoUrl,
          price: bel?.now_cost ? bel.now_cost / 10 : p.price,
          form: Number(bel?.form ?? 0),
          ownership: Number(bel?.selected_by_percent ?? p.ownership ?? 0),
          nextFixture: nextFixtureLabel(fixture?.fixtures),
          isBench: p.isBench,
        };
      }),
    [squad, fixturesByName, bootstrapById],
  );

  const availableRows = useMemo<RowData[]>(() => {
    return bootstrapElements
      .filter((el) => !squadIds.has(el.id))
      .map((el) => {
        const pos = ELEM_POS[el.element_type];
        if (!pos) return null;
        const team = teamMap.get(el.team);
        const abbr = team?.short_name ?? '';
        const pred = lookupPrediction(el.web_name, abbr);
        const fixture = fixturesByName.get(norm(el.web_name));
        return {
          id: el.id,
          name: el.web_name,
          position: pos,
          teamAbbr: abbr,
          xPts: pred?.xp ?? 0,
          avgDiff: avgDiff5(fixture?.fixtures),
          photoUrl: mkPhoto(el.code),
          price: el.now_cost ? el.now_cost / 10 : 0,
          form: Number(el.form ?? 0),
          ownership: Number(el.selected_by_percent ?? 0),
          nextFixture: nextFixtureLabel(fixture?.fixtures),
        } as RowData;
      })
      .filter((r): r is RowData => r != null);
  }, [bootstrapElements, squadIds, teamMap, lookupPrediction, fixturesByName]);

  const effectivePos = selectedOut ? selectedOut.position : posFilter;
  const q = norm(search);

  const matchesFilter = (p: RowData) => {
    if (effectivePos && p.position !== effectivePos) return false;
    if (q && !norm(p.name).includes(q) && !norm(p.teamAbbr).includes(q)) return false;
    return true;
  };

  const filteredSquad = useMemo(
    () => sortRows(squadRows.filter(matchesFilter), sortKey, sortDir),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [squadRows, effectivePos, q, sortKey, sortDir],
  );

  const filteredAvailable = useMemo(
    () => sortRows(availableRows.filter(matchesFilter), sortKey, sortDir).slice(0, 100),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [availableRows, effectivePos, q, sortKey, sortDir],
  );

  const handleSortToggle = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === 'desc' ? 'asc' : 'desc'));
    } else {
      setSortKey(key);
      setSortDir(key === 'avgDiff' ? 'asc' : 'desc');
    }
  };

  const handleSelectOut = (id: number) => {
    setSelectedOutId(id);
    setSearch('');
  };

  const handleTransferIn = (playerInId: number) => {
    if (selectedOutId == null) return;
    onTransfer(playerInId, selectedOutId);
    setSelectedOutId(null);
    setSearch('');
  };

  const handleCancel = () => {
    setSelectedOutId(null);
    setSearch('');
  };

  if (bootstrapElements.length === 0) {
    return (
      <DashboardCard>
        <DashboardHeader title="Custom Transfer Builder" />
        <div className="px-5 py-6">
          <p className="text-center text-sm text-slate-500">
            Loading player data…
          </p>
        </div>
      </DashboardCard>
    );
  }

  return (
    <DashboardCard>
      <DashboardHeader
        title="Custom Transfer Builder"
        description="Make transfers in your sandbox squad — changes are applied instantly to the pitch below"
      />

      {/* ─── Step indicator ─── */}
      {!selectedOut ? (
        <div className="border-b border-white/6 bg-[rgba(59,130,246,0.06)] px-4 py-2">
          <p className="text-xs font-semibold text-blue-300">
            Step 1 of 2 — Pick a player from your squad to transfer out
          </p>
          <p className="text-[11px] text-slate-500">
            Tap a player row or press the red "Out" button
          </p>
        </div>
      ) : (
        <div>
          <div className="flex items-center gap-2 border-b border-white/6 bg-[rgba(244,63,94,0.08)] px-4 py-2">
            <Avatar size="sm">
              {selectedOut.photoUrl ? <AvatarImage src={selectedOut.photoUrl} alt={selectedOut.name} /> : null}
              <AvatarFallback className="bg-slate-700">{getInitials(selectedOut.name)}</AvatarFallback>
            </Avatar>
            <div className="min-w-0 flex-1">
              <p className="text-[10px] text-slate-500">Removing from squad</p>
              <p className="truncate text-sm font-bold text-red-300">
                {selectedOut.name}
                <span className="text-xs font-normal text-slate-500">
                  {' '}({selectedOut.position} · {selectedOut.teamAbbr})
                </span>
              </p>
            </div>
            <Button variant="outline" onClick={handleCancel} className="h-6 rounded-md px-2 text-xs text-slate-300">
              Cancel
            </Button>
          </div>

          <div className="border-b border-white/6 bg-[rgba(16,185,129,0.06)] px-4 py-2">
            <p className="text-xs font-semibold text-green-300">
              Step 2 of 2 — Pick a replacement ({selectedOut.position})
            </p>
            <p className="text-[11px] text-slate-500">
              Tap a player or press the green "In" button — the transfer updates your sandbox pitch
            </p>
          </div>
        </div>
      )}

      <div>
        {/* Position filter + search */}
        <div className="flex flex-col gap-2 px-4 py-3">
          <div className="flex flex-wrap items-center gap-1">
            <Button
              variant={effectivePos == null ? 'solid' : 'ghost'}
              onClick={() => setPosFilter(null)}
              disabled={selectedOut != null}
              className={`h-6 rounded-md px-2 text-[10px] ${
                effectivePos == null ? 'bg-blue-500 text-white hover:bg-blue-600' : 'text-slate-400'
              }`}
            >
              ALL
            </Button>
            {ALL_POSITIONS.map((pos) => (
              <Button
                key={pos}
                variant={effectivePos === pos ? 'solid' : 'ghost'}
                onClick={() => setPosFilter(pos)}
                disabled={selectedOut != null}
                className={`h-6 rounded-md px-2 text-[10px] ${
                  effectivePos === pos ? 'bg-blue-500 text-white hover:bg-blue-600' : 'text-slate-400'
                }`}
              >
                {pos}
              </Button>
            ))}
          </div>

          <input
            placeholder="Search by name or team…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="h-8 w-full rounded-md border border-white/8 bg-white/4 px-3 text-sm text-white placeholder:text-slate-500 focus-visible:border-emerald-400 focus-visible:outline-none"
          />
        </div>

        {/* Sort controls */}
        <div className="flex flex-wrap items-center gap-1 px-4 pb-2">
          <span className="mr-1 text-[9px] uppercase text-slate-600">
            Sort
          </span>
          {SORT_OPTIONS.map((opt) => (
            <SortChip
              key={opt.key}
              label={opt.label}
              active={sortKey === opt.key}
              dir={sortDir}
              onClick={() => handleSortToggle(opt.key)}
            />
          ))}
        </div>

        {/* Column headers */}
        <div className="flex items-center gap-2 px-4 py-1">
          <div className="w-[32px]" />
          <span className="flex-1 text-[9px] uppercase tracking-wide text-slate-600">
            Player
          </span>
          <span className="min-w-[28px] text-right text-[9px] uppercase text-slate-600">
            Form
          </span>
          <span className="min-w-[32px] text-right text-[9px] uppercase text-slate-600">
            xP
          </span>
          <span className="min-w-[34px] text-right text-[9px] uppercase text-slate-600">
            Sel%
          </span>
          <span className="min-w-[30px] text-center text-[9px] uppercase text-slate-600">
            FDR
          </span>
          <div className="w-[40px]" />
        </div>

        {/* Scrollable list */}
        <div className="card-scroll max-h-[520px] overflow-y-auto pb-2">
          {/* Your Squad (shown in step 1) */}
          {!selectedOut && filteredSquad.length > 0 && (
            <>
              <p className="px-4 pb-1 pt-2 text-[10px] font-bold uppercase tracking-widest text-slate-500">
                Your Squad ({filteredSquad.length})
              </p>
              {filteredSquad.map((p) => (
                <PlayerRow
                  key={p.id}
                  player={p}
                  actionLabel="Out"
                  actionColor="red"
                  onAction={() => handleSelectOut(p.id)}
                  onRowClick={() => handleSelectOut(p.id)}
                />
              ))}
            </>
          )}

          {/* Available Players */}
          <p className="px-4 pb-1 pt-3 text-[10px] font-bold uppercase tracking-widest text-slate-500">
            {selectedOut
              ? `Available ${selectedOut.position}s (${filteredAvailable.length})`
              : `All Players (${filteredAvailable.length})`}
          </p>
          {filteredAvailable.length > 0 ? (
            filteredAvailable.map((p) => (
              <PlayerRow
                key={p.id}
                player={p}
                actionLabel={selectedOut ? 'In' : undefined}
                actionColor="green"
                onAction={selectedOut ? () => handleTransferIn(p.id) : undefined}
                onRowClick={selectedOut ? () => handleTransferIn(p.id) : undefined}
              />
            ))
          ) : (
            <p className="px-4 py-4 text-center text-sm text-slate-600">
              No players match your filters
            </p>
          )}
        </div>
      </div>
    </DashboardCard>
  );
};

export default CustomTransferBuilder;
