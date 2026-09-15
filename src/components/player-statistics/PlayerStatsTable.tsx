import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { buttonVariants } from '@/components/ui/button';
import { useMemo, useState } from 'react';
import type { PlayerStatsColumnKey } from './PlayerStatsFilters';
import type { PlayerStatsFixturePill, PlayerStatsRowModel } from '../../utils/playerStatsModel';
import { cn } from '@/lib/utils';

type SortDirection = 'asc' | 'desc';

type SortableColumnKey = Exclude<PlayerStatsColumnKey, 'nextFixtures'>;

type SortState = {
  key: SortableColumnKey;
  direction: SortDirection;
};

type PlayerStatsTableProps = {
  rows: PlayerStatsRowModel[];
  visibleColumns: PlayerStatsColumnKey[];
  onRowSelect?: (id: number) => void;
  onViewClick?: (id: number) => void;
  selectedRowId?: number;
  isLoading?: boolean;
  pageSize?: number;
  emptyText?: string;
};

const COLUMN_LABELS: Record<PlayerStatsColumnKey, string> = {
  name: 'Name',
  team: 'Team',
  pos: 'Position',
  price: 'Price',
  ownership: 'Ownership',
  minutes: 'Minutes',
  points: 'Points',
  xPts: 'xPts',
  goals: 'Goals',
  assists: 'Assists',
  xG: 'xG',
  xA: 'xA',
  xGI: 'xGI',
  goalsPer90: 'Goals/90',
  xGPer90: 'xG/90',
  nextFixtures: 'Next fixtures',
};

const SORTABLE_COLUMNS: Set<SortableColumnKey> = new Set([
  'name',
  'team',
  'pos',
  'price',
  'ownership',
  'minutes',
  'points',
  'xPts',
  'goals',
  'assists',
  'xG',
  'xA',
  'xGI',
  'goalsPer90',
  'xGPer90',
]);

const DEFAULT_PAGE_SIZE = 25;

const PlayerStatsTable = ({
  rows,
  visibleColumns,
  onRowSelect,
  onViewClick,
  selectedRowId,
  isLoading = false,
  pageSize = DEFAULT_PAGE_SIZE,
  emptyText = 'No players match the selected filters.',
}: PlayerStatsTableProps) => {
  const [sort, setSort] = useState<SortState>({ key: 'points', direction: 'desc' });
  const [page, setPage] = useState(1);

  const normalizedPageSize = Math.max(1, pageSize);

  const normalizedVisibleColumns = useMemo(
    () => visibleColumns.filter((column): column is PlayerStatsColumnKey => column in COLUMN_LABELS),
    [visibleColumns]
  );

  const sortedRows = useMemo(() => {
    const copy = [...rows];
    copy.sort((a, b) => compareRows(a, b, sort));
    return copy;
  }, [rows, sort]);

  const pageCount = Math.max(1, Math.ceil(sortedRows.length / normalizedPageSize));
  const normalizedPage = Math.min(page, pageCount);

  const pageRows = useMemo(() => {
    const start = (normalizedPage - 1) * normalizedPageSize;
    return sortedRows.slice(start, start + normalizedPageSize);
  }, [normalizedPage, normalizedPageSize, sortedRows]);

  const startIndex = sortedRows.length === 0 ? 0 : (normalizedPage - 1) * normalizedPageSize + 1;
  const endIndex = sortedRows.length === 0 ? 0 : Math.min(normalizedPage * normalizedPageSize, sortedRows.length);

  return (
    <div>
      <div className="card-scroll overflow-x-auto">
        <Table className="min-w-max border-collapse border-spacing-0">
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              {normalizedVisibleColumns.map((column) => {
                const sortable = SORTABLE_COLUMNS.has(column as SortableColumnKey);
                const isSorted = sortable && sort.key === column;

                return (
                  <TableHead
                    key={column}
                    className="sticky top-0 z-[1] bg-slate-900 whitespace-nowrap"
                    aria-sort={
                      sortable
                        ? isSorted
                          ? sort.direction === 'asc'
                            ? 'ascending'
                            : 'descending'
                          : 'none'
                        : undefined
                    }
                  >
                    {sortable ? (
                      <button
                        type="button"
                        onClick={() => setSort((current) => nextSortState(current, column as SortableColumnKey))}
                        aria-label={`Sort by ${COLUMN_LABELS[column]}`}
                        className="flex cursor-pointer items-center gap-1 p-0 hover:bg-white/8"
                      >
                        <span className="text-xs uppercase tracking-wide">
                          {COLUMN_LABELS[column]}
                        </span>
                        <span className={cn('text-xs', isSorted ? 'text-emerald-300' : 'text-slate-500')}>
                          {isSorted ? (sort.direction === 'asc' ? '▲' : '▼') : '↕'}
                        </span>
                      </button>
                    ) : (
                      <span className="text-xs uppercase tracking-wide text-slate-300">
                        {COLUMN_LABELS[column]}
                      </span>
                    )}
                  </TableHead>
                );
              })}
              {onViewClick ? (
                <TableHead className="sticky top-0 z-[1] bg-slate-900 text-right whitespace-nowrap">
                  <span className="text-xs uppercase tracking-wide text-slate-300">Action</span>
                </TableHead>
              ) : null}
            </TableRow>
          </TableHeader>

          <TableBody>
            {isLoading ? (
              <TableRow className="hover:bg-transparent">
                <TableCell colSpan={normalizedVisibleColumns.length + (onViewClick ? 1 : 0)} className="py-6 text-center text-slate-400">
                  Loading player statistics...
                </TableCell>
              </TableRow>
            ) : null}

            {!isLoading && pageRows.length === 0 ? (
              <TableRow className="hover:bg-transparent">
                <TableCell colSpan={normalizedVisibleColumns.length + (onViewClick ? 1 : 0)} className="py-6 text-center text-slate-400">
                  {emptyText}
                </TableCell>
              </TableRow>
            ) : null}

            {!isLoading
              ? pageRows.map((row) => {
                  const isClickable = Boolean(onRowSelect);
                  const isSelected = selectedRowId != null && selectedRowId === row.id;

                  return (
                    <TableRow
                      key={row.id}
                      tabIndex={isClickable ? 0 : -1}
                      role={isClickable ? 'button' : undefined}
                      className={cn(
                        'hover:bg-white/6 focus-visible:outline-2 focus-visible:outline-emerald-300 focus-visible:-outline-offset-2',
                        isClickable && 'cursor-pointer',
                        isSelected && 'bg-[rgba(56,189,248,0.12)] hover:bg-[rgba(56,189,248,0.12)]',
                      )}
                      onClick={() => onRowSelect?.(row.id)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') {
                          event.preventDefault();
                          onRowSelect?.(row.id);
                        }
                      }}
                    >
                      {normalizedVisibleColumns.map((column) => (
                        <TableCell key={`${row.id}-${column}`}>
                          {renderCell(row, column)}
                        </TableCell>
                      ))}
                      {onViewClick ? (
                        <TableCell className="text-right">
                          <button
                            type="button"
                            className={cn(buttonVariants({ variant: 'outline', size: 'sm' }), 'h-6 px-2 text-xs')}
                            onClick={(event) => {
                              event.stopPropagation();
                              onViewClick(row.id);
                            }}
                            aria-label={`View ${row.name}`}
                          >
                            View
                          </button>
                        </TableCell>
                      ) : null}
                    </TableRow>
                  );
                })
              : null}
          </TableBody>
        </Table>
      </div>

      <div className="mt-4 flex flex-wrap justify-between gap-4">
        <span className="text-sm text-slate-400">
          Showing {startIndex}-{endIndex} of {sortedRows.length}
        </span>
        <div className="flex items-center gap-2">
          <button
            type="button"
            className={cn(buttonVariants({ variant: 'outline', size: 'sm' }))}
            onClick={() => setPage((current) => Math.max(1, Math.min(current, pageCount) - 1))}
            disabled={normalizedPage <= 1}
          >
            Previous
          </button>
          <span className="text-sm text-slate-300">
            Page {normalizedPage} of {pageCount}
          </span>
          <button
            type="button"
            className={cn(buttonVariants({ variant: 'outline', size: 'sm' }))}
            onClick={() => setPage((current) => Math.min(pageCount, Math.min(current, pageCount) + 1))}
            disabled={normalizedPage >= pageCount}
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
};

function nextSortState(current: SortState, key: SortableColumnKey): SortState {
  if (current.key !== key) {
    return {
      key,
      direction: isNumericSortColumn(key) ? 'desc' : 'asc',
    };
  }

  return {
    key,
    direction: current.direction === 'asc' ? 'desc' : 'asc',
  };
}

function compareRows(a: PlayerStatsRowModel, b: PlayerStatsRowModel, sort: SortState): number {
  const directionMultiplier = sort.direction === 'asc' ? 1 : -1;

  if (sort.key === 'name') {
    return a.name.localeCompare(b.name) * directionMultiplier;
  }

  if (sort.key === 'team') {
    const teamCompare = a.teamAbbr.localeCompare(b.teamAbbr);
    if (teamCompare !== 0) {
      return teamCompare * directionMultiplier;
    }
    return a.name.localeCompare(b.name) * directionMultiplier;
  }

  if (sort.key === 'pos') {
    const posCompare = a.position.localeCompare(b.position);
    if (posCompare !== 0) {
      return posCompare * directionMultiplier;
    }
    return a.name.localeCompare(b.name) * directionMultiplier;
  }

  const numericDiff = getNumericValue(a, sort.key) - getNumericValue(b, sort.key);
  if (numericDiff !== 0) {
    return numericDiff * directionMultiplier;
  }

  return a.name.localeCompare(b.name) * directionMultiplier;
}

function getNumericValue(row: PlayerStatsRowModel, key: Exclude<SortableColumnKey, 'name' | 'team' | 'pos'>): number {
  if (key === 'price') {
    return parseDisplayNumber(row.price);
  }

  if (key === 'ownership') {
    return parseDisplayNumber(row.ownership);
  }

  if (key === 'minutes') return row.minutes;
  if (key === 'points') return row.points;
  if (key === 'xPts') return row.xPts;
  if (key === 'goals') return row.goals;
  if (key === 'assists') return row.assists;
  if (key === 'xG') return row.xG;
  if (key === 'xA') return row.xA;
  if (key === 'xGI') return row.xGI;
  if (key === 'goalsPer90') return row.goalsPer90;
  return row.xgPer90;
}

function parseDisplayNumber(value: string): number {
  const parsed = Number.parseFloat(value.replace(/[^\d.-]/g, ''));
  return Number.isFinite(parsed) ? parsed : 0;
}

function renderCell(row: PlayerStatsRowModel, column: PlayerStatsColumnKey) {
  if (column === 'name') {
    return (
      <span className="font-semibold text-white">
        {row.name}
      </span>
    );
  }

  if (column === 'team') return row.teamAbbr;
  if (column === 'pos') return row.position;
  if (column === 'price') return row.price;
  if (column === 'ownership') return row.ownership;
  if (column === 'minutes') return row.minutes;
  if (column === 'points') return row.points;
  if (column === 'xPts') return row.xPts.toFixed(1);
  if (column === 'goals') return row.goals;
  if (column === 'assists') return row.assists;
  if (column === 'xG') return row.xG.toFixed(2);
  if (column === 'xA') return row.xA.toFixed(2);
  if (column === 'xGI') return row.xGI.toFixed(2);
  if (column === 'goalsPer90') return row.goalsPer90.toFixed(2);
  if (column === 'xGPer90') return row.xgPer90.toFixed(2);

  return <FixturesPills fixtures={row.nextFixtures} />;
}

function FixturesPills({ fixtures }: { fixtures: PlayerStatsFixturePill[] }) {
  if (fixtures.length === 0) {
    return <span className="text-slate-500">-</span>;
  }

  return (
    <div className="flex items-center gap-1">
      {fixtures.map((fixture, index) => {
        const style = getDifficultyStyle(fixture.fdr);
        return (
          <span
            key={`${fixture.opponentAbbr}-${fixture.home ? 'H' : 'A'}-${index}`}
            className="rounded-md border px-2 py-1 text-[10px] normal-case"
            style={{ backgroundColor: style.bg, color: style.color, borderColor: style.borderColor }}
          >
            {fixture.home ? 'vs' : '@'} {fixture.opponentAbbr}
          </span>
        );
      })}
    </div>
  );
}

function getDifficultyStyle(difficulty: number) {
  if (difficulty === 1) {
    return {
      bg: 'rgba(16, 185, 129, 0.12)',
      color: '#34d399',
      borderColor: 'rgba(16, 185, 129, 0.22)',
    };
  }
  if (difficulty === 2) {
    return {
      bg: 'rgba(34, 197, 94, 0.12)',
      color: '#86efac',
      borderColor: 'rgba(34, 197, 94, 0.22)',
    };
  }
  if (difficulty === 3) {
    return {
      bg: 'rgba(100, 116, 139, 0.12)',
      color: '#cbd5e1',
      borderColor: 'rgba(100, 116, 139, 0.22)',
    };
  }
  if (difficulty === 4) {
    return {
      bg: 'rgba(251, 146, 60, 0.12)',
      color: '#fdba74',
      borderColor: 'rgba(251, 146, 60, 0.22)',
    };
  }
  return {
    bg: 'rgba(248, 113, 113, 0.12)',
    color: '#fca5a5',
    borderColor: 'rgba(248, 113, 113, 0.22)',
  };
}

function isNumericSortColumn(key: SortableColumnKey): boolean {
  return key !== 'name' && key !== 'team' && key !== 'pos';
}

export default PlayerStatsTable;
