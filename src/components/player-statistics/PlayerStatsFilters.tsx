import { FiChevronDown } from 'react-icons/fi';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Switch } from '@/components/ui/switch';
import { useEffect, useMemo, useState } from 'react';

export const PLAYER_STATS_COLUMNS_STORAGE_KEY = 'fpl-copilot:player-stats-columns-v1';

export type PlayerStatsColumnKey =
  | 'name'
  | 'team'
  | 'pos'
  | 'price'
  | 'ownership'
  | 'minutes'
  | 'points'
  | 'xPts'
  | 'goals'
  | 'assists'
  | 'xG'
  | 'xA'
  | 'xGI'
  | 'goalsPer90'
  | 'xGPer90'
  | 'nextFixtures';

export type PlayerStatsPresetKey =
  | 'all'
  | 'forwards-xg90'
  | 'midfield-creativity'
  | 'budget-differentials';

export type PlayerStatsPositionFilter = 'all' | 'GK' | 'DEF' | 'MID' | 'FWD';

export type PlayerStatsFiltersState = {
  search: string;
  team: string;
  position: PlayerStatsPositionFilter;
  preset: PlayerStatsPresetKey;
};

export type PlayerStatsSelectOption = {
  value: string;
  label: string;
};

export type PlayerStatsPresetOption = {
  key: PlayerStatsPresetKey;
  label: string;
};

export type PlayerStatsColumnDefinition = {
  key: PlayerStatsColumnKey;
  label: string;
};

export const PLAYER_STATS_DEFAULT_VISIBLE_COLUMNS: PlayerStatsColumnKey[] = [
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
  'xGI',
  'goalsPer90',
  'xGPer90',
  'nextFixtures',
];

export const PLAYER_STATS_COLUMN_DEFINITIONS: PlayerStatsColumnDefinition[] = [
  { key: 'name', label: 'Name' },
  { key: 'team', label: 'Team' },
  { key: 'pos', label: 'Position' },
  { key: 'price', label: 'Price' },
  { key: 'ownership', label: 'Ownership' },
  { key: 'minutes', label: 'Minutes' },
  { key: 'points', label: 'Points' },
  { key: 'xPts', label: 'xPts' },
  { key: 'goals', label: 'Goals' },
  { key: 'assists', label: 'Assists' },
  { key: 'xG', label: 'xG' },
  { key: 'xA', label: 'xA' },
  { key: 'xGI', label: 'xGI' },
  { key: 'goalsPer90', label: 'Goals/90' },
  { key: 'xGPer90', label: 'xG/90' },
  { key: 'nextFixtures', label: 'Next fixtures' },
];

export const PLAYER_STATS_POSITION_OPTIONS: PlayerStatsSelectOption[] = [
  { value: 'all', label: 'All positions' },
  { value: 'GK', label: 'Goalkeepers' },
  { value: 'DEF', label: 'Defenders' },
  { value: 'MID', label: 'Midfielders' },
  { value: 'FWD', label: 'Forwards' },
];

export const PLAYER_STATS_PRESET_OPTIONS: PlayerStatsPresetOption[] = [
  { key: 'all', label: 'All players' },
  { key: 'forwards-xg90', label: 'Forwards: xG/90' },
  { key: 'midfield-creativity', label: 'Midfield creators' },
  { key: 'budget-differentials', label: 'Budget differentials' },
];

export const PLAYER_STATS_DEFAULT_FILTERS: PlayerStatsFiltersState = {
  search: '',
  team: 'all',
  position: 'all',
  preset: 'all',
};

type PlayerStatsFiltersProps = {
  value: PlayerStatsFiltersState;
  onChange: (next: PlayerStatsFiltersState) => void;
  teamOptions: PlayerStatsSelectOption[];
  visibleColumns: PlayerStatsColumnKey[];
  onVisibleColumnsChange: (next: PlayerStatsColumnKey[]) => void;
  availableColumns?: PlayerStatsColumnDefinition[];
  positionOptions?: PlayerStatsSelectOption[];
  presetOptions?: PlayerStatsPresetOption[];
};

const selectClassName =
  'h-10 w-full appearance-none rounded-lg border border-white/8 bg-slate-800 px-3 pr-8 text-sm text-white hover:border-white/12 focus-visible:border-emerald-400 focus-visible:outline-none';

const PlayerStatsFilters = ({
  value,
  onChange,
  teamOptions,
  visibleColumns,
  onVisibleColumnsChange,
  availableColumns = PLAYER_STATS_COLUMN_DEFINITIONS,
  positionOptions = PLAYER_STATS_POSITION_OPTIONS,
  presetOptions = PLAYER_STATS_PRESET_OPTIONS,
}: PlayerStatsFiltersProps) => {
  const [isColumnPickerOpen, setIsColumnPickerOpen] = useState(false);
  const availableColumnKeys = useMemo(
    () => new Set(availableColumns.map((column) => column.key)),
    [availableColumns]
  );

  const normalizedVisibleColumns = useMemo(
    () => sanitizeVisibleColumns(visibleColumns, availableColumns),
    [availableColumns, visibleColumns]
  );

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col items-end gap-4 md:flex-row">
        <div className="w-full">
          <label htmlFor="player-stats-search" className="mb-2 block text-sm font-medium">
            Search players
          </label>
          <input
            id="player-stats-search"
            value={value.search}
            onChange={(event) => onChange({ ...value, search: event.target.value })}
            placeholder="Search by player name"
            aria-label="Search players"
            className="h-10 w-full rounded-lg border border-white/8 bg-slate-800 px-3 text-sm text-white hover:border-white/12 placeholder:text-slate-500 focus-visible:border-emerald-400 focus-visible:shadow-[0_0_0_1px_#34d399] focus-visible:outline-none"
          />
        </div>

        <div className="w-full">
          <label htmlFor="player-stats-team" className="mb-2 block text-sm font-medium">
            Team
          </label>
          <div className="relative">
            <select
              id="player-stats-team"
              value={value.team}
              onChange={(event) => onChange({ ...value, team: event.target.value })}
              aria-label="Filter by team"
              className={selectClassName}
            >
              <option value="all">All teams</option>
              {teamOptions.map((team) => (
                <option key={team.value} value={team.value}>
                  {team.label}
                </option>
              ))}
            </select>
            <FiChevronDown size={16} className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-slate-400" />
          </div>
        </div>

        <div className="w-full">
          <label htmlFor="player-stats-position" className="mb-2 block text-sm font-medium">
            Position
          </label>
          <div className="relative">
            <select
              id="player-stats-position"
              value={value.position}
              onChange={(event) =>
                onChange({
                  ...value,
                  position: isPositionFilter(event.target.value) ? event.target.value : 'all',
                })
              }
              aria-label="Filter by position"
              className={selectClassName}
            >
              {positionOptions.map((position) => (
                <option key={position.value} value={position.value}>
                  {position.label}
                </option>
              ))}
            </select>
            <FiChevronDown size={16} className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-slate-400" />
          </div>
        </div>

        <div className="w-full">
          <label htmlFor="player-stats-preset" className="mb-2 block text-sm font-medium">
            Preset
          </label>
          <div className="relative">
            <select
              id="player-stats-preset"
              value={value.preset}
              onChange={(event) =>
                onChange({
                  ...value,
                  preset: isPresetKey(event.target.value) ? event.target.value : 'all',
                })
              }
              aria-label="Filter preset"
              className={selectClassName}
            >
              {presetOptions.map((preset) => (
                <option key={preset.key} value={preset.key}>
                  {preset.label}
                </option>
              ))}
            </select>
            <FiChevronDown size={16} className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-slate-400" />
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between">
        <p className="text-sm text-slate-400">
          {normalizedVisibleColumns.length} of {availableColumns.length} columns visible
        </p>
        <Button
          type="button"
          variant="outline"
          onClick={() => setIsColumnPickerOpen(true)}
          aria-haspopup="dialog"
          aria-controls="player-stats-column-picker"
        >
          Customize columns
        </Button>
      </div>

      <Dialog open={isColumnPickerOpen} onOpenChange={(o) => !o && setIsColumnPickerOpen(false)}>
        <DialogContent id="player-stats-column-picker" className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>Visible columns</DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-4">
            {availableColumns.map((column) => {
              const controlId = `player-stats-column-${column.key}`;
              const isChecked = normalizedVisibleColumns.includes(column.key);

              return (
                <div
                  key={column.key}
                  className="flex items-center justify-between"
                >
                  <label htmlFor={controlId} className="text-sm font-medium">
                    {column.label}
                  </label>
                  <Switch
                    id={controlId}
                    checked={isChecked}
                    onCheckedChange={() => {
                      if (!availableColumnKeys.has(column.key)) {
                        return;
                      }

                      const next = isChecked
                        ? normalizedVisibleColumns.filter((key) => key !== column.key)
                        : [...normalizedVisibleColumns, column.key];
                      onVisibleColumnsChange(sanitizeVisibleColumns(next, availableColumns));
                    }}
                    aria-label={`Toggle ${column.label} column`}
                  />
                </div>
              );
            })}
          </div>
          <DialogFooter className="bg-transparent">
            <Button
              type="button"
              variant="ghost"
              onClick={() =>
                onVisibleColumnsChange(
                  sanitizeVisibleColumns(PLAYER_STATS_DEFAULT_VISIBLE_COLUMNS, availableColumns)
                )
              }
            >
              Reset defaults
            </Button>
            <Button type="button" onClick={() => setIsColumnPickerOpen(false)}>
              Done
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export function usePersistedPlayerStatsColumns(options?: {
  storageKey?: string;
  availableColumns?: PlayerStatsColumnDefinition[];
  initialColumns?: PlayerStatsColumnKey[];
}) {
  const storageKey = options?.storageKey ?? PLAYER_STATS_COLUMNS_STORAGE_KEY;
  const availableColumns = options?.availableColumns ?? PLAYER_STATS_COLUMN_DEFINITIONS;
  const initialColumns =
    options?.initialColumns ??
    sanitizeVisibleColumns(PLAYER_STATS_DEFAULT_VISIBLE_COLUMNS, availableColumns);

  const [visibleColumns, setVisibleColumns] = useState<PlayerStatsColumnKey[]>(() =>
    readPlayerStatsColumnsFromStorage({ storageKey, availableColumns, fallbackColumns: initialColumns })
  );

  useEffect(() => {
    const sanitized = sanitizeVisibleColumns(visibleColumns, availableColumns);
    writePlayerStatsColumnsToStorage(sanitized, storageKey);
  }, [availableColumns, storageKey, visibleColumns]);

  return {
    visibleColumns,
    setVisibleColumns,
    resetVisibleColumns: () =>
      setVisibleColumns(sanitizeVisibleColumns(PLAYER_STATS_DEFAULT_VISIBLE_COLUMNS, availableColumns)),
  };
}

export function readPlayerStatsColumnsFromStorage(input?: {
  storageKey?: string;
  availableColumns?: PlayerStatsColumnDefinition[];
  fallbackColumns?: PlayerStatsColumnKey[];
}): PlayerStatsColumnKey[] {
  const storageKey = input?.storageKey ?? PLAYER_STATS_COLUMNS_STORAGE_KEY;
  const availableColumns = input?.availableColumns ?? PLAYER_STATS_COLUMN_DEFINITIONS;
  const fallbackColumns =
    input?.fallbackColumns ??
    sanitizeVisibleColumns(PLAYER_STATS_DEFAULT_VISIBLE_COLUMNS, availableColumns);

  if (typeof window === 'undefined') {
    return fallbackColumns;
  }

  const stored = window.localStorage.getItem(storageKey);
  if (!stored) {
    return fallbackColumns;
  }

  try {
    const parsed = JSON.parse(stored);
    if (!Array.isArray(parsed)) {
      return fallbackColumns;
    }

    const parsedColumns = parsed.filter(isPlayerStatsColumnKey);
    return sanitizeVisibleColumns(parsedColumns, availableColumns);
  } catch {
    return fallbackColumns;
  }
}

export function writePlayerStatsColumnsToStorage(
  columns: PlayerStatsColumnKey[],
  storageKey = PLAYER_STATS_COLUMNS_STORAGE_KEY
) {
  if (typeof window === 'undefined') {
    return;
  }

  window.localStorage.setItem(storageKey, JSON.stringify(columns));
}

export function sanitizeVisibleColumns(
  columns: PlayerStatsColumnKey[],
  availableColumns: PlayerStatsColumnDefinition[] = PLAYER_STATS_COLUMN_DEFINITIONS
): PlayerStatsColumnKey[] {
  const available = new Set(availableColumns.map((column) => column.key));
  const deduped = Array.from(new Set(columns.filter((column) => available.has(column))));

  if (deduped.length > 0) {
    return deduped;
  }

  const defaultColumns = PLAYER_STATS_DEFAULT_VISIBLE_COLUMNS.filter((column) => available.has(column));
  if (defaultColumns.length > 0) {
    return defaultColumns;
  }

  return availableColumns.slice(0, 1).map((column) => column.key);
}

function isPositionFilter(value: string): value is PlayerStatsPositionFilter {
  return value === 'all' || value === 'GK' || value === 'DEF' || value === 'MID' || value === 'FWD';
}

function isPresetKey(value: string): value is PlayerStatsPresetKey {
  return (
    value === 'all' ||
    value === 'forwards-xg90' ||
    value === 'midfield-creativity' ||
    value === 'budget-differentials'
  );
}

function isPlayerStatsColumnKey(value: unknown): value is PlayerStatsColumnKey {
  return typeof value === 'string' && PLAYER_STATS_COLUMN_DEFINITIONS.some((column) => column.key === value);
}

export default PlayerStatsFilters;
