import { useEffect, useMemo, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { fetchJson } from '../api/fpl/client';
import { fplEndpoints } from '../api/fpl/endpoints';
import { getBootstrap } from '../api/fpl/fpl';
import PlayerLeaderboardCards, {
  buildPlayerLeaderboardCardsData,
} from '../components/player-statistics/PlayerLeaderboardCards';
import PlayerStatsFilters, {
  PLAYER_STATS_DEFAULT_FILTERS,
  type PlayerStatsFiltersState,
  usePersistedPlayerStatsColumns,
} from '../components/player-statistics/PlayerStatsFilters';
import PlayerStatsTable from '../components/player-statistics/PlayerStatsTable';
import { DashboardCard, DashboardHeader } from '@/components/ui/primitives';
import { Skeleton } from '@/components/ui/skeleton';
import { usePredictionsData } from '../hooks/usePredictionsData';
import { createTeamAbbreviationMap } from '../utils/playerStatsFormat';
import {
  mapBootstrapElementsToPlayerStatsRows,
  type FplFixtureLite,
  type PlayerStatsRowModel,
  type TeamLite,
} from '../utils/playerStatsModel';

const PlayersPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [rows, setRows] = useState<PlayerStatsRowModel[]>([]);
  const [nextGwId, setNextGwId] = useState<number | null>(null);
  const [teamOptions, setTeamOptions] = useState<Array<{ value: string; label: string }>>([]);
  const [filters, setFilters] = useState<PlayerStatsFiltersState>(PLAYER_STATS_DEFAULT_FILTERS);
  const [leaderboardCards, setLeaderboardCards] = useState<
    ReturnType<typeof buildPlayerLeaderboardCardsData>
  >([]);
  const { visibleColumns, setVisibleColumns } = usePersistedPlayerStatsColumns();
  const predictions = usePredictionsData(nextGwId);

  useEffect(() => {
    let active = true;

    const loadData = async () => {
      setLoading(true);
      setError(null);

      try {
        const [bootstrap, fixtures] = await Promise.all([
          getBootstrap(),
          fetchJson<FplFixtureLite[]>(fplEndpoints.fixtures()),
        ]);

        if (!active) {
          return;
        }

        const teams: TeamLite[] = bootstrap.teams.map((team) => ({
          id: team.id,
          short_name: team.short_name,
        }));

        const mappedRows = mapBootstrapElementsToPlayerStatsRows({
          elements: bootstrap.elements,
          teams,
          fixtures,
          nextFixturesLimit: 5,
        });

        const bootstrapEvents = (bootstrap as { events?: Array<{ id: number; is_next?: boolean }> }).events;
        const nextEv = bootstrapEvents?.find((e) => e.is_next);
        setNextGwId(nextEv?.id ?? null);

        const teamMap = createTeamAbbreviationMap(teams);
        const cards = buildPlayerLeaderboardCardsData({
          elements: bootstrap.elements,
          teamMap,
          topN: 3,
        });

        const nextTeamOptions = bootstrap.teams
          .slice()
          .sort((a, b) => a.short_name.localeCompare(b.short_name))
          .map((team) => ({ value: team.short_name, label: team.short_name }));

        setRows(mappedRows);
        setLeaderboardCards(cards);
        setTeamOptions(nextTeamOptions);
      } catch (loadError) {
        if (!active) {
          return;
        }

        const message =
          loadError instanceof Error ? loadError.message : 'Could not load player statistics.';
        setError(message);
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    };

    void loadData();

    return () => {
      active = false;
    };
  }, []);

  const rowsWithPredictions = useMemo(() => {
    return rows.map((row) => {
      const pred = predictions.lookupPrediction(row.name, row.teamAbbr);
      const xp = pred?.xp;
      if (pred != null && xp != null && Number.isFinite(xp)) {
        return { ...row, xPts: xp };
      }
      return row;
    });
  }, [rows, predictions]);

  const filteredRows = useMemo(() => {
    const searchQuery = filters.search.trim().toLowerCase();

    return rowsWithPredictions.filter((row) => {
      if (searchQuery && !row.name.toLowerCase().includes(searchQuery)) {
        return false;
      }

      if (filters.team !== 'all' && row.teamAbbr !== filters.team) {
        return false;
      }

      if (filters.position !== 'all' && row.position !== filters.position) {
        return false;
      }

      if (filters.preset === 'forwards-xg90') {
        return row.position === 'FWD' && row.xgPer90 > 0;
      }

      if (filters.preset === 'midfield-creativity') {
        return row.position === 'MID' && row.xA >= 1;
      }

      if (filters.preset === 'budget-differentials') {
        return parseNumberFromText(row.price) <= 6.5 && parseNumberFromText(row.ownership) < 10;
      }

      return true;
    });
  }, [filters, rowsWithPredictions]);

  const handleRowSelect = (id: number) => {
    navigate(`/players/${id}`, { state: { from: location.pathname } });
  };

  const handleViewClick = (id: number) => {
    navigate(`/players/${id}`, { state: { from: location.pathname } });
  };

  return (
    <div className="mx-auto flex w-full max-w-[90rem] flex-1 flex-col px-4 py-6 md:px-6 xl:px-10 xl:py-8">
      <div className="flex flex-col gap-6">
        <div className="flex flex-col gap-1.5">
          <h1 className="text-2xl font-bold leading-[1.33] text-white">Player Statistics</h1>
          <p className="text-sm text-slate-400">
            Season to date from official FPL bootstrap and fixtures data.
          </p>
        </div>

        {error && !loading ? (
          <div
            role="alert"
            className="rounded-xl border bg-[rgba(234,179,8,0.08)] px-4 py-3"
            style={{ borderColor: 'rgba(234, 179, 8, 0.2)' }}
          >
            <p className="text-sm text-yellow-300">
              Couldn&apos;t load player statistics. {error}
            </p>
          </div>
        ) : null}

        {loading ? (
          <div className="flex flex-col gap-6">
            <div className="flex flex-col gap-4 lg:flex-row">
              {Array.from({ length: 3 }).map((_, index) => (
                <Skeleton key={`leader-skeleton-${index}`} className="h-[220px] flex-1 rounded-2xl" />
              ))}
            </div>
            <DashboardCard>
              <DashboardHeader title="Filters" description="Search, narrow, and customize columns" />
              <div className="flex flex-col gap-4 px-5 py-4">
                <Skeleton className="h-[42px] rounded-md" />
                <Skeleton className="h-[42px] rounded-md" />
                <Skeleton className="h-[42px] rounded-md" />
              </div>
            </DashboardCard>
            <DashboardCard>
              <DashboardHeader title="Statistics table" description="All players" />
              <div className="flex flex-col gap-3 px-5 py-4">
                {Array.from({ length: 8 }).map((_, index) => (
                  <Skeleton key={`table-skeleton-${index}`} className="h-4" />
                ))}
              </div>
            </DashboardCard>
          </div>
        ) : (
          <>
            <PlayerLeaderboardCards cards={leaderboardCards} />

            <DashboardCard>
              <DashboardHeader
                title="Filters"
                description="Search by player, filter by team and position, and customize visible columns."
              />
              <div className="flex flex-col gap-4 px-5 py-4">
                <PlayerStatsFilters
                  value={filters}
                  onChange={setFilters}
                  teamOptions={teamOptions}
                  visibleColumns={visibleColumns}
                  onVisibleColumnsChange={setVisibleColumns}
                />
              </div>
            </DashboardCard>

            <DashboardCard>
              <DashboardHeader
                title="Statistics table"
                description="Sortable season metrics for all players in the game."
              />
              <div className="flex flex-col gap-4 px-5 py-4">
                <PlayerStatsTable
                  rows={filteredRows}
                  visibleColumns={visibleColumns}
                  onRowSelect={handleRowSelect}
                  onViewClick={handleViewClick}
                  emptyText="No players match your filters right now."
                />
              </div>
            </DashboardCard>
          </>
        )}
      </div>
    </div>
  );
};

function parseNumberFromText(value: string): number {
  const parsed = Number.parseFloat(value.replace(/[^\d.-]/g, ''));
  return Number.isFinite(parsed) ? parsed : 0;
}

export default PlayersPage;
