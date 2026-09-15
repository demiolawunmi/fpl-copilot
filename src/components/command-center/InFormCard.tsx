import { useEffect, useMemo, useState } from 'react';
import { getFormLast4 } from '../../api/backend';
import { getPlayerPhotoUrl } from '../../api/fpl/fpl';
import { DashboardCard, DashboardHeader } from '@/components/ui/primitives';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';

type NormalizedFormPlayer = {
  id: number;
  name: string;
  team: string;
  last4Points: number;
  last4Minutes: number;
  xG?: number;
  xA?: number;
  photoUrl?: string;
};

interface BootstrapElement {
  id: number;
  code: number;
  web_name: string;
  team: number;
}

interface Props {
  bootstrapElements?: BootstrapElement[];
}

const resolvePhotoUrl = (code: number | undefined): string | undefined => {
  if (!code) return undefined;
  return getPlayerPhotoUrl(code);
};

const getInitials = (name: string) =>
  name
    .split(' ')
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join('');

type CachedFormPlayer = Omit<NormalizedFormPlayer, 'photoUrl'>;

const IN_FORM_LIMIT = 20;

let cachedFormPlayers: CachedFormPlayer[] | null = null;

const InFormCard = ({ bootstrapElements }: Props) => {
  const [loading, setLoading] = useState(cachedFormPlayers == null);
  const [error, setError] = useState<string | null>(null);
  const [players, setPlayers] = useState<NormalizedFormPlayer[]>([]);

  const nameToCode = useMemo(() => {
    const map = new Map<string, number>();
    for (const el of bootstrapElements ?? []) {
      map.set(el.web_name.toLowerCase(), el.code);
    }
    return map;
  }, [bootstrapElements]);

  useEffect(() => {
    if (cachedFormPlayers) {
      setPlayers(cachedFormPlayers.map((player) => ({
        ...player,
        photoUrl: resolvePhotoUrl(nameToCode.get(player.name.toLowerCase())),
      })));
      setLoading(false);
      return;
    }

    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);
      try {
        const data = await getFormLast4();

        const normalized: CachedFormPlayer[] = data.map((player, idx) => {
          const id = player.player_id ?? player.element ?? idx;
          const name = player.player_name ?? player.name ?? player.web_name ?? `#${id}`;
          const team = player.team ?? player.team_short_name ?? player.team_name ?? '';
          const last4Points = Number(player.last4_points ?? player.last_4_points ?? player.points_last4 ?? 0);
          const last4Minutes = Number(player.last4_minutes ?? player.last_4_minutes ?? player.minutes_last4 ?? 0);

          return {
            id,
            name,
            team,
            last4Points,
            last4Minutes,
            xG: Number(player.xG ?? player.xg ?? player.expected_goals ?? 0) || undefined,
            xA: Number(player.xA ?? player.xa ?? player.expected_assists ?? 0) || undefined,
          };
        });

        const sorted = normalized.slice().sort((a, b) => {
          if (b.last4Points !== a.last4Points) return b.last4Points - a.last4Points;
          return b.last4Minutes - a.last4Minutes;
        }).slice(0, IN_FORM_LIMIT);

        cachedFormPlayers = sorted;

        if (!cancelled) {
          setPlayers(sorted.map((player) => ({
            ...player,
            photoUrl: resolvePhotoUrl(nameToCode.get(player.name.toLowerCase())),
          })));
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load form data');
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void load();

    return () => {
      cancelled = true;
    };
  }, [nameToCode]);

  return (
    <DashboardCard>
      <DashboardHeader title="In Form (Last 4)" description="Players performing well recently" />
      <div className="card-scroll max-h-96 overflow-y-auto px-5 py-4">
        {loading ? (
          <p className="py-4 text-center text-sm text-slate-400">Loading...</p>
        ) : error ? (
          <p className="py-4 text-center text-sm text-red-300">{error}</p>
        ) : players.length === 0 ? (
          <p className="py-4 text-center text-sm text-slate-400">No data available</p>
        ) : (
          <div className="flex flex-col gap-3">
            {players.map((player, idx) => (
              <div
                key={player.id}
                className={
                  idx === players.length - 1
                    ? 'pb-3'
                    : 'border-b border-white/6 pb-3'
                }
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex min-w-0 flex-1 items-center gap-2">
                    <span className="w-5 shrink-0 text-xs font-bold text-slate-500">{idx + 1}</span>
                    <Avatar size="sm" className="size-6 bg-slate-700 text-slate-300">
                      {player.photoUrl ? <AvatarImage src={player.photoUrl} alt={player.name} /> : null}
                      <AvatarFallback>{getInitials(player.name)}</AvatarFallback>
                    </Avatar>
                    <div className="min-w-0">
                      <p className="truncate text-sm font-semibold text-white">{player.name}</p>
                      <p className="text-xs text-slate-400">{player.team}</p>
                    </div>
                  </div>
                  <div className="shrink-0 text-right">
                    <p className="text-sm font-bold text-emerald-400">{player.last4Points}</p>
                    <p className="text-[10px] text-slate-500">pts</p>
                  </div>
                </div>
                <div className="ml-10 mt-1.5 flex flex-wrap items-center gap-3 text-xs">
                  <span className="flex items-center gap-1"><span className="text-slate-500">Min:</span><span className="font-medium text-slate-300">{player.last4Minutes}</span></span>
                  {player.xG !== undefined ? <span className="flex items-center gap-1"><span className="text-slate-500">xG:</span><span className="font-medium text-slate-300">{Number(player.xG).toFixed(2)}</span></span> : null}
                  {player.xA !== undefined ? <span className="flex items-center gap-1"><span className="text-slate-500">xA:</span><span className="font-medium text-slate-300">{Number(player.xA).toFixed(2)}</span></span> : null}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </DashboardCard>
  );
};

export default InFormCard;
