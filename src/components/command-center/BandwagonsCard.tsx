import { useEffect, useMemo, useState } from 'react';
import { getBandwagons } from '../../api/backend';
import { getPlayerPhotoUrl } from '../../api/fpl/fpl';
import { DashboardCard, DashboardHeader } from '@/components/ui/primitives';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';

type NormalizedBandwagonPlayer = {
  id: number;
  name: string;
  team: string;
  transfersIn: number;
  transfersOut: number;
  balance: number;
  photoUrl?: string;
};

interface BootstrapElement {
  id: number;
  code: number;
  web_name: string;
  team: number;
}

interface Props {
  /** FPL bootstrap elements list – used to resolve player photos */
  bootstrapElements?: BootstrapElement[];
}

const resolvePhotoUrl = (
  code: number | undefined,
): string | undefined => {
  if (!code) return undefined;
  return getPlayerPhotoUrl(code);
};

const compactNumber = new Intl.NumberFormat('en', {
  notation: 'compact',
  compactDisplay: 'short',
  maximumFractionDigits: 1,
});

const formatCompact = (value: number) => compactNumber.format(value);

const getInitials = (name: string) =>
  name
    .split(' ')
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join('');

const BANDWAGONS_LIMIT = 10;

let cachedBandwagons: NormalizedBandwagonPlayer[] | null = null;

const BandwagonsCard = ({ bootstrapElements }: Props) => {
  const [loading, setLoading] = useState(cachedBandwagons == null);
  const [error, setError] = useState<string | null>(null);
  const [players, setPlayers] = useState<NormalizedBandwagonPlayer[]>(cachedBandwagons ?? []);

  const nameToCode = useMemo(() => {
    const map = new Map<string, number>();
    for (const el of bootstrapElements ?? []) {
      map.set(el.web_name.toLowerCase(), el.code);
    }
    return map;
  }, [bootstrapElements]);

  useEffect(() => {
    if (cachedBandwagons) {
      setPlayers(cachedBandwagons.map((player) => {
        if (player.photoUrl) return player;
        const code = nameToCode.get(player.name.toLowerCase());
        return { ...player, photoUrl: resolvePhotoUrl(code) };
      }));
      setLoading(false);
      return;
    }

    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);
      try {
        const data = await getBandwagons();

        const normalized: NormalizedBandwagonPlayer[] = data.map((player, idx) => {
          const id = player.player_id ?? player.element ?? idx;
          const name = player.player_name ?? player.name ?? player.web_name ?? `#${id}`;
          const team = player.team ?? player.team_short_name ?? player.team_name ?? '';
          const transfersIn = Number(player.transfers_in ?? 0);
          const transfersOut = Number(player.transfers_out ?? 0);
          const balance = Number(player.transfers_balance ?? (transfersIn - transfersOut));

          let photoUrl = player.photo_url;
          if (!photoUrl) {
            const code = player.code ?? nameToCode.get(name.toLowerCase());
            photoUrl = resolvePhotoUrl(code);
          }

          return { id, name, team, transfersIn, transfersOut, balance, photoUrl };
        });

        const sorted = normalized.slice().sort((a, b) => {
          const magnitudeDelta = Math.abs(b.balance) - Math.abs(a.balance);
          if (magnitudeDelta !== 0) return magnitudeDelta;
          return b.balance - a.balance;
        }).slice(0, BANDWAGONS_LIMIT);
        cachedBandwagons = sorted;

        if (!cancelled) {
          setPlayers(sorted);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load bandwagons data');
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
      <DashboardHeader title="Bandwagons" description="Most transferred players" />
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
                    <p
                      className={`text-sm font-bold ${
                        player.balance < 0
                          ? 'text-red-300'
                          : player.balance > 0
                            ? 'text-emerald-400'
                            : 'text-slate-300'
                      }`}
                    >
                      {player.balance > 0 ? '+' : ''}
                      {formatCompact(player.balance)}
                    </p>
                    <p className="text-[10px] text-slate-500">net</p>
                  </div>
                </div>
                <div className="ml-10 mt-1.5 flex flex-wrap items-center gap-3 text-xs">
                  <span className="flex items-center gap-1"><span className="text-slate-500">In:</span><span className="font-medium text-emerald-400">{formatCompact(player.transfersIn)}</span></span>
                  <span className="flex items-center gap-1"><span className="text-slate-500">Out:</span><span className="font-medium text-red-300">{formatCompact(player.transfersOut)}</span></span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </DashboardCard>
  );
};

export default BandwagonsCard;
