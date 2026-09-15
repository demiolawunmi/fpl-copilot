import { useEffect, useState } from 'react';
import { getSeasonStatus, type SeasonStatus } from '../../api/backend';

const formatSeason = (season: string | null): string => {
  if (!season || season.length !== 4) return season ?? 'unknown';
  return `20${season.slice(0, 2)}/${season.slice(2)}`;
};

/**
 * Warns when backend snapshots (AIrsenal exports / teams.json) still belong to
 * a previous season compared to the live FPL feed. Renders nothing while data
 * is current or while the status is being fetched / unavailable.
 */
const SeasonStatusBanner = () => {
  const [status, setStatus] = useState<SeasonStatus | null>(null);

  useEffect(() => {
    let cancelled = false;
    getSeasonStatus()
      .then((s) => {
        if (!cancelled) setStatus(s);
      })
      .catch(() => {
        // backend unavailable – stay silent
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (!status || status.is_current || !status.fpl_season) return null;

  return (
    <div
      className="rounded-md border px-5 py-3"
      style={{
        backgroundColor: 'rgba(245, 158, 11, 0.10)',
        borderColor: 'rgba(245, 158, 11, 0.35)',
      }}
    >
      <div className="flex items-start gap-3">
        <span className="text-lg" aria-hidden>⚠️</span>
        <div>
          <p className="text-sm font-semibold text-orange-200">
            Season data needs refreshing
          </p>
          <p className="mt-1 text-xs text-slate-300">
            The live FPL season is {formatSeason(status.fpl_season)}, but backend
            snapshots are from {formatSeason(status.data_season)}. Squad,
            predictions, form and bandwagons are serving last-season data until
            you run <code>airsenal_update_db</code> plus the export
            pipeline.
          </p>
        </div>
      </div>
    </div>
  );
};

export default SeasonStatusBanner;
