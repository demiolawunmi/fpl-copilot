import type { TeamStatus } from '../../data/commandCenterMocks';
import { DashboardCard } from '@/components/ui/primitives';
import { Badge } from '@/components/ui/badge';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';

interface StatusStripConfig {
  // Tooltip vertical offset in pixels (how far above the bar the tooltip appears)
  tooltipOffsetPx?: number;
  // Wildcard season split cutoff in ISO format. Default points to Dec 30 13:00 (year can be changed by config)
  wildcardCutoffIso?: string;
  // If true, show the special note about first/second wildcard timing in the tooltip
  showWildcardSeasonSplit?: boolean;
}

interface Props {
  status: TeamStatus;
  config?: StatusStripConfig;
}

// Wildcard season split happens around the end of December. Derive the
// default cutoff from the current FPL season (Aug–May window) instead of a
// hardcoded date so it rolls over automatically each season.
const deriveWildcardCutoffIso = (): string => {
  const now = new Date();
  // Seasons run Aug–May: from July onwards we're in the season starting this year.
  const seasonStartYear = now.getMonth() >= 6 ? now.getFullYear() : now.getFullYear() - 1;
  return `${seasonStartYear}-12-30T13:00:00`;
};

const StatusStrip = ({ status, config }: Props) => {
  const chipNames = {
    wildcard: 'WC',
    freehit: 'FH',
    bboost: 'BB',
    tcaptain: 'TC',
  };

  const chipFull = {
    wildcard: 'Wildcard',
    freehit: 'Free Hit',
    bboost: 'Bench Boost',
    tcaptain: 'Triple Captain',
  } as const;

  const chipDesc: Record<string, string> = {
    wildcard: 'Replace your entire squad for this GW. Useful for fixture swings or large changes.',
    freehit: 'Temporarily replace your squad for one GW; your original squad returns after the GW.',
    bboost: 'Score points from your entire bench for a single GW (useful in double gameweeks).',
    tcaptain: 'Triple Captain: captain scores triple points for one GW (use on a premium double-gameweek).',
  };

  // Default configuration
  const defaultConfig: Required<StatusStripConfig> = {
    tooltipOffsetPx: 12, // how far above the bar the tooltip sits (px)
    wildcardCutoffIso: deriveWildcardCutoffIso(), // default cutoff (changeable via config)
    showWildcardSeasonSplit: true,
  };

  const cfg = { ...defaultConfig, ...(config ?? {}) };

  const deadline = new Date(status.deadline);
  const now = new Date();
  const hoursRemaining = Math.max(0, Math.floor((deadline.getTime() - now.getTime()) / (1000 * 60 * 60)));
  const daysRemaining = Math.floor(hoursRemaining / 24);
  const deadlinePassed = deadline < now;

  // Wildcard cutoff handling — parse the configured cutoff
  let wildcardCutoffDate: Date | null = null;
  try {
    wildcardCutoffDate = cfg.wildcardCutoffIso ? new Date(cfg.wildcardCutoffIso) : null;
    if (wildcardCutoffDate && isNaN(wildcardCutoffDate.getTime())) wildcardCutoffDate = null;
  } catch {
    wildcardCutoffDate = null;
  }

  const isAfterWildcardCutoff = wildcardCutoffDate ? now >= wildcardCutoffDate : false;
  const wildcardCutoffLabel = wildcardCutoffDate
    ? wildcardCutoffDate.toLocaleString(undefined, { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' })
    : cfg.wildcardCutoffIso;

  return (
    <DashboardCard className="px-5 py-4">
      <div className="flex flex-wrap items-center gap-6">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs uppercase tracking-wide text-slate-500">Free Transfers</span>
            <span className="text-base font-bold text-white">{status.freeTransfers}</span>
          </div>
        </div>

        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs uppercase tracking-wide text-slate-500">Bank</span>
            <span className="text-base font-bold text-emerald-400">£{status.bank.toFixed(1)}m</span>
          </div>
        </div>

        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs uppercase tracking-wide text-slate-500">Team Value</span>
            <span className="text-base font-bold text-white">£{status.teamValue.toFixed(1)}m</span>
          </div>
        </div>

        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs uppercase tracking-wide text-slate-500">Chips</span>
            <div className="flex items-center gap-1.5">
              {(Object.keys(chipNames) as Array<keyof typeof chipNames>).map((chipKey) => {
                const chip = status.chips[chipKey];
                const wildcardNote = cfg.showWildcardSeasonSplit && chipKey === 'wildcard'
                  ? isAfterWildcardCutoff
                    ? `After ${wildcardCutoffLabel}: first Wildcard is lost; second Wildcard is available.`
                    : `Second Wildcard becomes available after ${wildcardCutoffLabel}.`
                  : null;

                const tooltip = (
                  <div className="text-white">
                    <div className="flex items-start justify-between gap-2">
                      <span className="font-semibold text-white">{chipFull[chipKey]}</span>
                      <span className={`text-xs ${chip.available ? 'text-emerald-300' : 'text-slate-400'}`}>
                        {chip.available ? 'Available' : 'Used'}
                      </span>
                    </div>
                    <p className="mt-2 text-xs text-slate-300">{chipDesc[chipKey]}</p>
                    <p className="mt-2 text-[11px] text-slate-300">
                      {chip.used ? `Used: ${chip.used}` : chip.available ? 'Can be used this GW' : 'Not available'}
                    </p>
                    {wildcardNote ? (
                      <p className="mt-2 text-[11px] font-medium text-white/92">{wildcardNote}</p>
                    ) : null}
                  </div>
                );

                return (
                  <Tooltip key={chipKey} delayDuration={150}>
                    <TooltipTrigger asChild>
                      <Badge
                        className={`cursor-default rounded-md border px-2 py-1 text-[10px] font-bold normal-case ${
                          chip.available
                            ? 'border-[rgba(16,185,129,0.22)] bg-[rgba(16,185,129,0.12)] text-emerald-400'
                            : 'border-white/8 bg-white/6 text-slate-500'
                        }`}
                      >
                        {chipNames[chipKey]}
                      </Badge>
                    </TooltipTrigger>
                    <TooltipContent side="top" sideOffset={cfg.tooltipOffsetPx} className="border border-white/8 bg-[rgba(15,23,42,0.96)] px-3 py-2">
                      {tooltip}
                    </TooltipContent>
                  </Tooltip>
                );
              })}
            </div>
          </div>
        </div>

        <div className="ms-0 xl:ms-auto">
          <div className="flex items-center gap-2">
            <span className="text-xs uppercase tracking-wide text-slate-500">Deadline</span>
            <span className={`text-base font-bold ${deadlinePassed ? 'text-red-300' : 'text-yellow-300'}`}>
              {deadlinePassed ? 'Passed' : daysRemaining > 0 ? `${daysRemaining}d ${hoursRemaining % 24}h` : `${hoursRemaining}h`}
            </span>
          </div>
        </div>
      </div>
    </DashboardCard>
  );
};

export default StatusStrip;
