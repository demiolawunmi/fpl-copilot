import type { EnhancedPlayer } from '../../data/commandCenterMocks';
import { DashboardCard } from '@/components/ui/primitives';
import { Badge } from '@/components/ui/badge';

interface Props {
  realSquad: EnhancedPlayer[];
  sandboxSquad: EnhancedPlayer[];
  bank: number;
  bankDelta: number;
  freeTransfers: number;
  sandboxTransfersMade: number;
}

const DeltaStrip = ({ realSquad, sandboxSquad, bank, bankDelta, freeTransfers, sandboxTransfersMade }: Props) => {
  const realXPts = realSquad.filter((p) => !p.isBench).reduce((sum, p) => sum + p.xPts, 0);
  const sandboxXPts = sandboxSquad.filter((p) => !p.isBench).reduce((sum, p) => sum + p.xPts, 0);
  const xPtsDelta = sandboxXPts - realXPts;

  const realNext5 = realXPts * 5;
  const sandboxNext5 = sandboxXPts * 5;
  const next5Delta = sandboxNext5 - realNext5;

  const currentBank = Number((bank + bankDelta).toFixed(1));
  const hasChanges = Math.abs(xPtsDelta) > 0.01 || Math.abs(next5Delta) > 0.01 || Math.abs(bankDelta) > 0.001 || sandboxTransfersMade > 0;

  const extraTransfers = Math.max(0, sandboxTransfersMade - freeTransfers);
  const hitCost = extraTransfers * 4;

  const deltaColor = (value: number) => (value > 0 ? 'text-emerald-400' : value < 0 ? 'text-red-300' : 'text-slate-400');

  return (
    <DashboardCard className="px-5 py-4">
      <div className="flex flex-wrap items-center gap-6">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs uppercase tracking-wide text-slate-500">
              GW xPts
            </span>
            <span className="text-sm text-slate-400">
              {realXPts.toFixed(1)}
            </span>
            <span className="text-slate-600">→</span>
            <span className="text-sm font-bold text-white">
              {sandboxXPts.toFixed(1)}
            </span>
            {Math.abs(xPtsDelta) > 0.01 ? (
              <span className={`text-xs font-bold ${deltaColor(xPtsDelta)}`}>
                {xPtsDelta > 0 ? '+' : ''}
                {xPtsDelta.toFixed(1)}
              </span>
            ) : null}
          </div>
        </div>

        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs uppercase tracking-wide text-slate-500">
              Next 5 GWs
            </span>
            <span className="text-sm text-slate-400">
              {realNext5.toFixed(1)}
            </span>
            <span className="text-slate-600">→</span>
            <span className="text-sm font-bold text-white">
              {sandboxNext5.toFixed(1)}
            </span>
            {Math.abs(next5Delta) > 0.01 ? (
              <span className={`text-xs font-bold ${deltaColor(next5Delta)}`}>
                {next5Delta > 0 ? '+' : ''}
                {next5Delta.toFixed(1)}
              </span>
            ) : null}
          </div>
        </div>

        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs uppercase tracking-wide text-slate-500">
              Bank
            </span>
            <span className={`text-sm font-bold ${currentBank < 0 ? 'text-red-300' : 'text-emerald-400'}`}>
              £{currentBank.toFixed(1)}m
              {Math.abs(bankDelta) > 0.001 ? (
                <span className={`ml-1 text-xs ${deltaColor(bankDelta)}`}>
                  ({bankDelta > 0 ? '+' : ''}
                  {bankDelta.toFixed(1)}m)
                </span>
              ) : null}
            </span>
          </div>
        </div>

        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs uppercase tracking-wide text-slate-500">
              Transfers
            </span>
            <span className={`text-sm font-bold ${sandboxTransfersMade > freeTransfers ? 'text-red-300' : 'text-emerald-400'}`}>
              {sandboxTransfersMade}/{freeTransfers}
            </span>
            {hitCost > 0 && (
              <Badge className="rounded-md bg-red-500/20 px-2 py-0.5 text-[10px] text-red-300">
                −{hitCost} pts hit
              </Badge>
            )}
          </div>
        </div>

        {!hasChanges ? (
          <div className="ms-0 xl:ms-auto">
            <span className="text-xs italic text-slate-500">
              No changes yet
            </span>
          </div>
        ) : null}
      </div>
    </DashboardCard>
  );
};

export default DeltaStrip;
