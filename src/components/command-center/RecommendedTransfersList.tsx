import { useState } from 'react';
import type { RecommendedTransferItem } from '../../data/commandCenterMocks';
import { DashboardCard, DashboardHeader } from '@/components/ui/primitives';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';

interface Props {
  transfers: RecommendedTransferItem[];
  onApplyTransfer: (playerInId: number, playerOutId: number) => void;
}

const RecommendedTransfersList = ({ transfers, onApplyTransfer }: Props) => {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  const toggleExpand = (index: number) => {
    setExpandedIndex(expandedIndex === index ? null : index);
  };

  return (
    <DashboardCard>
      <DashboardHeader title="Recommended Transfers" description="AI-suggested moves for this gameweek" />
      <div className="card-scroll flex max-h-[24rem] flex-col gap-3 overflow-y-auto px-5 py-4">
        {transfers.length === 0 ? (
          <div className="py-2">
            <span className="text-sm text-slate-400">
              No transfer recommendations available yet. Try applying a model blend to generate suggestions.
            </span>
          </div>
        ) : null}
        {transfers.map((transfer, idx) => (
          <div key={idx} className={`pb-3 ${idx === transfers.length - 1 ? '' : 'border-b border-white/6'}`}>
            <div className="flex items-center justify-between gap-3">
              <div className="flex min-w-0 flex-1 items-center gap-3">
                <div className="flex min-w-0 items-center gap-2">
                  <span className="text-sm font-semibold text-red-300">{transfer.playerOut.name}</span>
                  <span className="text-slate-600">→</span>
                  <span className="text-sm font-semibold text-emerald-400">{transfer.playerIn.name}</span>
                </div>
                <Badge className="rounded-md border border-[rgba(16,185,129,0.22)] bg-[rgba(16,185,129,0.12)] px-2 py-1 text-[10px] normal-case text-emerald-400">
                  +{transfer.xPtsDelta.toFixed(1)} xPts
                </Badge>
              </div>
              <Button
                onClick={() => onApplyTransfer(transfer.playerIn.id, transfer.playerOut.id)}
                variant="outline"
                className="h-6 rounded-md border-[rgba(16,185,129,0.22)] px-2 text-xs text-emerald-400 hover:bg-[rgba(16,185,129,0.12)]"
              >
                Apply
              </Button>
            </div>

            <div className="mt-2 flex flex-wrap items-center gap-4 text-xs text-slate-400">
              <span>OUT: £{transfer.playerOut.price}m • {transfer.playerOut.team}</span>
              <span>IN: £{transfer.playerIn.price}m • {transfer.playerIn.team}</span>
            </div>

            <Button
              onClick={() => toggleExpand(idx)}
              variant="link"
              className="mt-2 h-6 px-0 text-xs text-emerald-400 hover:text-emerald-300"
            >
              {expandedIndex === idx ? '▼ Hide rationale' : '▶ Why this?'}
            </Button>

            {expandedIndex === idx ? (
              <div className="mt-2 border-l-2 border-white/8 pl-4">
                <span className="text-xs leading-relaxed text-slate-300">{transfer.why}</span>
              </div>
            ) : null}
          </div>
        ))}
      </div>
    </DashboardCard>
  );
};

export default RecommendedTransfersList;
