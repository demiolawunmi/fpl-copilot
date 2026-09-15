import { useState } from 'react';
import type { CommandCenterAISummary } from '../../data/commandCenterMocks';
import { DashboardCard, DashboardHeader } from '@/components/ui/primitives';
import { cn } from '@/lib/utils';

interface Props {
  summary: CommandCenterAISummary;
  onRefresh?: () => void;
  isRefreshing?: boolean;
  disableRefresh?: boolean;
}

const AICommandSummary = ({ summary, onRefresh, isRefreshing = false, disableRefresh = false }: Props) => {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  const toggleExpand = (index: number) => {
    setExpandedIndex(expandedIndex === index ? null : index);
  };

  return (
    <DashboardCard>
      <DashboardHeader
        title={summary.title}
        description={`AI-powered insights for GW ${summary.gameweek}`}
        action={
          <button
            type="button"
            onClick={onRefresh}
            disabled={disableRefresh || !onRefresh || isRefreshing}
            className={cn(
              'inline-flex h-6 items-center gap-2 rounded-lg border border-white/8 px-2 text-xs font-semibold text-slate-300 transition-colors',
              'hover:bg-white/6 hover:text-white',
              (disableRefresh || !onRefresh || isRefreshing) && 'cursor-not-allowed opacity-45',
            )}
          >
            {isRefreshing ? (
              <>
                <svg
                  className="size-3.5 animate-spin"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="3"
                >
                  <circle cx="12" cy="12" r="10" strokeOpacity="0.25" />
                  <path d="M22 12a10 10 0 0 1-10 10" strokeLinecap="round" />
                </svg>
                Refreshing
              </>
            ) : (
              'Refresh'
            )}
          </button>
        }
      />

      <div className="px-5 py-4">
        <ul className="flex flex-col gap-4">
          {summary.bullets.map((bullet, idx) => (
            <li key={idx}>
              <div className="flex flex-col gap-2">
                <div className="flex items-start gap-3">
                  <Dot tone={bullet.tone} />
                  <div className="min-w-0 flex-1">
                    <p className="text-sm leading-[1.45] text-slate-200">{bullet.text}</p>
                    <button
                      type="button"
                      className="mt-1 cursor-pointer text-xs font-semibold text-emerald-400 hover:text-emerald-300"
                      onClick={() => toggleExpand(idx)}
                    >
                      {expandedIndex === idx ? '▼ Hide details' : '▶ Why?'}
                    </button>
                  </div>
                </div>

                {expandedIndex === idx ? (
                  <div className="ml-8 border-l-2 border-white/8 pl-4">
                    <p className="text-xs leading-[1.7] text-slate-400">{bullet.why}</p>
                  </div>
                ) : null}
              </div>
            </li>
          ))}
        </ul>
      </div>
    </DashboardCard>
  );
};

function Dot({ tone }: { tone: 'good' | 'info' | 'warn' }) {
  const bgClass = tone === 'good' ? 'bg-emerald-400' : tone === 'warn' ? 'bg-yellow-400' : 'bg-sky-400';
  return <span className={cn('mt-2 size-2 shrink-0 rounded-full', bgClass)} />;
}

export default AICommandSummary;
