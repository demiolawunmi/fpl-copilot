import { Loader2 } from 'lucide-react';
import { DashboardCard, DashboardHeader } from '@/components/ui/primitives';
import { Button } from '@/components/ui/button';

interface Props {
  onAutoCaptain: () => void;
  onAutoBench: () => void;
  onRollTransfer: () => void;
  /** Opens the optimization dialog (weeks ahead + run). */
  onOpenOptimization: () => void;
  /** True while POST / optimize is in flight (long-running). */
  isOptimizationLoading?: boolean;
}

const QuickActions = ({
  onAutoCaptain,
  onAutoBench,
  onRollTransfer,
  onOpenOptimization,
  isOptimizationLoading = false,
}: Props) => {
  return (
    <DashboardCard>
      <DashboardHeader title="Quick Actions" />
      <div className="flex flex-col gap-3 px-5 py-4">
        <Button onClick={onAutoCaptain} variant="outline" className="justify-start border-white/12 text-slate-200 hover:bg-white/6 hover:text-white">
          ⚡ Auto-pick Captain (Highest xPts)
        </Button>
        <Button onClick={onAutoBench} variant="outline" className="justify-start border-white/12 text-slate-200 hover:bg-white/6 hover:text-white">
          🔄 Auto-pick Bench Order
        </Button>
        <Button
          onClick={onOpenOptimization}
          disabled={isOptimizationLoading}
          variant="outline"
          className="justify-start border-[rgba(59,130,246,0.22)] text-blue-300 hover:bg-[rgba(59,130,246,0.12)] hover:text-blue-200"
        >
          {isOptimizationLoading ? (
            <>
              <Loader2 size={16} className="animate-spin" />
              Running optimization…
            </>
          ) : (
            '🧠 Run AIrsenal Optimization'
          )}
        </Button>
        <Button onClick={onRollTransfer} variant="outline" className="justify-start border-[rgba(16,185,129,0.22)] text-emerald-400 hover:bg-[rgba(16,185,129,0.12)] hover:text-emerald-300">
          💡 Explore Transfers (Go to Sandbox)
        </Button>
      </div>
    </DashboardCard>
  );
};

export default QuickActions;
