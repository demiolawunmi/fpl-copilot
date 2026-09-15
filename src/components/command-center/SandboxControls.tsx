import { DashboardCard } from '@/components/ui/primitives';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';

interface Props {
  sandboxMode: boolean;
  onToggleSandboxMode: () => void;
  onUndo: () => void;
  onReset: () => void;
  onApply: () => void;
  canUndo: boolean;
}

const SandboxControls = ({
  sandboxMode,
  onToggleSandboxMode,
  onUndo,
  onReset,
  onApply,
  canUndo,
}: Props) => {
  return (
    <DashboardCard className="px-5 py-4">
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-3">
          <span className="text-sm text-slate-400">Sandbox Mode</span>
          <Switch checked={sandboxMode} onCheckedChange={onToggleSandboxMode} />
        </div>

        <div className="hidden h-6 w-px bg-white/8 md:block" />

        <Button
          onClick={onUndo}
          disabled={!canUndo}
          variant="outline"
          size="sm"
          className={`border-white/8 ${canUndo ? 'text-slate-200 hover:bg-white/6 hover:text-white' : 'text-slate-600'}`}
        >
          ↶ Undo
        </Button>

        <Button
          onClick={onReset}
          variant="outline"
          size="sm"
          className="border-white/8 text-slate-200 hover:bg-white/6 hover:text-white"
        >
          ⟲ Reset
        </Button>

        <Button
          onClick={onApply}
          variant="outline"
          size="sm"
          className="border-[rgba(16,185,129,0.22)] text-emerald-400 hover:bg-[rgba(16,185,129,0.12)] hover:text-emerald-300"
        >
          ✓ Apply to Team
        </Button>
      </div>
    </DashboardCard>
  );
};

export default SandboxControls;
