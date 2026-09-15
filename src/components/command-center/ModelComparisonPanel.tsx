import { Loader2 } from 'lucide-react';
import type { ModelSource } from '../../data/commandCenterMocks';
import { DashboardCard, DashboardHeader } from '@/components/ui/primitives';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';

type ApplyStatus = 'idle' | 'submitting' | 'queued' | 'running' | 'completed' | 'failed';

interface Props {
  models: ModelSource[];
  blendTotal: number;
  blendRemaining: number;
  isBlendInvalid: boolean;
  onModelWeightChange: (modelId: string, nextWeight: number) => void;
  applyStatus: ApplyStatus;
  statusMessage?: React.ReactNode;
  canRetry?: boolean;
  onApply: () => void;
}

const getStatusTone = (applyStatus: ApplyStatus) => {
  if (applyStatus === 'completed') return { classes: 'bg-green-900 text-green-200', label: 'Completed' };
  if (applyStatus === 'failed') return { classes: 'bg-red-900 text-red-200', label: 'Failed' };
  if (applyStatus === 'running') return { classes: 'bg-blue-900 text-blue-200', label: 'Running' };
  if (applyStatus === 'queued' || applyStatus === 'submitting') return { classes: 'bg-orange-900 text-orange-200', label: 'Pending' };
  return { classes: 'bg-white/6 text-slate-300', label: 'Idle' };
};

const ModelComparisonPanel = ({
  models,
  blendTotal,
  blendRemaining,
  isBlendInvalid,
  onModelWeightChange,
  applyStatus,
  statusMessage,
  canRetry,
  onApply,
}: Props) => {
  const tone = getStatusTone(applyStatus);
  const isBusy = applyStatus === 'submitting' || applyStatus === 'queued' || applyStatus === 'running';
  const buttonLabel = applyStatus === 'failed' && canRetry ? 'Retry Apply' : 'Apply Blend';

  return (
    <DashboardCard>
      <DashboardHeader
        title="Model Sources"
        description="Data blending & predictions"
        action={(
          <Button
            size="sm"
            onClick={onApply}
            disabled={isBlendInvalid || isBusy}
            className={`${applyStatus === 'failed' ? 'bg-red-500 hover:bg-red-600' : 'bg-blue-500 hover:bg-blue-600'} disabled:cursor-not-allowed disabled:opacity-45`}
          >
            {isBusy && <Loader2 size={16} className="animate-spin" />}
            {isBusy ? (applyStatus === 'submitting' ? 'Submitting' : 'Applying') : buttonLabel}
          </Button>
        )}
      />
      <div className="px-5 py-4">
        <div className="flex flex-col gap-3">
          {models.map((model) => (
            <div key={model.id} className="rounded-lg border border-white/8 bg-white/4 px-3 py-2">
              <div className="flex flex-col gap-2">
                <div className="flex items-center justify-between gap-3">
                  <span className="text-sm text-slate-200">{model.name}</span>
                  <Badge className="rounded-md border border-white/8 bg-white/6 px-2 py-1 normal-case text-emerald-400">
                    {model.weight}%
                  </Badge>
                </div>
                <Slider
                  value={[model.weight]}
                  min={0}
                  max={100}
                  step={1}
                  onValueChange={(nextValue) => onModelWeightChange(model.id, nextValue[0])}
                />
              </div>
            </div>
          ))}

          <div className={`rounded-lg border px-3 py-2 ${isBlendInvalid ? 'border-red-300 bg-red-900' : 'border-white/8 bg-white/4'}`}>
            <div className="flex flex-col gap-1">
              <p className={`text-xs ${isBlendInvalid ? 'text-red-200' : 'text-slate-400'}`}>
                Blend Total: {blendTotal}%
              </p>
              <p className={`text-xs ${isBlendInvalid ? 'text-red-200' : 'text-slate-400'}`}>
                Remaining: {Math.max(0, blendRemaining)}%
              </p>
              {isBlendInvalid ? (
                <div className="rounded-md border border-red-300 bg-[rgba(127,29,29,0.65)] px-2 py-1.5">
                  <p className="text-sm font-semibold text-red-100">
                    Total ratio cannot exceed 100%.
                  </p>
                  <p className="text-xs text-red-200">
                    Apply Blend is disabled until one or more source weights are reduced.
                  </p>
                </div>
              ) : null}
            </div>
          </div>

          <div className="rounded-lg border border-white/8 bg-white/4 px-3 py-2">
            <div className="flex flex-col gap-1">
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs text-slate-400">Apply Status</span>
                <Badge className={`rounded-md border border-white/8 px-2 py-0.5 normal-case ${tone.classes}`}>
                  {tone.label}
                </Badge>
              </div>
              <div className="text-sm">
                {statusMessage ?? (
                  <p className="text-slate-200">
                    Apply to submit blend job and refresh model output.
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </DashboardCard>
  );
};

export default ModelComparisonPanel;
