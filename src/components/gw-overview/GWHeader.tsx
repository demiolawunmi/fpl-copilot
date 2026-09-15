import { Button } from '@/components/ui/button';
import { FiChevronLeft, FiChevronRight } from 'react-icons/fi';
import type { GWInfo } from '../../data/gwOverviewMocks';

interface Props {
  info: GWInfo;
  onPrev?: () => void;
  onNext?: () => void;
  disablePrev?: boolean;
  disableNext?: boolean;
}

const GWHeader = ({ info, onPrev, onNext, disablePrev, disableNext }: Props) => (
  <div className="flex items-center justify-between gap-4">
    <Button
      aria-label="Previous gameweek"
      onClick={onPrev}
      disabled={disablePrev}
      variant="ghost"
      size="icon"
      className={`bg-white/6 hover:bg-white/8 ${disablePrev ? 'text-slate-500' : 'text-white'}`}
    >
      <FiChevronLeft size={20} />
    </Button>

    <div className="flex-1 text-center">
      <h2 className="text-2xl font-bold leading-[1.33]">Gameweek {info.gameweek}</h2>
      <p className="mt-1 text-sm text-slate-400">
        {info.teamName} · {info.manager} · ID {info.teamId}
      </p>
    </div>

    <Button
      aria-label="Next gameweek"
      onClick={onNext}
      disabled={disableNext}
      variant="ghost"
      size="icon"
      className={`bg-white/6 hover:bg-white/8 ${disableNext ? 'text-slate-500' : 'text-white'}`}
    >
      <FiChevronRight size={20} />
    </Button>
  </div>
);

export default GWHeader;
