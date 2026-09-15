import { useState } from 'react';
import { cn } from '@/lib/utils';
import { fplEndpoints, type PlayerPhotoSize } from '../../api/fpl/endpoints';

type Size = 'sm' | 'md' | 'lg' | 'hero';

/** All sizes use 110x140 CDN path (confirmed working); only the rendered box size changes. */
const SIZE_MAP: Record<Size, { box: string; photoSize: PlayerPhotoSize }> = {
  sm: { box: 'size-7', photoSize: '110x140' },
  md: { box: 'size-9', photoSize: '110x140' },
  lg: { box: 'size-14', photoSize: '110x140' },
  hero: { box: 'size-24', photoSize: '110x140' },
};

interface PlayerHeadshotProps {
  code: number;
  name: string;
  size?: Size;
}

const PlayerHeadshot = ({ code, name, size = 'md' }: PlayerHeadshotProps) => {
  const [failed, setFailed] = useState(false);
  const cfg = SIZE_MAP[size];

  if (!code) {
    return (
      <div
        className={cn(
          'flex shrink-0 items-center justify-center rounded-full bg-slate-700 font-bold text-white uppercase',
          cfg.box,
          'text-[10px]',
        )}
      >
        {name.slice(0, 3)}
      </div>
    );
  }

  return (
    <div className={cn('shrink-0 overflow-hidden rounded-full bg-[rgba(51,65,85,0.4)]', cfg.box)}>
      <img
        src={failed ? fplEndpoints.playerPlaceholder(cfg.photoSize) : fplEndpoints.playerPhoto(code, cfg.photoSize)}
        alt={name}
        className="h-full w-full object-cover object-top"
        loading="lazy"
        onError={() => setFailed(true)}
      />
    </div>
  );
};

export default PlayerHeadshot;
