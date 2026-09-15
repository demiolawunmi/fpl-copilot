import type { VideoInsight } from '../../data/commandCenterMocks';
import { DashboardCard, DashboardHeader } from '@/components/ui/primitives';

interface Props {
  videos: VideoInsight[];
}

const VideoInsightsCard = ({ videos }: Props) => {
  return (
    <DashboardCard>
      <DashboardHeader title="Gameweek Videos" description="Curated FPL content" />
      <div className="card-scroll flex max-h-80 flex-col gap-3 overflow-y-auto px-5 py-4">
        {videos.map((video, idx) => (
          <div
            key={video.id}
            className={cnBorder(idx === videos.length - 1)}
          >
            <p className="text-sm font-semibold text-white">{video.title}</p>
            <div className="mt-2 flex items-center gap-2 text-xs text-slate-400">
              <span>{video.source}</span>
              <span>•</span>
              <span>{video.duration}</span>
            </div>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {video.tags.map((tag) => (
                <span
                  key={tag}
                  className="rounded-md border border-white/8 bg-white/6 px-2 py-0.5 text-[10px] font-medium text-slate-300"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </DashboardCard>
  );
};

function cnBorder(isLast: boolean): string {
  return isLast ? 'pb-3' : 'border-b border-white/6 pb-3';
}

export default VideoInsightsCard;
