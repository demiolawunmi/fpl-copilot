import type { VideoInsight } from '../../data/commandCenterMocks';
import { DashboardCard, DashboardHeader } from '@/components/ui/primitives';

interface Props {
  videos: VideoInsight[];
}

const VideoInsightsStrip = ({ videos }: Props) => {
  return (
    <DashboardCard>
      <DashboardHeader title="Gameweek Videos" description="Curated FPL content" />
      <div className="px-5 py-4">
        <div className="card-scroll flex items-stretch gap-4 overflow-x-auto pb-2">
          {videos.map((video) => (
            <div key={video.id} className="min-w-72 max-w-72 shrink-0">
              <div className="flex h-full flex-col gap-2 rounded-xl border border-white/8 bg-[rgba(30,41,59,0.4)] p-4">
                <p className="line-clamp-2 text-sm font-semibold text-white">{video.title}</p>
                <div className="flex items-center gap-2 text-xs text-slate-400">
                  <span>{video.source}</span>
                  <span>•</span>
                  <span>{video.duration}</span>
                </div>
                <div className="mt-auto flex flex-wrap gap-1.5">
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
            </div>
          ))}
        </div>
      </div>
    </DashboardCard>
  );
};

export default VideoInsightsStrip;
