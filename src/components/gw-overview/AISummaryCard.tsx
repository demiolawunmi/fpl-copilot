import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { DashboardCard, DashboardHeader } from "@/components/ui/primitives";
import type { AISummary, AISummaryTone, GWInfo } from "../../data/gwOverviewMocks";

type Props = {
    gwInfo: GWInfo;
    summary: AISummary;
};

export default function AiSummaryCard({ gwInfo, summary }: Props) {
    return (
        <DashboardCard as="section">
            <DashboardHeader
                title={summary.heading}
                description={`GW ${gwInfo.gameweek} • ${gwInfo.teamName} • ${gwInfo.manager} • ${gwInfo.teamId}`}
                action={
                    <Badge className="rounded-full border border-white/8 bg-white/6 px-2 py-1 normal-case text-slate-300">
                        MVP
                    </Badge>
                }
            />

            <div className="px-5 py-4">
                <p className="text-sm leading-[1.67] text-slate-300">
                    {summary.intro}
                </p>

                <ul className="mt-4 flex flex-col gap-3">
                    {summary.items.map((item, idx) => (
                        <li key={idx} className="flex items-start gap-3">
                            <Dot tone={item.tone} />
                            <p className="text-sm leading-tight text-slate-200">
                                {item.text}
                            </p>
                        </li>
                    ))}
                </ul>

                <div className="mt-5 flex items-center justify-between gap-3">
                    <p className="text-xs text-slate-500">
                        {summary.footerHint ?? "More detail coming soon."}
                    </p>
                    <Button size="sm" variant="outline" disabled className="border-white/8 text-slate-500">
                        Refresh
                    </Button>
                </div>
            </div>
        </DashboardCard>
    );
}

function Dot({ tone }: { tone: AISummaryTone }) {
    const bg = tone === "good" ? "bg-emerald-400" : tone === "warn" ? "bg-yellow-400" : "bg-sky-400";
    return <div className={`mt-2 size-2 shrink-0 rounded-full ${bg}`} />;
}
