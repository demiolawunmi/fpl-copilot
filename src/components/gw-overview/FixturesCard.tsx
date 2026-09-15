import { useEffect, useMemo, useState } from "react";
import { FiChevronRight } from "react-icons/fi";
import type { Fixture } from "../../data/gwOverviewMocks";
import { Button } from "@/components/ui/button";
import { DashboardCard, DashboardHeader } from "@/components/ui/primitives";

interface Props {
  fixtures: Fixture[];
  isCurrentGw?: boolean; // true => newest→oldest by default
  heightPx?: number; // computed height to match pitch card bottom
}

type Order = "newest" | "oldest";

/* ── team badge (real image or fallback colored circle) ── */
const Badge = ({ abbr, color, badge }: { abbr: string; color: string; badge?: string }) => {
  const [failed, setFailed] = useState(false);
  if (badge && !failed) {
    return (
      <img
        src={badge}
        alt={abbr}
        className="size-8 object-contain"
        loading="lazy"
        onError={() => setFailed(true)}
      />
    );
  }
  return (
    <div
      className="flex size-8 shrink-0 items-center justify-center rounded-full text-[10px] font-bold text-white"
      style={{ backgroundColor: color }}
    >
      {abbr}
    </div>
  );
};

function toTime(f: Fixture) {
  // Prefer dateISO if you add it; fallback to Date.parse(date) if not.
  // (But Date.parse("Sat 15 Feb") is not reliable across browsers — add dateISO!)
  return f.dateISO ? Date.parse(f.dateISO) : Date.parse(f.date);
}

export default function FixturesCard({ fixtures, isCurrentGw = true, heightPx }: Props) {
  // Default behavior:
  // - current GW: newest → oldest
  // - old GW: oldest → newest
  const defaultOrder: Order = isCurrentGw ? "newest" : "oldest";
  const [order, setOrder] = useState<Order>(defaultOrder);

  // If isCurrentGw changes (user switches GW), reset to the new default
  // (Optional) If you DON'T want it to reset, remove this memo/logic.
  useEffect(() => {
    setOrder(defaultOrder);
  }, [defaultOrder]);

  const { groupKeys, grouped } = useMemo(() => {
    // group by display date
    const grouped = fixtures.reduce<Record<string, Fixture[]>>((acc, f) => {
      (acc[f.date] ??= []).push(f);
      return acc;
    }, {});

    // compute a representative timestamp per date group (min or max)
    const keyToTime = new Map<string, number>();
    for (const [date, arr] of Object.entries(grouped)) {
      const times = arr.map(toTime).filter((t) => Number.isFinite(t));
      // pick earliest time in group as group anchor
      keyToTime.set(date, times.length ? Math.min(...times) : 0);
    }

    const groupKeys = Object.keys(grouped).sort((a, b) => {
      const ta = keyToTime.get(a) ?? 0;
      const tb = keyToTime.get(b) ?? 0;
      return order === "newest" ? tb - ta : ta - tb;
    });

    // Also sort matches inside each date by kickoff time
    for (const k of Object.keys(grouped)) {
      grouped[k].sort((a, b) => {
        const ta = toTime(a);
        const tb = toTime(b);
        return order === "newest" ? tb - ta : ta - tb;
      });
    }

    return { groupKeys, grouped };
  }, [fixtures, order]);

  return (
    <DashboardCard
      className="flex flex-col"
      style={{ height: heightPx ? `${heightPx}px` : "520px" }}
    >
      <DashboardHeader
        title="Fixtures"
        action={
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500">Order</span>
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="h-6 border-white/8 px-2 text-xs text-slate-200 hover:bg-white/6"
              onClick={() => setOrder((o) => (o === "newest" ? "oldest" : "newest"))}
            >
              {order === "newest" ? "Newest → Oldest" : "Oldest → Newest"}
            </Button>
          </div>
        }
      />

      <div className="card-scroll flex-1 overflow-auto">
        {groupKeys.map((date) => {
          const matches = grouped[date];
          return (
            <div key={date}>
              {/* date header */}
              <div className="bg-gradient-to-r from-slate-800 to-slate-900 px-5 py-2">
                <p className="text-xs font-semibold text-slate-400">{date}</p>
              </div>

              {matches.map((m, i) => (
                <div
                  key={`${date}-${i}`}
                  className="flex items-center gap-3 border-b border-white/6 px-5 py-3 hover:bg-white/4"
                >
                  {/* home */}
                  <Badge abbr={m.homeAbbr} color={m.homeColor} badge={m.homeBadge} />
                  <span className="line-clamp-1 w-20 text-right text-sm text-slate-300">
                    {m.homeTeam}
                  </span>

                  {/* score */}
                  <div className="mx-2 min-w-[56px] rounded-lg bg-white/6 px-3 py-1 text-center">
                    <span className="text-sm font-bold text-white">
                      {m.homeScore} – {m.awayScore}
                    </span>
                  </div>

                  {/* away */}
                  <span className="line-clamp-1 w-20 text-sm text-slate-300">
                    {m.awayTeam}
                  </span>
                  <Badge abbr={m.awayAbbr} color={m.awayColor} badge={m.awayBadge} />

                  {/* chevron */}
                  <FiChevronRight size={16} className="ml-auto text-slate-600" />
                </div>
              ))}
            </div>
          );
        })}
      </div>
    </DashboardCard>
  );
}
