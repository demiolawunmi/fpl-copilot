import type { TeamFixtureRatingsRow } from '../../types/fixturesRatings';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { DashboardCard, DashboardHeader } from '@/components/ui/primitives';

function cellBg(difficulty: number | null): string {
  if (difficulty == null) return 'bg-white/4';
  const tier = Math.round(difficulty);
  if (tier <= 2) return 'bg-[rgba(16,185,129,0.18)]';
  if (tier === 3) return 'bg-[rgba(234,179,8,0.16)]';
  return 'bg-[rgba(244,63,94,0.16)]';
}

function formatCellValue(d: number | null, mode: Mode): string {
  if (d == null) return '—';
  if (mode === 'elo') return d.toFixed(1);
  return String(Math.round(d));
}

type Mode = 'official' | 'elo';

interface Props {
  title: string;
  description?: string;
  gameweekIds: number[];
  teams: TeamFixtureRatingsRow[];
  mode: Mode;
}

export default function FdrMatrixTable({
  title,
  description,
  gameweekIds,
  teams,
  mode,
}: Props) {
  const values = (row: TeamFixtureRatingsRow) =>
    mode === 'official' ? row.officialFdr : row.eloBasedFdr;

  return (
    <DashboardCard>
      <DashboardHeader
        title={title}
        description={description}
      />
      <div className="card-scroll overflow-auto">
        <Table
          className="min-w-max"
          style={{ borderCollapse: 'separate', borderSpacing: 0 }}
        >
          <TableHeader>
            <TableRow>
              <TableHead className="sticky left-0 z-[2] border-b border-white/6 bg-slate-800 px-3 py-2.5 text-xs font-semibold tracking-wider uppercase text-slate-400">
                Team
              </TableHead>
              {gameweekIds.map((gw) => (
                <TableHead
                  key={gw}
                  className="border-b border-white/6 px-2 py-2.5 text-center text-xs font-semibold whitespace-nowrap text-slate-400"
                >
                  GW {gw}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {teams.map((row) => (
              <TableRow key={row.shortName}>
                <TableCell className="sticky left-0 z-[1] border-b border-white/6 bg-slate-900 px-3 py-2 font-medium whitespace-nowrap text-slate-200">
                  {row.shortName}
                </TableCell>
                {values(row).map((d, i) => (
                  <TableCell
                    key={`${row.shortName}-${gameweekIds[i]}`}
                    className="border-b border-white/6 px-1.5 py-1.5 text-center"
                  >
                    <div
                      className={`inline-flex min-h-[28px] min-w-[32px] items-center justify-center rounded-md border border-white/6 px-2 ${cellBg(d)}`}
                    >
                      <span
                        className={`text-xs font-bold ${d == null ? 'text-slate-500' : 'text-white'}`}
                      >
                        {formatCellValue(d, mode)}
                      </span>
                    </div>
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </DashboardCard>
  );
}
