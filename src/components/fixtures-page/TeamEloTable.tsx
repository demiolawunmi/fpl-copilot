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

interface Props {
  teams: TeamFixtureRatingsRow[];
}

export default function TeamEloTable({ teams }: Props) {
  return (
    <DashboardCard>
      <DashboardHeader
        title="Team ratings"
        description="ClubElo from /api/fdr/elo; mean Copilot FDR across the matrix window from /api/fdr/team."
      />
      <div className="card-scroll max-h-[280px] overflow-auto md:max-h-none">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="text-xs font-semibold tracking-wider uppercase text-slate-500">
                Team
              </TableHead>
              <TableHead className="text-right text-xs font-semibold tracking-wider uppercase text-slate-500">
                Elo
              </TableHead>
              <TableHead className="text-right text-xs font-semibold tracking-wider uppercase text-slate-500">
                Custom FDR (Elo)
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {teams.map((row) => {
              const vals = row.eloBasedFdr.filter((x) => x != null) as number[];
              const fallbackAvg =
                vals.length > 0
                  ? vals.reduce((a, b) => a + b, 0) / vals.length
                  : null;
              const customDisplay =
                row.eloFdrSummary != null ? row.eloFdrSummary : fallbackAvg;

              return (
                <TableRow key={row.shortName} className="hover:bg-white/4">
                  <TableCell className="font-medium text-slate-200">
                    {row.shortName}
                  </TableCell>
                  <TableCell
                    className={`text-right font-mono ${row.elo == null ? 'text-slate-500' : 'text-white'}`}
                  >
                    {row.elo == null ? '—' : row.elo.toFixed(0)}
                  </TableCell>
                  <TableCell
                    className={`text-right font-mono ${customDisplay == null ? 'text-slate-500' : 'text-slate-200'}`}
                  >
                    {customDisplay == null ? (
                      <span className="text-slate-500">—</span>
                    ) : (
                      customDisplay.toFixed(2)
                    )}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
    </DashboardCard>
  );
}
