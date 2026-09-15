import { Badge } from '@/components/ui/badge';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { DashboardCard, DashboardHeader } from '@/components/ui/primitives';
import type { Injury } from '../../data/gwOverviewMocks';

interface Props {
  injuries: Injury[];
}

const statusPalette: Record<Injury['status'], { bg: string; color: string }> = {
  Injured: { bg: 'bg-[rgba(248,113,113,0.12)]', color: 'text-red-300' },
  Suspended: { bg: 'bg-[rgba(251,146,60,0.12)]', color: 'text-orange-300' },
  Doubtful: { bg: 'bg-[rgba(250,204,21,0.12)]', color: 'text-yellow-300' },
};

const InjuriesTable = ({ injuries }: Props) => (
  <DashboardCard>
    <DashboardHeader title="Injuries & Suspensions" />
    <div className="w-full overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="text-slate-500">Player</TableHead>
            <TableHead className="text-slate-500">Team</TableHead>
            <TableHead className="text-slate-500">Status</TableHead>
            <TableHead className="text-slate-500">Return</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {injuries.map((inj, i) => {
            const palette = statusPalette[inj.status];
            return (
              <TableRow key={i} className="hover:bg-white/4">
                <TableCell className="font-medium text-white">
                  {inj.player}
                </TableCell>
                <TableCell className="text-slate-400">{inj.team}</TableCell>
                <TableCell>
                  <Badge
                    className={`rounded-full px-2.5 py-0.5 normal-case ${palette.bg} ${palette.color}`}
                  >
                    {inj.status}
                  </Badge>
                </TableCell>
                <TableCell className="text-slate-400">{inj.returnDate}</TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  </DashboardCard>
);

export default InjuriesTable;
