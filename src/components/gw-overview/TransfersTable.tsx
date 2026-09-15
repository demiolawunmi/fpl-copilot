import type { Transfer } from '../../data/gwOverviewMocks';
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
  transfers: Transfer[];
}

const TransfersTable = ({ transfers }: Props) => (
  <DashboardCard>
    <DashboardHeader title="Transfers" />
    <div className="w-full overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="text-slate-500">In</TableHead>
            <TableHead className="text-slate-500">Out</TableHead>
            <TableHead className="text-slate-500">Cost</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {transfers.map((t, i) => (
            <TableRow key={i} className="hover:bg-white/4">
              <TableCell className="font-medium text-emerald-400">{t.playerIn}</TableCell>
              <TableCell className="font-medium text-red-300">{t.playerOut}</TableCell>
              <TableCell className="text-slate-400">{t.cost}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  </DashboardCard>
);

export default TransfersTable;
