import { createContext, useContext, type ReactNode } from 'react';
import { useCoreData, type CoreData } from '../hooks/useCoreData';
import { useTeamId } from './TeamIdContext';

const CoreContext = createContext<CoreData | null>(null);

export function CoreProvider({ children }: { children: ReactNode }) {
  const { teamId } = useTeamId();
  const data = useCoreData(teamId);
  return <CoreContext.Provider value={data}>{children}</CoreContext.Provider>;
}

export function useCore(): CoreData {
  const ctx = useContext(CoreContext);
  if (!ctx) throw new Error('useCore must be used within CoreProvider');
  return ctx;
}
