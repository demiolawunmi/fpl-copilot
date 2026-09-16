import { createContext, useContext, useState, useCallback, type ReactNode } from 'react';

interface TeamIdContextType {
  teamId: string | null;
  setTeamId: (id: string) => void;
  clearTeamId: () => void;
}

const TeamIdContext = createContext<TeamIdContextType | undefined>(undefined);

export const TeamIdProvider = ({ children }: { children: ReactNode }) => {
  const [teamId, setTeamIdState] = useState<string | null>(
    () => localStorage.getItem('teamId')
  );

  const setTeamId = useCallback((id: string) => {
    localStorage.setItem('teamId', id);
    setTeamIdState(id);
  }, []);

  const clearTeamId = useCallback(() => {
    localStorage.removeItem('teamId');
    setTeamIdState(null);
  }, []);

  return (
    <TeamIdContext.Provider value={{ teamId, setTeamId, clearTeamId }}>
      {children}
    </TeamIdContext.Provider>
  );
};

export const useTeamId = (): TeamIdContextType => {
  const ctx = useContext(TeamIdContext);
  if (!ctx) throw new Error('useTeamId must be used within TeamIdProvider');
  return ctx;
};

