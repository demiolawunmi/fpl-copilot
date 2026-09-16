import {
  createContext,
  useContext,
  useState,
  type ReactNode,
} from 'react';
import { useCore } from './CoreContext';
import { futureXpts, xiXpts } from '../domain/projection';
import type { Player } from '../domain/types';

export interface TransferRecord {
  outId: number;
  inId: number;
  outName: string;
  inName: string;
  gain: number;
}

interface Snapshot {
  xi: number[];
  bench: number[];
  captainId: number | null;
  viceId: number | null;
  transfers: TransferRecord[];
}

interface RealSquad {
  xi: number[];
  bench: number[];
  captainId: number | null;
  viceId: number | null;
}

export interface SandboxState extends RealSquad {
  initialized: boolean;
  on: boolean;
  transfers: TransferRecord[];
  history: Snapshot[];
  outId: number | null;
  inId: number | null;
}

export interface SandboxDelta {
  count: number;
  free: number;
  hits: number;
  hitCost: number;
  realGw: number;
  sandGw: number;
  gwDelta: number;
  real5: number;
  sand5: number;
  d5: number;
  bank: number;
  bankDelta: number;
}

interface SquadContextValue extends RealSquad {
  sandbox: SandboxState;
  byId: Map<number, Player>;
  setCaptain: (id: number) => void;
  setVice: (id: number) => void;
  swap: (aId: number, bId: number) => void;
  quickCaptain: () => void;
  quickBench: () => void;
  autoPickXi: () => void;
  toggleSandbox: () => void;
  resetSandbox: () => void;
  undo: () => void;
  applyToTeam: () => void;
  applySandboxTransfer: () => boolean;
  applyRecommendation: (outId: number, inId: number) => boolean;
  pickOut: (id: number) => void;
  pickIn: (id: number) => void;
  clearPair: () => void;
  sandboxDelta: () => SandboxDelta;
}

const SquadContext = createContext<SquadContextValue | null>(null);

const EMPTY_SANDBOX: SandboxState = {
  initialized: false,
  on: false,
  xi: [],
  bench: [],
  captainId: null,
  viceId: null,
  transfers: [],
  history: [],
  outId: null,
  inId: null,
};

function replaceIn(list: number[], outId: number, inId: number): number[] {
  const idx = list.indexOf(outId);
  if (idx === -1) return list;
  const copy = list.slice();
  copy[idx] = inId;
  return copy;
}

function initFromReal(s: SandboxState, real: RealSquad): SandboxState {
  return {
    ...s,
    initialized: true,
    xi: real.xi.slice(),
    bench: real.bench.slice(),
    captainId: real.captainId,
    viceId: real.viceId,
  };
}

export function SquadProvider({ children }: { children: ReactNode }) {
  const core = useCore();
  const [overrides, setOverrides] = useState<RealSquad | null>(null);
  const [sandboxState, setSandbox] = useState<SandboxState>(EMPTY_SANDBOX);

  const real: RealSquad = overrides ?? {
    xi: core.xi,
    bench: core.bench,
    captainId: core.captainId,
    viceId: core.viceId,
  };

  const sandbox: SandboxState = sandboxState.initialized
    ? sandboxState
    : {
        ...sandboxState,
        xi: real.xi,
        bench: real.bench,
        captainId: real.captainId,
        viceId: real.viceId,
      };

  const name = (id: number) => core.playersById.get(id)?.name ?? `#${id}`;
  const gainOf = (outId: number, inId: number) =>
    Math.round(((core.playersById.get(inId)?.xpts ?? 0) - (core.playersById.get(outId)?.xpts ?? 0)) * 10) / 10;

  const snap = (s: SandboxState): SandboxState => ({
    ...s,
    history: [
      ...s.history.slice(-29),
      {
        xi: s.xi.slice(),
        bench: s.bench.slice(),
        captainId: s.captainId,
        viceId: s.viceId,
        transfers: s.transfers.slice(),
      },
    ],
  });

  const setCaptain = (id: number) => {
    setOverrides((o) => {
      const base = o ?? real;
      const vice = base.viceId === id ? base.captainId : base.viceId;
      return { ...base, captainId: id, viceId: vice };
    });
    setSandbox((s) => {
      if (!s.on) return s;
      const base = s.initialized ? s : initFromReal(s, real);
      const vice = base.viceId === id ? base.captainId : base.viceId;
      return { ...base, captainId: id, viceId: vice };
    });
  };

  const setVice = (id: number) => {
    setOverrides((o) => {
      const base = o ?? real;
      const cap = base.captainId === id ? base.viceId : base.captainId;
      return { ...base, captainId: cap, viceId: id };
    });
    setSandbox((s) => {
      if (!s.on) return s;
      const base = s.initialized ? s : initFromReal(s, real);
      const cap = base.captainId === id ? base.viceId : base.captainId;
      return { ...base, captainId: cap, viceId: id };
    });
  };

  const applySwap = (s: RealSquad, aId: number, bId: number): RealSquad => {
    const inXi = s.xi.indexOf(aId) > -1;
    const xi = s.xi.slice();
    const bench = s.bench.slice();
    if (inXi) {
      xi[xi.indexOf(aId)] = bId;
      bench[bench.indexOf(bId)] = aId;
    } else {
      bench[bench.indexOf(aId)] = bId;
      xi[xi.indexOf(bId)] = aId;
    }
    return { ...s, xi, bench };
  };

  const swap = (aId: number, bId: number) => {
    setOverrides((o) => applySwap(o ?? real, aId, bId));
    setSandbox((s) => {
      if (!s.on) return s;
      const base = s.initialized ? s : initFromReal(s, real);
      return snap(applySwap(base, aId, bId) as SandboxState);
    });
  };

  const quickCaptain = () => {
    const pool = sandbox.on ? sandbox.xi : real.xi;
    const best = pool
      .slice()
      .sort((a, b) => (core.playersById.get(b)?.xpts ?? 0) - (core.playersById.get(a)?.xpts ?? 0))[0];
    if (best != null) setCaptain(best);
  };

  const quickBench = () => {
    const sort = (list: number[]) =>
      list.slice().sort((a, b) => (core.playersById.get(b)?.xpts ?? 0) - (core.playersById.get(a)?.xpts ?? 0));
    setOverrides((o) => {
      const base = o ?? real;
      return { ...base, bench: sort(base.bench) };
    });
    setSandbox((s) => {
      if (!s.on) return s;
      const base = s.initialized ? s : initFromReal(s, real);
      return { ...base, bench: sort(base.bench) };
    });
  };

  /**
   * Pick the highest projected legal starting XI from the 15-man squad.
   * Enumerates every valid outfield formation (DEF 3-5 / MID 2-5 / FWD 1-3)
   * and picks the top players per position, then keeps the best-scoring one.
   */
  const autoPickXi = () => {
    const all = [...real.xi, ...real.bench]
      .map((id) => core.playersById.get(id))
      .filter((p): p is Player => Boolean(p));
    if (all.length < 11) return;

    const byPos: Record<Player['pos'], Player[]> = { GK: [], DEF: [], MID: [], FWD: [] };
    for (const p of all) byPos[p.pos].push(p);
    for (const key of Object.keys(byPos) as Player['pos'][]) {
      byPos[key].sort((a, b) => b.xpts - a.xpts);
    }

    let best: { total: number; players: Player[] } | null = null;
    for (let d = 3; d <= 5; d++) {
      for (let m = 2; m <= 5; m++) {
        for (let f = 1; f <= 3; f++) {
          if (d + m + f !== 10) continue;
          if (
            byPos.GK.length < 1 ||
            byPos.DEF.length < d ||
            byPos.MID.length < m ||
            byPos.FWD.length < f
          ) {
            continue;
          }
          const players = [
            byPos.GK[0],
            ...byPos.DEF.slice(0, d),
            ...byPos.MID.slice(0, m),
            ...byPos.FWD.slice(0, f),
          ];
          const total = players.reduce((s, p) => s + p.xpts, 0);
          if (!best || total > best.total) best = { total, players };
        }
      }
    }
    if (!best) return;

    const xiIds = best.players.map((p) => p.id);
    const benchIds = all
      .filter((p) => !xiIds.includes(p.id))
      .sort((a, b) => b.xpts - a.xpts)
      .map((p) => p.id);
    const ranked = best.players.slice().sort((a, b) => b.xpts - a.xpts);
    const next: RealSquad = {
      xi: xiIds,
      bench: benchIds,
      captainId: ranked[0]?.id ?? null,
      viceId: ranked[1]?.id ?? null,
    };

    setOverrides(next);
    setSandbox((s) => {
      if (!s.on) return s;
      return snap(initFromReal(s, next));
    });
  };

  const toggleSandbox = () =>
    setSandbox((s) => {
      const base = s.initialized ? s : initFromReal(s, real);
      return { ...base, on: !base.on };
    });

  const resetSandbox = () =>
    setSandbox((s) => {
      const base = initFromReal(s, real);
      return snap({ ...base, transfers: [], outId: null, inId: null });
    });

  const undo = () =>
    setSandbox((s) => {
      const last = s.history[s.history.length - 1];
      if (!last) return s;
      return {
        ...s,
        initialized: true,
        xi: last.xi,
        bench: last.bench,
        captainId: last.captainId,
        viceId: last.viceId,
        transfers: last.transfers,
        history: s.history.slice(0, -1),
      };
    });

  const applyToTeam = () => {
    setOverrides({
      xi: sandbox.xi.slice(),
      bench: sandbox.bench.slice(),
      captainId: sandbox.captainId,
      viceId: sandbox.viceId,
    });
    setSandbox((s) => ({ ...s, on: false, transfers: [], history: [] }));
  };

  const applyRec = (outId: number, inId: number): boolean => {
    if (!outId || !inId) return false;
    const inXi = sandbox.xi.indexOf(outId) > -1;
    const inBench = sandbox.bench.indexOf(outId) > -1;
    if (!inXi && !inBench) return false;
    const record: TransferRecord = {
      outId,
      inId,
      outName: name(outId),
      inName: name(inId),
      gain: gainOf(outId, inId),
    };
    setSandbox((s) => {
      const base = initFromReal(s, real);
      const next = snap(base);
      return {
        ...next,
        xi: replaceIn(next.xi, outId, inId),
        bench: replaceIn(next.bench, outId, inId),
        captainId: next.captainId === outId ? inId : next.captainId,
        viceId: next.viceId === outId ? inId : next.viceId,
        transfers: [...next.transfers, record],
        outId: null,
        inId: null,
      };
    });
    return true;
  };

  const applySandboxTransfer = (): boolean => {
    if (sandbox.outId == null || sandbox.inId == null) return false;
    return applyRec(sandbox.outId, sandbox.inId);
  };

  const pickOut = (id: number) =>
    setSandbox((s) => ({ ...initFromReal(s, real), outId: id, inId: null }));
  const pickIn = (id: number) => setSandbox((s) => ({ ...initFromReal(s, real), inId: id }));
  const clearPair = () => setSandbox((s) => ({ ...initFromReal(s, real), outId: null, inId: null }));

  const sandboxDelta = (): SandboxDelta => {
    const manager = core.manager;
    const free = manager?.freeTransfers ?? 1;
    const bank = manager?.bank ?? 0;
    const realCap = real.captainId ?? real.xi[0] ?? null;
    const sandCap = sandbox.captainId ?? sandbox.xi[0] ?? null;
    const realGw = xiXpts(real.xi, core.playersById, realCap);
    const sandGw = xiXpts(sandbox.xi, core.playersById, sandCap);
    const real5 = futureXpts(real.xi, core.playersById, realCap, core.fixtureIndex, core.nextGW, 5, 38);
    const sand5 = futureXpts(sandbox.xi, core.playersById, sandCap, core.fixtureIndex, core.nextGW, 5, 38);
    const spend = sandbox.transfers.reduce((t, tr) => {
      const pin = core.playersById.get(tr.inId)?.price ?? 0;
      const pout = core.playersById.get(tr.outId)?.price ?? 0;
      return t + (pin - pout);
    }, 0);
    const hits = Math.max(0, sandbox.transfers.length - free);
    return {
      count: sandbox.transfers.length,
      free,
      hits,
      hitCost: hits * 4,
      realGw,
      sandGw,
      gwDelta: Math.round((sandGw - realGw) * 10) / 10,
      real5,
      sand5,
      d5: Math.round((sand5 - real5) * 10) / 10,
      bank: Math.round((bank - spend) * 10) / 10,
      bankDelta: Math.round(-spend * 10) / 10,
    };
  };

  return (
    <SquadContext.Provider
      value={{
        ...real,
        sandbox,
        byId: core.playersById,
        setCaptain,
        setVice,
        swap,
        quickCaptain,
        quickBench,
        autoPickXi,
        toggleSandbox,
        resetSandbox,
        undo,
        applyToTeam,
        applySandboxTransfer,
        applyRecommendation: applyRec,
        pickOut,
        pickIn,
        clearPair,
        sandboxDelta,
      }}
    >
      {children}
    </SquadContext.Provider>
  );
}

export function useSquad(): SquadContextValue {
  const ctx = useContext(SquadContext);
  if (!ctx) throw new Error('useSquad must be used within SquadProvider');
  return ctx;
}
