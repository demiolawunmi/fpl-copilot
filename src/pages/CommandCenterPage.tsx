import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import { toast } from 'sonner';
import { ChevronUp, ChevronDown, Loader2 } from 'lucide-react';
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { buttonVariants } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { useNavigate, useLocation } from 'react-router-dom';
import { useTeamId } from '../context/TeamIdContext';
import {
  mockCommandCenterAISummary,
  mockFixturesSnapshot,
  mockRecommendedTransfers,
  mockModelSources,
  mockVideoInsights,
} from '../data/commandCenterMocks';
import type {
  CommandCenterAISummary,
  EnhancedPlayer,
  ModelSource,
  RecommendedTransferItem,
  SandboxAction,
  TeamStatus,
} from '../data/commandCenterMocks';

// Import components
import StatusStrip from '../components/command-center/StatusStrip';
import SeasonStatusBanner from '../components/command-center/SeasonStatusBanner';
import PitchCard from '../components/gw-overview/PitchCard';
import AICommandSummary from '../components/command-center/AICommandSummary';
import InjuriesSuspensionsCard from '../components/command-center/InjuriesSuspensionsCard';
import FixturesSnapshot from '../components/command-center/FixturesSnapshot';
import QuickActions from '../components/command-center/QuickActions';
import SandboxControls from '../components/command-center/SandboxControls';
import DeltaStrip from '../components/command-center/DeltaStrip';
import RecommendedTransfersList from '../components/command-center/RecommendedTransfersList';
import CustomTransferBuilder from '../components/command-center/CustomTransferBuilder';
import ModelComparisonPanel from '../components/command-center/ModelComparisonPanel';
import SandboxCharts from '../components/command-center/SandboxCharts';
import AskCopilotChat from '../components/command-center/AskCopilotChat';
import VideoInsightsStrip from '../components/command-center/VideoInsightsStrip';
import { DashboardCard } from '@/components/ui/primitives';

// Command Center hook – targets the NEXT GW and uses /api/fpl/my-team picks
import { useCommandCenterData } from '../hooks/useCommandCenterData';
import { usePredictionsData } from '../hooks/usePredictionsData';
import type { Player as UiPlayer } from '../data/gwOverviewMocks';
import InFormCard from '../components/command-center/InFormCard';
import BandwagonsCard from '../components/command-center/BandwagonsCard';
import { getOpponentDifficulty } from '../utils/difficulty';
import { runAirsenal } from '../api/backend';
import {
  submitCopilotBlendJob,
  pollCopilotBlendJob,
  isApiError,
  getCopilotBlendSnapshot,
  getCopilotBlendSnapshotGlobal,
  type CopilotSourceWeights,
  type CopilotBlendJobStatusResponse,
  type CopilotErrorResponse,
  type CopilotHybridResultPayload,
  type CopilotBlendSubmitRequest,
  type CopilotBlendSnapshot,
} from '../api/backend';
import { elementTypeToPosition, getPlayerPhotoUrl } from '../api/fpl/fpl';

type Tab = 'pick-team' | 'sandbox';

type BlendApplyPhase = 'idle' | 'submitting' | 'queued' | 'running' | 'completed' | 'failed';

type BlendApplyUiState = {
  phase: BlendApplyPhase;
  jobId?: string;
  message?: string;
  retryable?: boolean;
  error?: CopilotErrorResponse | null;
};

const BLEND_SCHEMA_VERSION = '1.0';
const BLEND_POLL_INTERVAL_MS = 1500;
const BLEND_POLL_TIMEOUT_MS = 90_000;
const DEFAULT_BLEND_TAB: Tab = 'pick-team';

const parseInitialTab = (rawSearch: string): Tab => {
  const tab = new URLSearchParams(rawSearch).get('tab');
  return tab === 'sandbox' || tab === 'pick-team' ? tab : DEFAULT_BLEND_TAB;
};

const parseBlendWeightsFromSearch = (rawSearch: string, sources: ModelSource[]): Map<string, number> => {
  const raw = new URLSearchParams(rawSearch).get('blend');
  if (!raw) return new Map();

  const values = raw
    .split(',')
    .map((entry) => entry.trim())
    .filter(Boolean)
    .map((entry) => {
      const [id, valueText] = entry.split(':').map((part) => part.trim());
      const weight = Number.parseInt(valueText ?? '', 10);
      return {
        id,
        weight,
      };
    })
    .filter((entry) => entry.id && Number.isFinite(entry.weight));

  if (values.length === 0) return new Map();

  const validIds = new Set(sources.map((source) => source.id));
  const map = new Map<string, number>();
  for (const entry of values) {
    if (!validIds.has(entry.id)) continue;
    map.set(entry.id, Math.max(0, Math.min(100, entry.weight)));
  }
  return map;
};

const clampWeeksAhead = (n: number) => Math.min(38, Math.max(1, Math.round(n)));

const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();

const confidenceToTone = (confidence: number): 'good' | 'info' | 'warn' => {
  if (confidence >= 0.67) return 'good';
  if (confidence >= 0.4) return 'info';
  return 'warn';
};

const sleep = (ms: number) => new Promise<void>((resolve) => {
  window.setTimeout(resolve, ms);
});

/** Only hydrate saved JSON when it matches the logged-in FPL entry (strict when team id is set). */
const snapshotMatchesTeam = (
  snap: CopilotBlendSnapshot,
  currentTeamId: number | null,
): boolean => {
  const st = snap.input?.fpl_team_id;
  if (currentTeamId != null && currentTeamId > 0) {
    return typeof st === 'number' && st === currentTeamId;
  }
  return st == null;
};

const createCorrelationId = () => {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }

  return `cc-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
};

/** FPL formation rules: exactly 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD, 11 starters total. */
function validateFormation(squad: EnhancedPlayer[]): string | null {
  const starters = squad.filter((p) => !p.isBench);
  const gk = starters.filter((p) => p.position === 'GK').length;
  const def = starters.filter((p) => p.position === 'DEF').length;
  const mid = starters.filter((p) => p.position === 'MID').length;
  const fwd = starters.filter((p) => p.position === 'FWD').length;
  if (gk !== 1) return `Must have exactly 1 starting GK (would have ${gk})`;
  if (def < 3) return `Need at least 3 starting DEF (would have ${def})`;
  if (def > 5) return `Max 5 starting DEF (would have ${def})`;
  if (mid < 2) return `Need at least 2 starting MID (would have ${mid})`;
  if (mid > 5) return `Max 5 starting MID (would have ${mid})`;
  if (fwd < 1) return `Need at least 1 starting FWD (would have ${fwd})`;
  if (fwd > 3) return `Max 3 starting FWD (would have ${fwd})`;
  if (starters.length !== 11) return `Must have 11 starters (would have ${starters.length})`;
  return null;
}

const mapUiPlayerToEnhanced = (
  p: UiPlayer,
  lookupPrediction: (name: string, teamAbbr?: string) => import('../api/backend').PredictionPlayer | undefined,
  fixturesByName: Map<string, import('../api/backend').PlayerFixture>,
): EnhancedPlayer => {
  const pred = lookupPrediction(p.name, p.teamAbbr);
  const fixture = fixturesByName.get(norm(p.name));

  return {
    id: p.id ?? 0,
    name: p.name,
    position: p.position,
    team: '',
    teamAbbr: p.teamAbbr ?? '',
    price: p.sellingPrice ? p.sellingPrice / 10 : 0,
    xPts: pred?.xp ?? 0,
    points: p.points ?? 0,
    minutesRisk: 'Unknown',
    injuryStatus: 'Available',
    isCaptain: p.isCaptain,
    isViceCaptain: p.isViceCaptain,
    isBench: p.isBench,
    photoUrl: p.photoUrl,
    opponents: fixture?.fixtures?.map((f) =>
      `${f.is_home ? 'H' : 'A'} ${f.opponent_short}`
    ) ?? p.opponents,
  };
};

const CommandCenterPage = () => {
  const { teamId } = useTeamId();
  const navigate = useNavigate();
  const location = useLocation();
  const [activeTab, setActiveTab] = useState<Tab>(() => parseInitialTab(location.search));

  // Dedicated Command Center hook – always targets next GW, uses backend picks
  const cc = useCommandCenterData(teamId);
  
  // AIrsenal predictions & fixtures hook
  const predictions = usePredictionsData(cc.nextGW > 0 ? cc.nextGW : null);

  // Bootstrap elements list for photo resolution in side-cards
  const bootstrapElements = useMemo(() => cc.bootstrap?.elements ?? [], [cc.bootstrap]);

  // State for sandbox
  const [sandboxSquad, setSandboxSquad] = useState<EnhancedPlayer[]>([]);
  const [sandboxActions, setSandboxActions] = useState<SandboxAction[]>([]);
  const [sandboxMode, setSandboxMode] = useState(false);
  const [optimizationLoading, setOptimizationLoading] = useState(false);
  const [optimizationDialogOpen, setOptimizationDialogOpen] = useState(false);
  const [weeksAhead, setWeeksAhead] = useState(3);
  const [swapSelection, setSwapSelection] = useState<number | null>(null);
  const [sandboxBankDelta, setSandboxBankDelta] = useState(0);
  const [blendApplyState, setBlendApplyState] = useState<BlendApplyUiState>({ phase: 'idle' });
  const [completedBlendPayload, setCompletedBlendPayload] = useState<CopilotHybridResultPayload | null>(null);
  const [savedBlendInput, setSavedBlendInput] = useState<CopilotBlendSubmitRequest | null>(null);
  const [modelSources, setModelSources] = useState<ModelSource[]>(() => {
    const defaults = mockModelSources.map((source) => ({ ...source }));
    const seeded = parseBlendWeightsFromSearch(location.search, defaults);
    if (seeded.size === 0) return defaults;
    return defaults.map((source) => (
      seeded.has(source.id)
        ? { ...source, weight: seeded.get(source.id) ?? source.weight }
        : source
    ));
  });
  const activeBlendPollRunRef = useRef(0);

  const blendTotal = useMemo(
    () => modelSources.reduce((sum, source) => sum + source.weight, 0),
    [modelSources],
  );
  const blendRemaining = 100 - blendTotal;
  const isBlendInvalid = blendTotal > 100;

  const blendStatusMessage = useMemo(() => {
    // Map UI states: valid-zero (no suggestions), degraded (provider/schema fallback), pending/running, failed
    if (blendApplyState.phase === 'completed' && completedBlendPayload) {
      const transferCount = completedBlendPayload.recommended_transfers.length;
      const confidencePct = Math.round(completedBlendPayload.core.confidence * 100);

      if (completedBlendPayload.degraded_mode?.is_degraded) {
        const code = completedBlendPayload.degraded_mode.code ?? 'FALLBACK';
        const msg = completedBlendPayload.degraded_mode.message ?? '';
        return (
          <p className="font-medium text-orange-300">
            <span className="mr-1">⚠️</span>
            Degraded Output ({code}): {msg}
          </p>
        );
      }

      if (transferCount === 0) {
        // Valid zero: model intentionally returned no confident suggestions
        return (
          <p className="font-medium text-blue-300">
            <span className="mr-1">ℹ️</span>
            No confident transfer suggestions. Confidence: {confidencePct}%
          </p>
        );
      }

      return (
        <p className="font-medium text-green-300">
          <span className="mr-1">✅</span>
          Blend ready: {transferCount} transfer suggestion(s), {confidencePct}% confidence.
        </p>
      );
    }

    if (blendApplyState.phase === 'running' || blendApplyState.phase === 'submitting' || blendApplyState.phase === 'queued') {
      return <p className="text-blue-200">{blendApplyState.message ?? 'Blend job in progress...'}</p>;
    }

    if (blendApplyState.phase === 'failed') {
      const errorCode = blendApplyState.error?.error.code;
      const baseMsg = blendApplyState.message ?? 'Blend job failed.';
      return (
        <p className="font-medium text-red-300">
          <span className="mr-1">❌</span>
          {errorCode ? `${baseMsg} [${errorCode}]` : baseMsg}
        </p>
      );
    }

    return <p className="text-slate-300">{blendApplyState.message}</p>;
  }, [blendApplyState, completedBlendPayload]);

  const realSquad = useMemo(() => {
    if (cc.loading || cc.error != null || cc.squad.length === 0 || predictions.loading) {
      return [] as EnhancedPlayer[];
    }

    return cc.squad.map((p) =>
      mapUiPlayerToEnhanced(
        p as UiPlayer,
        predictions.lookupPrediction,
        predictions.fixturesByName,
      ),
    );
  }, [cc.loading, cc.error, cc.squad, predictions.loading, predictions.lookupPrediction, predictions.fixturesByName]);

  const currentSandboxSquad = sandboxActions.length > 0 ? sandboxSquad : realSquad;
  const hasLiveMyTeam = cc.myTeam != null && realSquad.length > 0;

  const teamStatus: TeamStatus = useMemo(() => {
    const mt = cc.myTeam;
    if (!mt) {
      return {
        freeTransfers: 0,
        bank: 0,
        teamValue: Number(realSquad.reduce((sum, player) => sum + (player.price || 0), 0).toFixed(1)),
        chips: {
          wildcard: { available: false },
          freehit: { available: false },
          bboost: { available: false },
          tcaptain: { available: false },
        },
        deadline: cc.bootstrap?.events.find((e) => e.is_next)?.deadline_time ?? new Date().toISOString(),
      };
    }

    const chipMap: Record<string, keyof TeamStatus['chips']> = {
      wildcard: 'wildcard',
      freehit: 'freehit',
      bboost: 'bboost',
      '3xc': 'tcaptain',
    };

    const chips: TeamStatus['chips'] = {
      wildcard: { available: false },
      freehit: { available: false },
      bboost: { available: false },
      tcaptain: { available: false },
    };

    for (const c of mt.chips) {
      const key = chipMap[c.name];
      if (!key) continue;
      const isAvailable = c.status_for_entry === 'available' && !c.is_pending;
      const usedGW = c.played_by_entry.length > 0 ? `GW ${c.played_by_entry[0]}` : undefined;
      chips[key] = { available: isAvailable, used: usedGW };
    }

    const nextEvent = cc.bootstrap?.events.find((e) => e.is_next);
    const deadline = nextEvent?.deadline_time ?? new Date().toISOString();

    const transferLimit = Number(mt.transfers.limit ?? 0);
    const transfersMade = Number(mt.transfers.made ?? 0);
    const bankTenths = Number(mt.transfers.bank ?? 0);
    const backendValueTenths = Number(mt.transfers.value ?? 0);
    const derivedSquadValue = realSquad.reduce((sum, player) => sum + (player.price || 0), 0);

    return {
      freeTransfers: Math.max(0, transferLimit - transfersMade),
      bank: Number((bankTenths / 10).toFixed(1)),
      teamValue: Number(((backendValueTenths > 0 ? backendValueTenths / 10 : derivedSquadValue)).toFixed(1)),
      chips,
      deadline,
    };
  }, [cc.myTeam, cc.bootstrap, realSquad]);

  const nextGW = cc.nextGW;
  const teamName = cc.gwInfo?.teamName ?? 'My Team';

  useEffect(() => {
    let cancelled = false;
    const parsed = teamId?.trim() ? Number.parseInt(teamId.trim(), 10) : NaN;
    const currentTeamId = Number.isFinite(parsed) && parsed > 0 ? parsed : null;
    const gw = nextGW > 0 ? nextGW : 0;
    if (gw <= 0) return;

    async function loadSnapshot() {
      try {
        const snap =
          currentTeamId != null
            ? await getCopilotBlendSnapshot(gw, currentTeamId)
            : await getCopilotBlendSnapshotGlobal(gw);
        if (cancelled || !snap) return;
        if (!snapshotMatchesTeam(snap, currentTeamId)) return;
        setCompletedBlendPayload(snap.result);
        setSavedBlendInput(snap.input as CopilotBlendSubmitRequest);
        setBlendApplyState({
          phase: 'completed',
          message: 'Loaded saved blend.',
        });
      } catch {
        // Backend offline or not running — keep local state
      }
    }

    void loadSnapshot();
    return () => {
      cancelled = true;
    };
  }, [teamId, nextGW]);

  const hybridSummary = useMemo<CommandCenterAISummary>(() => {
    if (!completedBlendPayload) {
      return mockCommandCenterAISummary;
    }

    const ask = completedBlendPayload.ask_copilot;
    const bulletTone = confidenceToTone(ask.confidence);
    const rationaleText = ask.rationale.filter((item) => item.trim().length > 0);
    const primaryText = (ask.answer?.trim() ?? '') || completedBlendPayload.core.summary;

    const bullets = [
      {
        text: primaryText,
        why: rationaleText.join(' ') || completedBlendPayload.core.summary,
        tone: bulletTone,
      },
      ...rationaleText.slice(0, 4).map((item) => ({
        text: item,
        why: completedBlendPayload.core.summary,
        tone: bulletTone,
      })),
    ];

    return {
      title: `AI Summary (GW ${nextGW || mockCommandCenterAISummary.gameweek})`,
      gameweek: nextGW || mockCommandCenterAISummary.gameweek,
      bullets: bullets.length > 0 ? bullets : mockCommandCenterAISummary.bullets,
    };
  }, [completedBlendPayload, nextGW]);

  const hybridRecommendedTransfers = useMemo<RecommendedTransferItem[]>(() => {
    if (!completedBlendPayload) {
      return mockRecommendedTransfers;
    }

    const teamById = new Map((cc.bootstrap?.teams ?? []).map((team) => [team.id, team]));

    const resolveFromCurrentData = (
      playerId: number,
      playerName: string,
      fplApiId?: number,
    ): EnhancedPlayer => {
      const resolvedId = fplApiId ?? playerId;
      const fromSquad = (fplApiId != null
        ? currentSandboxSquad.find((player) => player.id === fplApiId)
          ?? realSquad.find((player) => player.id === fplApiId)
        : undefined)
        ?? currentSandboxSquad.find((player) => player.id === playerId)
        ?? realSquad.find((player) => player.id === playerId)
        ?? currentSandboxSquad.find((player) => norm(player.name) === norm(playerName))
        ?? realSquad.find((player) => norm(player.name) === norm(playerName));

      if (fromSquad) {
        return {
          ...fromSquad,
          id: resolvedId,
          name: playerName,
        };
      }

      const fromBootstrap = (fplApiId != null
        ? (cc.bootstrap?.elements ?? []).find((element) => element.id === fplApiId)
        : undefined)
        ?? (cc.bootstrap?.elements ?? []).find((element) => element.id === playerId)
        ?? (cc.bootstrap?.elements ?? []).find((element) => norm(element.web_name) === norm(playerName));

      const bootstrapTeam = fromBootstrap ? teamById.get(fromBootstrap.team) : undefined;
      const resolvedName = fromBootstrap?.web_name ?? playerName;
      const teamAbbr = bootstrapTeam?.short_name ?? '';
      const predictedXp = predictions.lookupPrediction(resolvedName, teamAbbr)?.xp ?? 0;

      return {
        id: resolvedId,
        name: resolvedName,
        position: fromBootstrap ? elementTypeToPosition(fromBootstrap.element_type) : 'MID',
        team: bootstrapTeam?.name ?? '',
        teamAbbr,
        price: fromBootstrap ? Number(((fromBootstrap.now_cost ?? 0) / 10).toFixed(1)) : 0,
        xPts: predictedXp,
        points: 0,
        minutesRisk: 'Unknown',
        injuryStatus: 'Available',
        opponents: [],
      };
    };

    return completedBlendPayload.recommended_transfers.map((transfer) => {
      const playerIn = resolveFromCurrentData(transfer.in.player_id, transfer.in.player_name, transfer.in.fpl_api_id);
      const playerOut = resolveFromCurrentData(transfer.out.player_id, transfer.out.player_name, transfer.out.fpl_api_id);
      const xPtsDelta = Number((playerIn.xPts - playerOut.xPts).toFixed(1));
      return {
        playerIn,
        playerOut,
        xPtsDelta,
        why: transfer.reason,
      };
    });
  }, [cc.bootstrap, completedBlendPayload, currentSandboxSquad, predictions, realSquad]);

  // Sandbox handlers
  const handleUndo = () => {
    if (sandboxActions.length === 0) return;
    const newActions = [...sandboxActions];
    newActions.pop();
    setSandboxActions(newActions);
    // Recompute squad from actions
    // TODO: implement proper undo logic
  };

  const handleReset = () => {
    setSandboxSquad(realSquad.map((player) => ({ ...player })));
    setSandboxActions([]);
    setSandboxBankDelta(0);
  };

  const handleApplyToTeam = () => {
    alert('Applied to team (UI only)');
  };

  const getBlendFailureState = useCallback((params: {
    message: string;
    retryable?: boolean;
    error?: CopilotErrorResponse | null;
    jobId?: string;
  }): BlendApplyUiState => ({
    phase: 'failed',
    message: params.message,
    retryable: params.retryable ?? true,
    error: params.error ?? null,
    jobId: params.jobId,
  }), []);

  const handleModelWeightChange = useCallback((modelId: string, nextWeight: number) => {
    const boundedWeight = Math.max(0, Math.min(100, Math.round(nextWeight)));
    setModelSources((prev) => prev.map((source) => (
      source.id === modelId ? { ...source, weight: boundedWeight } : source
    )));
  }, []);

  const applyModelBlend = useCallback(async () => {
    if (isBlendInvalid) {
      setBlendApplyState(getBlendFailureState({
        message: 'Blend total exceeds 100%. Reduce source weights before applying.',
        retryable: false,
      }));
      return;
    }

    const blendableSources = modelSources.filter((source) => source.backendField);
    const totalWeight = blendableSources.reduce((sum, source) => sum + source.weight, 0);
    if (totalWeight <= 0) {
      setBlendApplyState(getBlendFailureState({
        message: 'Set at least one source weight above 0 before applying.',
        retryable: false,
      }));
      return;
    }

    // The backend requires source_weights to sum to exactly 1.0, but the UI
    // allows any total up to 100 — normalise so the relative weights are kept
    // regardless of how the sliders happen to add up.
    const sourceWeights: Record<string, number> = {};
    for (const source of blendableSources) {
      if (!source.backendField) continue;
      sourceWeights[source.backendField] = source.weight / totalWeight;
    }

    const sandboxTransferCount = sandboxActions.filter((action) => action.type === 'transfer').length;
    const blendFreeTransfers = Math.max(0, teamStatus.freeTransfers - sandboxTransferCount);
    const blendBank = Number((teamStatus.bank + sandboxBankDelta).toFixed(1));
    const blendGameweek = nextGW > 0 ? nextGW : undefined;
    const blendCurrentSquad = currentSandboxSquad.map((player) => ({
      fpl_api_id: player.id,
      player_name: player.name,
      team: player.team,
      position: player.position,
      price: player.price,
      x_pts: player.xPts,
    }));

    const runId = Date.now();
    activeBlendPollRunRef.current = runId;
    const isCurrentRun = () => activeBlendPollRunRef.current === runId;

    setCompletedBlendPayload(null);
    setSavedBlendInput(null);
    setBlendApplyState({ phase: 'submitting', message: 'Submitting blend request...' });

    try {
      const correlationId = createCorrelationId();
      const parsedTeamId = teamId?.trim() ? Number.parseInt(teamId.trim(), 10) : undefined;
      const fplTeamId =
        parsedTeamId != null && Number.isFinite(parsedTeamId) && parsedTeamId > 0
          ? parsedTeamId
          : undefined;

      const blendRequestBody: CopilotBlendSubmitRequest = {
        schema_version: BLEND_SCHEMA_VERSION,
        correlation_id: correlationId,
        source_weights: sourceWeights as unknown as CopilotSourceWeights,
        gameweek: blendGameweek,
        bank: blendBank,
        free_transfers: blendFreeTransfers,
        current_squad: blendCurrentSquad,
        task: 'hybrid',
        force_refresh: true,
        ...(fplTeamId != null ? { fpl_team_id: fplTeamId } : {}),
      };

      const accepted = await submitCopilotBlendJob(blendRequestBody);

      if (!isCurrentRun()) {
        return;
      }

      setBlendApplyState({
        phase: 'queued',
        jobId: accepted.job_id,
        message: 'Blend job queued...',
      });

      const startedAt = Date.now();
      let latestStatus: CopilotBlendJobStatusResponse | null = null;

      while (isCurrentRun()) {
        latestStatus = await pollCopilotBlendJob(accepted.job_id);
        if (!isCurrentRun()) {
          return;
        }

        if (latestStatus.status === 'queued') {
          setBlendApplyState({
            phase: 'queued',
            jobId: accepted.job_id,
            message: 'Blend job queued...',
          });
        }

        if (latestStatus.status === 'running') {
          setBlendApplyState({
            phase: 'running',
            jobId: accepted.job_id,
            message: 'Generating hybrid model output...',
          });
        }

        if (latestStatus.status === 'completed') {
          const resultPayload = latestStatus.result;
          if (!resultPayload) {
            setBlendApplyState(getBlendFailureState({
              message: 'Blend job completed without result payload.',
              retryable: true,
              jobId: accepted.job_id,
            }));
            return;
          }

          setCompletedBlendPayload(resultPayload);
          setSavedBlendInput(blendRequestBody);
          setBlendApplyState({
            phase: 'completed',
            jobId: accepted.job_id,
            message: resultPayload.degraded_mode.is_degraded
              ? 'Blend applied with degraded fallback output.'
              : 'Blend applied successfully.',
          });
          return;
        }

        if (latestStatus.status === 'failed') {
          const backendError = latestStatus.error;
          const message = backendError?.error.message ?? 'Blend job failed on backend.';
          setBlendApplyState(getBlendFailureState({
            message,
            retryable: backendError?.error.retryable ?? true,
            error: backendError,
            jobId: accepted.job_id,
          }));
          return;
        }

        if (Date.now() - startedAt > BLEND_POLL_TIMEOUT_MS) {
          setBlendApplyState(getBlendFailureState({
            message: 'Blend job timed out while polling. Retry to continue.',
            retryable: true,
            jobId: accepted.job_id,
          }));
          return;
        }

        await sleep(BLEND_POLL_INTERVAL_MS);
      }
    } catch (error) {
      if (!isCurrentRun()) {
        return;
      }

      if (isApiError(error)) {
        setBlendApplyState(getBlendFailureState({
          message: error.message,
          retryable: error.status === 0 || error.status >= 500,
        }));
        return;
      }

      setBlendApplyState(getBlendFailureState({
        message: error instanceof Error ? error.message : 'Blend apply failed unexpectedly.',
        retryable: true,
      }));
    }
  }, [currentSandboxSquad, getBlendFailureState, isBlendInvalid, modelSources, nextGW, sandboxActions, sandboxBankDelta, teamId, teamStatus.bank, teamStatus.freeTransfers]);

  function handleRefreshAISummary() {
    if (isBlendInvalid) {
      toast.warning('Blend weights invalid', {
        description: 'Total exceeds 100%. Open AI Sandbox and reduce source weights before refreshing.',
        duration: 6000,
      });
      return;
    }
    if (nextGW <= 0) {
      toast.warning('Gameweek not ready', {
        description: 'Wait for team data to load, then try again.',
        duration: 5000,
      });
      return;
    }
    void applyModelBlend();
  }

  const blendJobBusy =
    blendApplyState.phase === 'submitting' ||
    blendApplyState.phase === 'queued' ||
    blendApplyState.phase === 'running';

  useEffect(() => () => {
    activeBlendPollRunRef.current = 0;
  }, []);

  const handleTransfer = (playerInId: number, playerOutId: number) => {
    const sourceSquad = currentSandboxSquad;
    const action: SandboxAction = {
      type: 'transfer',
      payload: { playerInId, playerOutId },
      timestamp: new Date(),
    };

    let inPlayer: EnhancedPlayer | undefined = realSquad.find((p) => p.id === playerInId);

    if (!inPlayer) {
      const el = bootstrapElements.find((e) => e.id === playerInId);
      if (el) {
        const team = cc.bootstrap?.teams.find((t) => t.id === el.team);
        const pred = predictions.lookupPrediction(el.web_name, team?.short_name);
        const fixture = predictions.fixturesByName.get(norm(el.web_name));
        inPlayer = {
          id: el.id,
          name: el.web_name,
          position: elementTypeToPosition(el.element_type),
          team: team?.name ?? '',
          teamAbbr: team?.short_name ?? '',
          price: el.now_cost ? el.now_cost / 10 : 0,
          xPts: pred?.xp ?? 0,
          points: 0,
          minutesRisk: 'Unknown',
          injuryStatus: 'Available',
          photoUrl: getPlayerPhotoUrl(el.code),
          opponents: fixture?.fixtures?.map((f) => `${f.is_home ? 'H' : 'A'} ${f.opponent_short}`) ?? [],
        };
      }
    }

    if (!inPlayer) return;

    const outPlayer = sourceSquad.find((p) => p.id === playerOutId);
    const outPrice = outPlayer?.price ?? 0;
    const inPrice = inPlayer.price ?? 0;
    const priceDelta = outPrice - inPrice;

    const newActions = [...sandboxActions, action];
    const transfersMade = newActions.filter((a) => a.type === 'transfer').length;
    const freeTransfers = teamStatus.freeTransfers;
    const newBank = teamStatus.bank + sandboxBankDelta + priceDelta;

    if (newBank < 0) {
      toast.error('Insufficient funds', {
        description: `This transfer would leave you with £${newBank.toFixed(1)}m. You need more bank.`,
        duration: 4000,
      });
      return;
    }

    setSandboxBankDelta((prev) => Number((prev + priceDelta).toFixed(1)));
    setSandboxActions(newActions);
    setSandboxSquad(() => {
      const outIdx = sourceSquad.findIndex((p) => p.id === playerOutId);
      if (outIdx === -1) return sourceSquad;
      const out = sourceSquad[outIdx];
      const next = sourceSquad.slice();
      next[outIdx] = { ...inPlayer, isBench: out.isBench, isCaptain: false, isViceCaptain: false };
      return next;
    });

    const hitMsg = transfersMade > freeTransfers
      ? ` (−${(transfersMade - freeTransfers) * 4} pts hit)`
      : '';

    const desc = `${outPlayer?.name ?? 'Player'} out → ${inPlayer.name} in${hitMsg}`;
    if (transfersMade > freeTransfers) {
      toast.warning('Transfer applied', { description: desc, duration: 4000 });
    } else {
      toast.success('Transfer applied', { description: desc, duration: 4000 });
    }
  };

  const handleSetCaptain = (playerId: number) => {
    const newSquad = currentSandboxSquad.map((p) => {
      if (p.id === playerId) {
        return { ...p, isCaptain: true, isViceCaptain: false };
      } else if (p.isCaptain) {
        return { ...p, isCaptain: false, isViceCaptain: true };
      } else {
        return { ...p, isViceCaptain: false };
      }
    });
    setSandboxSquad(newSquad);
    setSandboxActions((prev) => [...prev, { type: 'captain', payload: { playerId }, timestamp: new Date() }]);
  };

  const handleSetViceCaptain = (playerId: number) => {
    const player = currentSandboxSquad.find((p) => p.id === playerId);
    if (!player || player.isBench) return;
    if (player.isCaptain) {
      toast.info('Cannot assign', { description: 'The captain cannot also be vice-captain.', duration: 3000 });
      return;
    }
    const newSquad = currentSandboxSquad.map((p) => {
      if (p.id === playerId) return { ...p, isViceCaptain: true, isCaptain: false };
      return { ...p, isViceCaptain: false };
    });
    setSandboxSquad(newSquad);
    setSandboxActions((prev) => [...prev, { type: 'vice_captain', payload: { playerId }, timestamp: new Date() }]);
  };

  const handleAutoCaptain = () => {
    const starters = currentSandboxSquad.filter((p) => !p.isBench);
    if (starters.length === 0) return;
    const best = starters.reduce((a, b) => (a.xPts > b.xPts ? a : b));
    handleSetCaptain(best.id);
  };

  const handleAutoBench = () => {
    // Simple heuristic: sort by xPts, put lowest on bench
    // TODO: implement proper bench logic
    alert('Auto-bench feature coming soon');
  };

  const handleRollTransfer = () => {
    setActiveTab('sandbox');
  };

  const handleSandboxPlayerClick = useCallback(
    (player: import('../data/gwOverviewMocks').Player) => {
      const clickedId = player.id;
      if (clickedId == null) return;

      if (swapSelection == null) {
        setSwapSelection(clickedId);
        return;
      }

      if (swapSelection === clickedId) {
        setSwapSelection(null);
        return;
      }

      const squad = [...currentSandboxSquad];
      const idxA = squad.findIndex((p) => p.id === swapSelection);
      const idxB = squad.findIndex((p) => p.id === clickedId);
      if (idxA === -1 || idxB === -1) {
        setSwapSelection(null);
        return;
      }

      const a = { ...squad[idxA] };
      const b = { ...squad[idxB] };

      if (a.isBench === b.isBench) {
        toast.info('Invalid swap', {
          description: 'Select one starter and one bench player to swap.',
          duration: 3000,
        });
        setSwapSelection(null);
        return;
      }

      // Simulate the swap and validate the resulting formation
      const tmpBench = a.isBench;
      a.isBench = b.isBench;
      b.isBench = tmpBench;
      if (a.isBench) { a.isCaptain = false; a.isViceCaptain = false; }
      if (b.isBench) { b.isCaptain = false; b.isViceCaptain = false; }

      const simulated = squad.slice();
      simulated[idxA] = a;
      simulated[idxB] = b;

      const err = validateFormation(simulated);
      if (err) {
        toast.warning('Invalid formation', {
          description: err,
          duration: 4000,
        });
        setSwapSelection(null);
        return;
      }

      setSandboxSquad(simulated);
      setSandboxActions((prev) => [
        ...prev,
        { type: 'bench_order', payload: { playerA: swapSelection, playerB: clickedId }, timestamp: new Date() },
      ]);
      setSwapSelection(null);
    },
    [swapSelection, currentSandboxSquad],
  );

  const handleOpenOptimizationDialog = useCallback(() => {
    setWeeksAhead(3);
    setOptimizationDialogOpen(true);
  }, []);

  const handleConfirmOptimization = useCallback(async () => {
    const idStr = teamId?.trim();
    const fplTeamId = idStr ? Number.parseInt(idStr, 10) : Number.NaN;
    if (idStr == null || idStr === "" || !Number.isFinite(fplTeamId) || fplTeamId <= 0) {
      toast.warning('FPL team ID required', {
        description: 'Set your team ID in the app (navbar) so the optimizer knows which squad to run for.',
        duration: 6000,
      });
      return;
    }

    const w = clampWeeksAhead(weeksAhead);
    setOptimizationLoading(true);
    try {
      const res = await runAirsenal({
        action: 'pipeline',
        fpl_team_id: fplTeamId,
        gameweek: 'auto',
        weeks_ahead: w,
      });
      setOptimizationDialogOpen(false);
      const desc = res.ok
        ? `Action “${res.action}” completed (${res.steps?.length ?? 0} step(s)).`
        : `Completed with ok: false for “${res.action}”.`;
      if (res.ok) {
        toast.success('AIrsenal pipeline finished', { description: desc, duration: 8000 });
        window.setTimeout(() => window.location.reload(), 2000);
      } else {
        toast.warning('AIrsenal pipeline finished', { description: desc, duration: 8000 });
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Request failed';
      toast.error('AIrsenal run failed', {
        description: message,
        duration: 12000,
      });
    } finally {
      setOptimizationLoading(false);
    }
  }, [teamId, weeksAhead]);

  const mappedPickTeamSquad = currentSandboxSquad.map((p) => {
    const fixture = predictions.fixturesByName.get(norm(p.name));
    const firstFixture = fixture?.fixtures?.[0];

    let chipLabel: string | undefined;
    let chipDifficulty: number | undefined;

    if (firstFixture) {
      chipDifficulty = getOpponentDifficulty(firstFixture.opponent_short, firstFixture.difficulty);
      chipLabel = `${firstFixture.is_home ? 'H' : 'A'} ${firstFixture.opponent_short}`;
    } else if (p.opponents && p.opponents.length > 0) {
      chipLabel = p.opponents.join(', ');
      const firstOpponent = p.opponents[0] ?? '';
      const opponentShort = firstOpponent.replace(/^H\s+|^A\s+/i, '').trim();
      if (opponentShort) {
        chipDifficulty = getOpponentDifficulty(opponentShort);
      }
    } else {
      chipLabel = p.teamAbbr || undefined;
    }

    const pred = predictions.lookupPrediction(p.name, p.teamAbbr);
    const displayPoints = pred?.xp ?? p.xPts ?? 0;
    const roundedPoints = Number(displayPoints.toFixed(2));

    return {
      id: p.id,
      name: p.name,
      position: p.position,
      points: roundedPoints,
      isCaptain: p.isCaptain,
      isViceCaptain: p.isViceCaptain,
      isBench: p.isBench,
      photoUrl: p.photoUrl,
      teamAbbr: p.teamAbbr,
      chipLabel,
      chipDifficulty,
    };
  });

  const mappedSandboxSquad = currentSandboxSquad.map((p) => {
    const fixture = predictions.fixturesByName.get(norm(p.name));
    const firstFixture = fixture?.fixtures?.[0];

    let chipLabel: string | undefined;
    let chipDifficulty: number | undefined;

    if (firstFixture) {
      chipDifficulty = getOpponentDifficulty(firstFixture.opponent_short, firstFixture.difficulty);
      chipLabel = `${firstFixture.is_home ? 'H' : 'A'} ${firstFixture.opponent_short}`;
    } else if (p.opponents && p.opponents.length > 0) {
      chipLabel = p.opponents.join(', ');
      const opponentShort = (p.opponents[0] ?? '').replace(/^[HA]\s+/i, '').trim();
      if (opponentShort) chipDifficulty = getOpponentDifficulty(opponentShort);
    } else {
      chipLabel = p.teamAbbr || undefined;
    }

    const pred = predictions.lookupPrediction(p.name, p.teamAbbr);
    const displayPoints = pred?.xp ?? p.xPts ?? 0;

    return {
      id: p.id,
      name: p.name,
      position: p.position,
      points: Number(displayPoints.toFixed(2)),
      isCaptain: p.isCaptain,
      isViceCaptain: p.isViceCaptain,
      isBench: p.isBench,
      photoUrl: p.photoUrl,
      teamAbbr: p.teamAbbr,
      chipLabel,
      chipDifficulty,
    };
  });

  const loadingCard = (
    <DashboardCard className="p-6">
      <p className="text-center text-slate-400">Loading squad...</p>
    </DashboardCard>
  );

  const errorCard = (
    <DashboardCard className="p-6 bg-[rgba(127,29,29,0.18)] border-[rgba(248,113,113,0.22)]">
      <p className="text-center text-red-300">{cc.error}</p>
    </DashboardCard>
  );

  const emptyMyTeamCard = (
    <DashboardCard className="p-6">
      <div className="flex flex-col items-center gap-2">
        <p className="text-center font-semibold text-slate-300">No backend my_team.json squad loaded</p>
        <p className="text-center text-sm text-slate-500">
          Command Center is waiting for `/api/files/my_team` so it can render your real draft, bank, transfers, and chips.
        </p>
      </div>
    </DashboardCard>
  );

  return (
    <div className="flex flex-1 flex-col gap-6 px-4 py-6 md:px-6 xl:px-10 xl:py-8">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl font-bold leading-[1.33] text-white">Command Center</h1>
        <p className="text-sm text-slate-400">
          Gameweek {nextGW || '…'} • Team: {teamName} • ID: {teamId}
        </p>
      </div>

      <StatusStrip status={teamStatus} />

      <SeasonStatusBanner />

      <DashboardCard>
        <div className="flex border-b border-white/6">
          {(['pick-team', 'sandbox'] as const).map((tab) => (
            <button
              key={tab}
              type="button"
              onClick={() => setActiveTab(tab)}
              className={cn(
                'flex-1 cursor-pointer rounded-none border-b-2 px-4 py-3 text-sm font-semibold capitalize transition-colors',
                activeTab === tab ? 'border-emerald-400 text-emerald-400' : 'border-transparent text-slate-400',
                'hover:bg-transparent hover:text-white',
              )}
            >
              {tab === 'pick-team' ? `Pick Team (GW ${nextGW || '…'})` : 'AI Sandbox'}
            </button>
          ))}
        </div>

        <div className="p-4 md:p-6">
          {activeTab === 'pick-team' ? (
            <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
              <div className="col-span-1 xl:col-span-2">
                <div className="flex flex-col gap-6">
                  {cc.loading ? loadingCard : !hasLiveMyTeam ? (cc.error ? errorCard : emptyMyTeamCard) : (
                    <PitchCard
                      squad={mappedPickTeamSquad}
                      onPlayerClick={(player) => {
                        const id = player?.id;
                        if (id == null || typeof id !== 'number' || Number.isNaN(id) || id <= 0) return;
                        navigate(`/players/${id}`, { state: { from: location.pathname } });
                      }}
                    />
                  )}
                  <AICommandSummary
                    summary={hybridSummary}
                    onRefresh={handleRefreshAISummary}
                    isRefreshing={blendJobBusy}
                    disableRefresh={isBlendInvalid || nextGW <= 0}
                  />
                </div>
              </div>

              <div className="col-span-1">
                <div className="flex flex-col gap-6">
                  <QuickActions
                    onAutoCaptain={handleAutoCaptain}
                    onAutoBench={handleAutoBench}
                    onOpenOptimization={handleOpenOptimizationDialog}
                    onRollTransfer={handleRollTransfer}
                    isOptimizationLoading={optimizationLoading}
                  />
                  <InFormCard bootstrapElements={bootstrapElements} />
                  <BandwagonsCard bootstrapElements={bootstrapElements} />
                  <InjuriesSuspensionsCard />
                  <FixturesSnapshot fixtures={mockFixturesSnapshot} />
                </div>
              </div>
            </div>
          ) : (
            <div className="flex flex-col gap-6">
              <SandboxControls
                sandboxMode={sandboxMode}
                onToggleSandboxMode={() => setSandboxMode(!sandboxMode)}
                onUndo={handleUndo}
                onReset={handleReset}
                onApply={handleApplyToTeam}
                canUndo={sandboxActions.length > 0}
              />

              <DeltaStrip
                realSquad={realSquad}
                sandboxSquad={currentSandboxSquad}
                bank={teamStatus.bank}
                bankDelta={sandboxBankDelta}
                freeTransfers={teamStatus.freeTransfers}
                sandboxTransfersMade={sandboxActions.filter((a) => a.type === 'transfer').length}
              />

              <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
                <div className="col-span-1 xl:col-span-2">
                  <div className="flex flex-col gap-6">
                    <RecommendedTransfersList
                      transfers={hybridRecommendedTransfers}
                      onApplyTransfer={handleTransfer}
                    />
                    <CustomTransferBuilder
                      squad={currentSandboxSquad}
                      bootstrapElements={bootstrapElements}
                      bootstrapTeams={cc.bootstrap?.teams ?? []}
                      lookupPrediction={predictions.lookupPrediction}
                      fixturesByName={predictions.fixturesByName}
                      onTransfer={handleTransfer}
                    />
                    {cc.loading ? loadingCard : !hasLiveMyTeam ? (cc.error ? errorCard : emptyMyTeamCard) : (
                      <PitchCard
                        squad={mappedSandboxSquad}
                        onPlayerClick={handleSandboxPlayerClick}
                        selectedPlayerId={swapSelection}
                        swapHint="Tap another player to swap (bench ↔ starting XI)"
                        onSetCaptain={(p) => {
                          if (p.id == null) return;
                          handleSetCaptain(p.id);
                        }}
                        onSetViceCaptain={(p) => {
                          if (p.id == null) return;
                          handleSetViceCaptain(p.id);
                        }}
                      />
                    )}
                    <SandboxCharts squad={currentSandboxSquad} />
                  </div>
                </div>

                <div className="col-span-1">
                  <div className="flex flex-col gap-6">
                    <ModelComparisonPanel
                      models={modelSources}
                      blendTotal={blendTotal}
                      blendRemaining={blendRemaining}
                      isBlendInvalid={isBlendInvalid}
                      onModelWeightChange={handleModelWeightChange}
                      applyStatus={blendApplyState.phase}
                      statusMessage={blendStatusMessage}
                      canRetry={blendApplyState.retryable}
                      onApply={() => {
                        void applyModelBlend();
                      }}
                    />
                    <AskCopilotChat
                      hybridPayload={completedBlendPayload}
                      blendInput={savedBlendInput}
                      applyPhase={blendApplyState.phase}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </DashboardCard>

      <VideoInsightsStrip videos={mockVideoInsights} />

      <Dialog
        open={optimizationDialogOpen}
        onOpenChange={(open) => {
          if (!open && !optimizationLoading) setOptimizationDialogOpen(false);
        }}
      >
        <DialogContent
          overlayClassName="bg-black/70 backdrop-blur-[4px]"
          className="max-w-[calc(100%-2rem)] gap-0 rounded-lg border border-white/8 bg-slate-900 p-0 sm:max-w-md"
          onInteractOutside={(e) => {
            if (optimizationLoading) e.preventDefault();
          }}
          onEscapeKeyDown={(e) => {
            if (optimizationLoading) e.preventDefault();
          }}
        >
          <DialogHeader className="gap-0 border-b border-white/6 px-6 py-4">
            <DialogTitle className="text-base text-white">Run AIrsenal pipeline</DialogTitle>
          </DialogHeader>
          <div className="px-6 py-4">
            <div className="flex flex-col gap-1">
              <label className="text-sm text-slate-300">Weeks ahead</label>
              <div className="relative w-[140px]">
                <input
                  type="number"
                  min={1}
                  max={38}
                  value={weeksAhead}
                  onChange={(e) => {
                    const n = Number(e.target.value);
                    if (Number.isNaN(n)) return;
                    setWeeksAhead(clampWeeksAhead(n));
                  }}
                  onBlur={() => setWeeksAhead(clampWeeksAhead(weeksAhead))}
                  disabled={optimizationLoading}
                  className="h-8 w-full rounded-md border border-white/8 bg-white/4 px-2 pr-6 text-sm text-white transition-colors hover:border-white/12 focus-visible:border-emerald-400 focus-visible:shadow-[0_0_0_1px_#34d399] focus-visible:outline-none disabled:opacity-50"
                />
                <div className="absolute inset-y-0 right-0 flex w-5 flex-col overflow-hidden rounded-r-md border-l border-white/8">
                  <button
                    type="button"
                    tabIndex={-1}
                    aria-label="Increase"
                    disabled={optimizationLoading}
                    onClick={() => setWeeksAhead(clampWeeksAhead(weeksAhead + 1))}
                    className="flex flex-1 cursor-pointer items-center justify-center border-b border-white/8 text-slate-300 hover:bg-white/6 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    <ChevronUp size={12} />
                  </button>
                  <button
                    type="button"
                    tabIndex={-1}
                    aria-label="Decrease"
                    disabled={optimizationLoading}
                    onClick={() => setWeeksAhead(clampWeeksAhead(weeksAhead - 1))}
                    className="flex flex-1 cursor-pointer items-center justify-center text-slate-300 hover:bg-white/6 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    <ChevronDown size={12} />
                  </button>
                </div>
              </div>
              <p className="text-xs text-slate-500">
                Planning horizon for the run (1–38). Default is 3. Runs update DB →
                predict → optimize → export, so it can take several minutes.
              </p>
            </div>
          </div>
          <DialogFooter className="gap-2 border-t border-white/6 bg-transparent">
            <button
              type="button"
              className={cn(buttonVariants({ variant: 'ghost', size: 'sm' }), 'text-slate-400')}
              onClick={() => setOptimizationDialogOpen(false)}
              disabled={optimizationLoading}
            >
              Cancel
            </button>
            <button
              type="button"
              className={cn(buttonVariants({ size: 'sm' }), 'bg-blue-500 hover:bg-blue-600')}
              onClick={() => void handleConfirmOptimization()}
              disabled={optimizationLoading}
            >
              {optimizationLoading ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Running…
                </>
              ) : (
                'Run pipeline'
              )}
            </button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default CommandCenterPage;
