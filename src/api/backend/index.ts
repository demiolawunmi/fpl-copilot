export { backendFetch, isApiError } from "./client";
export type { ApiError } from "./client";
export { getMyTeam } from "./myTeam";
export type {
  MyTeamResponse,
  MyTeamPick,
  MyTeamChip,
  MyTeamTransfers,
} from "./myTeam";

// AIrsenal endpoints
export {
  getFormLast4,
  getPredictions,
  getFixturesByPlayer,
  runAirsenal,
} from "./airsenal";
export type {
  FormLast4Player,
  FormLast4Response,
  PredictionPlayer,
  PredictionsResponse,
  PlayerFixture,
  FixturesByPlayerResponse,
  AirsenalRunAction,
  AirsenalRunRequest,
  AirsenalRunResponse,
} from "./airsenal";

// Bandwagons endpoint
export { getBandwagons } from "./bandwagons";
export { getSeasonStatus } from "./seasonStatus";
export type { SeasonStatus } from "./seasonStatus";
export type { BandwagonPlayer, BandwagonsResponse } from "./bandwagons";

// Injury News endpoint
export { getInjuryNews } from "./injuryNews";
export type { InjuryNewsPlayer, InjuryNewsResponse } from "./injuryNews";

// FDR + ClubElo (copilot backend)
export { getFdrEloSnapshot, getTeamFdrFixtures } from "./fdr";
export type {
  FdrEloSnapshot,
  FdrEloRatingRow,
  TeamFdrFixtureItem,
  FdrFixtureSaturated,
  FdrFixtureMetrics,
} from "./fdr";

export {
  submitCopilotBlendJob,
  pollCopilotBlendJob,
  getCopilotBlendJobStatus,
  getCopilotBlendJobResult,
  copilotBlendSnapshotPath,
  copilotBlendSnapshotGlobalPath,
  getCopilotBlendSnapshot,
  getCopilotBlendSnapshotGlobal,
  postCopilotChat,
} from "./blendJobs";
export type {
  CopilotBlendJobStatus,
  CopilotSourceWeights,
  CopilotBlendSubmitRequest,
  CopilotBlendSubmitAcceptedResponse,
  CopilotHybridCore,
  CopilotTransferPlayerRef,
  CopilotRecommendedTransfer,
  CopilotAskCopilotResponse,
  CopilotDegradedMode,
  CopilotHybridResultPayload,
  CopilotErrorField,
  CopilotErrorDetail,
  CopilotErrorResponse,
  CopilotBlendJobStatusResponse,
  CopilotBlendSnapshot,
  CopilotChatTurn,
  CopilotChatRequest,
  CopilotChatResponse,
} from "./blendJobs";
