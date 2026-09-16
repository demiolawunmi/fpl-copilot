import { backendFetch, extractArrayPayload } from './client';

export interface LatestTransferRow {
  element_in?: number;
  element_out?: number;
  element_in_cost?: number;
  element_out_cost?: number;
  entry?: number;
  event?: number;
  time?: string;
  player_in_id?: number;
  player_out_id?: number;
  in_player_id?: number;
  out_player_id?: number;
  player_in?: string;
  player_out?: string;
  in_name?: string;
  out_name?: string;
  player_in_name?: string;
  player_out_name?: string;
  web_name_in?: string;
  web_name_out?: string;
  in_team?: string;
  out_team?: string;
  player_in_team?: string;
  player_out_team?: string;
  in_team_short_name?: string;
  out_team_short_name?: string;
  player_in_team_short_name?: string;
  player_out_team_short_name?: string;
  in_position?: string;
  out_position?: string;
  player_in_position?: string;
  player_out_position?: string;
  in_price?: number;
  out_price?: number;
  player_in_price?: number;
  player_out_price?: number;
  xp_gain?: number;
  xp_delta?: number;
  xpts_delta?: number;
  xPtsDelta?: number;
  expected_points_delta?: number;
  rationale?: string;
  reason?: string;
  why?: string;
  summary?: string;
}

export type TransfersLatestResponse = LatestTransferRow[];

export async function getTransfersLatest(): Promise<TransfersLatestResponse> {
  const payload = await backendFetch<unknown>('/api/files/transfers_latest');
  return extractArrayPayload<LatestTransferRow>(payload, [
    'data',
    'items',
    'results',
    'transfers',
    'recommendations',
  ]);
}

