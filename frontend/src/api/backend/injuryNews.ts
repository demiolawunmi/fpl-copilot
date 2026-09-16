import { backendFetch, extractArrayPayload } from './client';

export interface InjuryNewsPlayer {
  player_id?: number;
  fpl_api_id?: number;
  id?: number;
  element?: number;
  name?: string;
  player_name?: string;
  web_name?: string;
  team?: string;
  team_code?: string;
  team_short_name?: string;
  team_name?: string;
  chance_next_round?: number | null;
  chance_of_playing_next_round?: number | null;
  chance_this_round?: number | null;
  prob_available?: number | null;
  news?: string;
  source_news?: string;
  status?: string;
  absence_type?: string;
  position?: string;
  position_group?: string;
}

export type InjuryNewsResponse = InjuryNewsPlayer[];

export async function getInjuryNews(): Promise<InjuryNewsResponse> {
  const payload = await backendFetch<unknown>('/api/fdr/injuries');
  return extractArrayPayload<InjuryNewsPlayer>(payload, [
    'injuries',
    'data',
    'items',
    'results',
    'players',
    'injury_news',
    'injuries',
    'news',
  ]);
}

