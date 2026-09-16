/**
 * Bandwagons API - most transferred players
 */
import { backendFetch, extractArrayPayload } from "./client";

// ── Bandwagons Types ──

export interface BandwagonPlayer {
  player_id?: number;
  element?: number;
  code?: number;
  player_name?: string;
  name?: string;
  web_name?: string;
  team?: string;
  team_short_name?: string;
  team_name?: string;
  transfers_in?: number;
  transfers_out?: number;
  transfers_balance?: number;
  photo_url?: string;
}

export type BandwagonsResponse = BandwagonPlayer[];

// ── API Call ──

/**
 * Get most transferred players (bandwagons)
 */
export async function getBandwagons(): Promise<BandwagonsResponse> {
  const payload = await backendFetch<unknown>("/api/files/bandwagons");
  return extractArrayPayload<BandwagonPlayer>(payload, [
    "data",
    "items",
    "results",
    "players",
    "bandwagons",
  ]);
}
