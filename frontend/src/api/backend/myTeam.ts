/**
 * /api/fpl/my-team – authenticated endpoint from our backend.
 *
 * Returns the manager's current squad picks, available chips,
 * and transfer budget info so the Command Center can work with
 * real data instead of mocks.
 */
import { backendFetch } from "./client";

// ── Response types ──

export interface MyTeamPick {
  element: number;        // FPL player id
  position: number;       // 1-15 (starters + bench)
  multiplier: number;     // 0 = bench, 1 = normal, 2 = captain, 3 = TC
  is_captain: boolean;
  is_vice_captain: boolean;
  element_type: number;   // 1 GK, 2 DEF, 3 MID, 4 FWD
  selling_price: number;  // in tenths (e.g. 50 = £5.0m)
  purchase_price: number;
}

export interface MyTeamChip {
  id: number;
  status_for_entry: "available" | "played" | "unavailable";
  played_by_entry: number[];
  name: "wildcard" | "freehit" | "bboost" | "3xc";
  number: number;
  start_event: number;
  stop_event: number;
  chip_type: "transfer" | "team";
  is_pending: boolean;
}

export interface MyTeamTransfers {
  cost: number;    // total hit cost so far
  status: string;  // e.g. "cost"
  limit: number;   // free-transfer cap this GW
  made: number;    // transfers already made
  bank: number;    // bank balance in tenths (e.g. 3 = £0.3m)
  value: number;   // squad value in tenths
}

export interface MyTeamResponse {
  picks: MyTeamPick[];
  picks_last_updated: string;
  chips: MyTeamChip[];
  transfers: MyTeamTransfers;
}

// ── API call ──

export async function getMyTeam(): Promise<MyTeamResponse> {
  return backendFetch<MyTeamResponse>("/api/files/my_team");
}

export interface RefreshTeamResponse {
  ok: boolean;
  my_team_mtime: number | null;
  stdout?: string;
}

/**
 * Re-pull the authenticated squad from the official FPL API (backend runs the
 * private FPL adapter and rewrites my_team.json). Use after changing your team
 * on the official platform, then re-fetch `getMyTeam()`.
 */
export async function refreshMyTeam(): Promise<RefreshTeamResponse> {
  return backendFetch<RefreshTeamResponse>("/api/fpl/refresh-team", { method: "POST" });
}


