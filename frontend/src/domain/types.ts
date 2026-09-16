/** Normalized domain models for the FPL Copilot UI. */

export type Position = 'GK' | 'DEF' | 'MID' | 'FWD';
export type PlayerStatus = 'a' | 'd' | 'i' | 's';

export interface Club {
  id: number;
  code: number;
  name: string;
  short: string;
  color: string;
}

export interface Last4 {
  points: number;
  minutes: number;
  xgi: number;
  xgc: number;
}

export interface Player {
  id: number;
  code: number;
  name: string;
  fullName: string;
  pos: Position;
  teamId: number;
  club: Club;
  price: number;
  own: number;
  mins: number;
  pts: number;
  form: number;
  status: PlayerStatus;
  chance: number;
  news: string;
  /** Backend absence classification, e.g. injury / suspension / loan_out. */
  absenceType: string | null;
  g: number;
  a: number;
  xg: number;
  xa: number;
  xgi: number;
  cs: number;
  saves: number;
  saves90: number;
  gc: number;
  penaltiesSaved: number;
  bonus: number;
  threat: number;
  creativity: number;
  ict: number;
  xgc: number;
  saveRate: number;
  ppg: number;
  pts90: number;
  g90: number;
  a90: number;
  xg90: number;
  xa90: number;
  xgi90: number;
  threat90: number;
  creat90: number;
  /** Projected points for the upcoming gameweek. */
  xpts: number;
  /** Source of the xPts projection. */
  xptsSource: 'airsenal' | 'fpl' | 'estimate';
  last4: Last4 | null;
  transfersIn: number;
  transfersOut: number;
  transfersNet: number;
}

export interface FixtureLite {
  id: number;
  event: number | null;
  teamH: number;
  teamA: number;
  kickoff: string | null;
  finished: boolean;
  homeScore: number | null;
  awayScore: number | null;
  difficultyH: number | null;
  difficultyA: number | null;
}

export interface TeamGwFixture {
  gw: number;
  opp: Club;
  home: boolean;
  officialFdr: number;
  kickoff: string | null;
  finished: boolean;
}

export interface Outlook {
  label: string;
  tone: 'pos' | 'info' | 'warn' | 'neg';
}

export interface SquadPick {
  element: number;
  position: number;
  multiplier: number;
  isCaptain: boolean;
  isViceCaptain: boolean;
  elementType: number;
  sellingPrice: number;
  purchasePrice: number;
}

export interface ManagerTeam {
  name: string;
  manager: string;
  teamId: string;
  overallRank: number;
  overallRankDelta: number;
  gwPoints: number;
  gwRank: number;
  average: number;
  highest: number;
  bank: number;
  squadValue: number;
  freeTransfers: number;
  chips: ChipInfo[];
  picks: SquadPick[];
  lastUpdated: string | null;
}

export interface ChipInfo {
  key: 'WC' | 'FH' | 'BB' | 'TC';
  name: string;
  used: boolean;
  usedGw?: number;
  note: string;
}

export interface GameweekMeta {
  id: number;
  name: string;
  deadline: string | null;
  finished: boolean;
  isCurrent: boolean;
  isNext: boolean;
  average: number;
  highest: number;
}
