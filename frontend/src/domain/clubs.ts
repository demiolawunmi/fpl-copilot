/** Brand colours for the 2026/27 Premier League clubs, keyed by short code. */

export const CLUB_COLORS: Record<string, string> = {
  ARS: '#EF0107',
  AVL: '#670E36',
  BOU: '#DA291C',
  BRE: '#E30613',
  BHA: '#0057B8',
  CHE: '#034694',
  COV: '#6BCFF1',
  CRY: '#1B458F',
  EVE: '#003399',
  FUL: '#CC0000',
  HUL: '#F18A00',
  IPS: '#3A64A3',
  LEE: '#1D428A',
  LIV: '#C8102E',
  MCI: '#6CABDD',
  MUN: '#DA291C',
  NEW: '#241F20',
  NFO: '#DD0000',
  SUN: '#EB172B',
  TOT: '#132257',
  WHU: '#7A263A',
  WOL: '#FDB913',
};

export const clubColor = (short: string | null | undefined): string =>
  (short && CLUB_COLORS[short]) || '#64748F';
