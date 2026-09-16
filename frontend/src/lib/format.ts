/** Shared formatting helpers for the FPL Copilot UI. */

export const money = (v: number | null | undefined): string =>
  v == null ? '—' : `£${Number(v).toFixed(1)}m`;

export const num = (v: number | null | undefined, d = 0): string =>
  v == null ? '—' : Number(v).toFixed(d);

export const pct = (v: number | null | undefined): string =>
  v == null ? '—' : `${Number(v).toFixed(1)}%`;

export const plus = (v: number): string => (v > 0 ? `+${v}` : `${v}`);

export const sign = (v: number, d = 1): string =>
  `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(d)}`;

export const clamp = (v: number, a: number, b: number): number => Math.max(a, Math.min(b, v));

export const round1 = (v: number): number => Math.round(v * 10) / 10;

export const initials = (name: string): string =>
  name
    .split(' ')
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0])
    .join('')
    .toUpperCase();

export const surname = (name: string): string => name.split(' ').slice(-1)[0] ?? name;

export const norm = (s: string | null | undefined): string =>
  (s ?? '')
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();

export const toNum = (v: unknown, fallback = 0): number => {
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'string' && v.trim() !== '') {
    const n = Number(v);
    if (Number.isFinite(n)) return n;
  }
  return fallback;
};

const parseMs = (value: string | number | null | undefined): number => {
  if (value == null) return Number.POSITIVE_INFINITY;
  if (typeof value === 'number') return value;
  const t = Date.parse(value);
  return Number.isFinite(t) ? t : Number.POSITIVE_INFINITY;
};

export const kick = (value: string | number | null | undefined): string => {
  const t = parseMs(value);
  if (!Number.isFinite(t)) return 'TBC';
  return new Date(t).toLocaleString('en-GB', {
    weekday: 'short',
    day: 'numeric',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
  });
};

export const dayLabel = (value: string | number | null | undefined): string => {
  const t = parseMs(value);
  if (!Number.isFinite(t)) return 'Date TBC';
  return new Date(t).toLocaleDateString('en-GB', {
    weekday: 'long',
    day: 'numeric',
    month: 'long',
  });
};

export const timeLabel = (value: string | number | null | undefined): string => {
  const t = parseMs(value);
  if (!Number.isFinite(t)) return '--:--';
  return new Date(t).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
};

export const countdown = (value: string | number | null | undefined): string => {  const t = parseMs(value);
  if (!Number.isFinite(t)) return '—';
  let s = Math.max(0, Math.floor((t - Date.now()) / 1000));
  const d = Math.floor(s / 86400);
  s -= d * 86400;
  const h = Math.floor(s / 3600);
  s -= h * 3600;
  const m = Math.floor(s / 60);
  s -= m * 60;
  return `${d > 0 ? `${d}d ` : ''}${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
};

/** Compact large counts: 2223000 → "2.22m", 314000 → "314k". */
export const compactCount = (n: number): string => {
  const abs = Math.abs(n);
  if (abs >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}m`;
  if (abs >= 1_000) return `${Math.round(n / 1_000)}k`;
  return String(n);
};

export const titleCase = (s: string): string =>  s.replace(/\w\S*/g, (w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase());

/** True when a deadline is within `withinMs` of now. */
export const isDeadlineUrgent = (
  value: string | number | null | undefined,
  withinMs = 2 * 86400000,
): boolean => {
  const t = parseMs(value);
  return Number.isFinite(t) && t - Date.now() < withinMs;
};
