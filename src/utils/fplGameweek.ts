/**
 * Resolve which bootstrap "event" drives picks, fixtures (?event=gw), and live data.
 *
 * FPL does not always set `is_current` immediately when the round moves forward.
 * If we only fall back to "last finished" GW, we stay one week behind (e.g. stuck on
 * GW30 while GW31 fixtures already exist at /fixtures/?event=31).
 */

export type FplBootstrapEventLike = {
  id: number;
  finished: boolean;
  is_current: boolean;
  is_next: boolean;
  deadline_time?: string;
};

function lastFinishedEvent<T extends FplBootstrapEventLike>(events: T[]): T | undefined {
  const finished = events.filter((e) => e.finished);
  if (!finished.length) return undefined;
  return finished.reduce((a, b) => (a.id >= b.id ? a : b));
}

/** Whether a gameweek's deadline has passed, i.e. the round is underway or already played. */
function hasStarted(event: FplBootstrapEventLike): boolean {
  if (event.finished) return true;
  if (!event.deadline_time) return false;
  return Date.now() >= new Date(event.deadline_time).getTime();
}

/**
 * Prefer the live round, then the upcoming deadline week, then the most recently finished round.
 *
 * If FPL still has `is_current` on a gameweek that is already `finished` while `is_next`
 * points at a later GW whose deadline has passed (e.g. stuck on GW30 while GW31 is
 * underway), prefer `is_next` so `/fixtures/?event=` matches the round actually in play.
 * If the next GW's deadline has not passed yet, keep the current (finished) round so that
 * stats/points for the round just played are shown instead of an empty upcoming one.
 */
export function resolvePrimaryGameweekEvent<T extends FplBootstrapEventLike>(
  events: T[],
): T | undefined {
  const byCurrent = events.find((e) => e.is_current);
  const byNext = events.find((e) => e.is_next);
  const byLastFinished = lastFinishedEvent(events);

  if (byCurrent?.finished && byNext && byNext.id > byCurrent.id && hasStarted(byNext)) {
    return byNext;
  }

  return byCurrent ?? byNext ?? byLastFinished;
}
