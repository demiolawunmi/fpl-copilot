/** Tiny in-memory TTL cache for expensive, mostly-static FPL payloads. */

interface CacheEntry<T> {
  at: number;
  value: Promise<T>;
}

const store = new Map<string, CacheEntry<unknown>>();

export function cached<T>(key: string, ttlMs: number, fn: () => Promise<T>): Promise<T> {
  const now = Date.now();
  const hit = store.get(key);
  if (hit && now - hit.at < ttlMs) {
    return hit.value as Promise<T>;
  }
  const value = fn().catch((err: unknown) => {
    store.delete(key);
    throw err;
  });
  store.set(key, { at: now, value });
  return value;
}

/** Drop cached entries (all, or those whose key starts with `prefix`). */
export function clearCache(prefix?: string): void {
  if (!prefix) {
    store.clear();
    return;
  }
  for (const key of [...store.keys()]) {
    if (key.startsWith(prefix)) store.delete(key);
  }
}
