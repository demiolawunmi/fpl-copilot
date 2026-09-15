import { debugLog } from "./debug";

/** Statuses that are FPL-side transient throttling/outages worth retrying. */
const RETRYABLE_STATUSES = new Set([429, 500, 502, 503, 504]);

const RETRY_DELAYS_MS = [400, 1_000, 2_500];

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export async function fetchJson<T = unknown>(url: string): Promise<T> {
    for (let attempt = 0; ; attempt++) {
        debugLog("[FPL] GET", url, `attempt ${attempt + 1}`);

        let res: Response;
        try {
            res = await fetch(url, {
                method: "GET",
                headers: { Accept: "application/json" },
            });
        } catch (err) {
            // Network failure – retry like a transient error, then rethrow.
            if (attempt < RETRY_DELAYS_MS.length) {
                debugLog("[FPL] NETWORK RETRY", url, attempt + 1);
                await sleep(RETRY_DELAYS_MS[attempt]);
                continue;
            }
            throw err;
        }

        if (res.ok) {
            const data = (await res.json()) as T;
            debugLog("[FPL] OK", url, data);
            return data;
        }

        const text = await res.text().catch(() => "");
        if (RETRYABLE_STATUSES.has(res.status) && attempt < RETRY_DELAYS_MS.length) {
            debugLog("[FPL] RETRY", res.status, url, `attempt ${attempt + 2}`);
            await sleep(RETRY_DELAYS_MS[attempt]);
            continue;
        }

        debugLog("[FPL] ERROR", res.status, text);
        const hint =
            res.status === 503 || res.status === 429
                ? " (FPL is rate-limiting or busy — try again shortly)"
                : "";
        throw new Error(
            `FPL request failed: ${res.status} ${res.statusText}${hint}`,
        );
    }
}
