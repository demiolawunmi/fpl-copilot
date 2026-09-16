/**
 * Generic fetch wrapper that talks to our FastAPI backend.
 *
 * In dev the Vite proxy rewrites `/api/…` → `http://localhost:8000/api/…`
 * so we can simply `fetch("/api/…")`.
 *
 * In production we fall back to VITE_API_BASE_URL if it's set, otherwise "/api".
 */
import { debugLog } from "../fpl/debug";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

export interface ApiError {
  name: "ApiError";
  message: string;
  status: number;
  statusText: string;
  method: string;
  path: string;
  details?: unknown;
  bodyText?: string;
}

function buildApiError(params: {
  status: number;
  statusText: string;
  method: string;
  path: string;
  details?: unknown;
  bodyText?: string;
}): ApiError {
  const hint = params.bodyText ? ` — ${params.bodyText}` : "";
  return {
    name: "ApiError",
    message: `Backend request failed: ${params.status} ${params.statusText}${hint}`,
    status: params.status,
    statusText: params.statusText,
    method: params.method,
    path: params.path,
    details: params.details,
    bodyText: params.bodyText,
  };
}

export function isApiError(error: unknown): error is ApiError {
  return (
    typeof error === "object" &&
    error !== null &&
    "name" in error &&
    (error as { name?: unknown }).name === "ApiError" &&
    "status" in error
  );
}

export async function backendFetch<T = unknown>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  // In dev with the Vite proxy we can use a relative path.
  // In production builds (or if the env var is set) we prepend the base URL.
  const url = `${API_BASE}${path}`;
  const method = init?.method ?? "GET";
  debugLog(`[BACKEND] ${method}`, url);

  const res = await fetch(url, {
    cache: init?.cache ?? 'no-store',
    ...init,
    headers: {
      Accept: 'application/json',
      ...init?.headers,
    },
  });

  if (!res.ok) {
    const rawText = await res.text().catch(() => "");
    const bodyText = rawText.length > 500 ? `${rawText.slice(0, 500)}...` : rawText;
    let details: unknown;

    try {
      details = rawText ? JSON.parse(rawText) : undefined;
    } catch {
      details = undefined;
    }

    const apiError = buildApiError({
      status: res.status,
      statusText: res.statusText,
      method,
      path,
      details,
      bodyText: bodyText || undefined,
    });

    debugLog("[BACKEND] ERROR", apiError.status, apiError);
    throw apiError;
  }

  const data = (await res.json()) as T;
  debugLog("[BACKEND] OK", url, data);
  return data;
}

export function extractArrayPayload<T>(payload: unknown, preferredKeys: string[] = []): T[] {
  if (Array.isArray(payload)) {
    return payload as T[];
  }

  if (payload && typeof payload === 'object') {
    const record = payload as Record<string, unknown>;

    for (const key of preferredKeys) {
      const value = record[key];
      if (Array.isArray(value)) {
        return value as T[];
      }
    }

    for (const value of Object.values(record)) {
      if (Array.isArray(value)) {
        return value as T[];
      }
    }
  }

  return [];
}
