export const FPL_DEBUG = import.meta.env.VITE_FPL_DEBUG === "true";

export function debugLog(...args: unknown[]) {
    if (FPL_DEBUG) console.info(...args);
}
