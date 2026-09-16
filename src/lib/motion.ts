/** Count-up animation for stat elements, matching the design system's motion language. */

export const prefersReducedMotion = (): boolean =>
  typeof window !== 'undefined' &&
  window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function format(
  value: number,
  decimals: number,
  separator: boolean,
  prefix: string,
  suffix: string,
): string {
  const body = separator
    ? Number(value).toLocaleString('en-GB', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
      })
    : Number(value).toFixed(decimals);
  return `${prefix}${body}${suffix}`;
}

export function animateCounts(root: Document | HTMLElement = document): void {
  const nodes = root.querySelectorAll<HTMLElement>('[data-count]');
  nodes.forEach((el) => {
    const raw = el.getAttribute('data-count');
    if (raw == null || raw === '') return;
    const target = Number.parseFloat(raw);
    if (!Number.isFinite(target)) return;

    const decimals = Number.parseInt(el.getAttribute('data-dec') ?? '0', 10);
    const separator = el.getAttribute('data-sep') === '1';
    const prefix = el.getAttribute('data-prefix') ?? '';
    const suffix = el.getAttribute('data-suffix') ?? '';

    if (prefersReducedMotion()) {
      el.textContent = format(target, decimals, separator, prefix, suffix);
      return;
    }

    const start = performance.now();
    const duration = 620;
    const frame = (now: number) => {
      const k = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - k, 3);
      el.textContent = format(target * eased, decimals, separator, prefix, suffix);
      if (k < 1) requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  });
}
