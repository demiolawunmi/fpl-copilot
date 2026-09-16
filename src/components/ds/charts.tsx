export interface BarItem {
  k: string;
  v: number;
  label?: string;
  color?: string;
}

export function Bars({
  items,
  max,
  highlight,
}: {
  items: BarItem[];
  max?: number;
  highlight?: boolean;
}) {
  const top = max ?? Math.max(...items.map((i) => i.v), 1);
  return (
    <div className="bars">
      {items.map((it, i) => (
        <div className="bar-row" key={`${it.k}-${i}`}>
          <span className={`small ${i === 0 && highlight ? 'pos' : 'muted'}`}>{it.k}</span>
          <span className="bar-track">
            <span
              className="bar"
              style={{
                ['--w' as string]: `${((it.v / top) * 100).toFixed(1)}%`,
                ['--i' as string]: i,
                background: it.color ?? 'var(--accent)',
              }}
            />
          </span>
          <span className="small mono" style={{ textAlign: 'right' }}>
            {it.label ?? it.v}
          </span>
        </div>
      ))}
    </div>
  );
}

export interface ColumnItem {
  k: string;
  v: number;
  color?: string;
}

export function ColChart({ items, max }: { items: ColumnItem[]; max?: number }) {
  const top = max ?? Math.max(...items.map((i) => i.v), 1);
  return (
    <div className="col-chart">
      {items.map((it, i) => (
        <div className="col" key={`${it.k}-${i}`}>
          <span className="small mono">{it.v}</span>
          <span
            className="bar-v"
            style={{
              ['--w' as string]: `${((it.v / top) * 100).toFixed(1)}%`,
              ['--i' as string]: i,
              background: it.color ?? 'var(--accent)',
            }}
          />
          <span className="tiny muted">{it.k}</span>
        </div>
      ))}
    </div>
  );
}

export function LineTrend({ points, labels }: { points: number[]; labels?: (string | number)[] }) {
  const w = 260;
  const h = 84;
  const pad = 8;
  const max = Math.max(...points, 1);
  const n = points.length;
  const step = n > 1 ? (w - pad * 2) / (n - 1) : 0;
  const xAt = (i: number) => (n > 1 ? pad + i * step : w / 2);
  const yAt = (v: number) => h - pad - (v / max) * (h - pad * 2);
  const d = points.map((v, i) => `${i ? 'L' : 'M'}${xAt(i).toFixed(1)} ${yAt(v).toFixed(1)}`).join(' ');
  return (
    <div>
      <svg
        viewBox={`0 0 ${w} ${h}`}
        width="100%"
        height={84}
        preserveAspectRatio="none"
        role="img"
        aria-label="Points trend over the last gameweeks"
      >
        <path
          d={d}
          fill="none"
          stroke="var(--accent-2)"
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />
        {points.map((v, i) => {
          const c = v >= 6 ? 'var(--pos)' : v >= 3 ? 'var(--info)' : 'var(--neg)';
          return <circle key={i} cx={xAt(i).toFixed(1)} cy={yAt(v).toFixed(1)} r={3.5} fill={c} />;
        })}
      </svg>
      {labels ? (
        <div style={{ position: 'relative', height: 16, marginTop: 6 }}>
          {labels.map((l, i) => (
            <span
              key={i}
              className="tiny faint"
              style={{
                position: 'absolute',
                left: `${(xAt(i) / w) * 100}%`,
                transform: 'translateX(-50%)',
                whiteSpace: 'nowrap',
              }}
            >
              {l}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}
