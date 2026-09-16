import type { CSSProperties, ReactNode } from 'react';

export function Card({
  title,
  actions,
  children,
  className = '',
  bodyClassName = '',
  id,
  style,
  reveal,
  index,
}: {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
  id?: string;
  style?: CSSProperties;
  reveal?: boolean;
  index?: number;
}) {
  const cls = ['card', reveal ? 'reveal' : '', className].filter(Boolean).join(' ');
  const mergedStyle: CSSProperties | undefined =
    index != null ? { ['--i' as string]: index, ...style } : style;
  return (
    <section className={cls} id={id} style={mergedStyle}>
      {title != null ? (
        <header className="card-head">
          <h3>{title}</h3>
          <span className="spacer" />
          {actions}
        </header>
      ) : null}
      <div className={`card-body ${bodyClassName}`.trim()}>{children}</div>
    </section>
  );
}

export function StatTile({
  label,
  value,
  detail,
  valueClassName = '',
  index,
  count,
  decimals,
  prefix,
  suffix,
  separator,
}: {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
  valueClassName?: string;
  index?: number;
  count?: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  separator?: boolean;
}) {
  return (
    <div className="stat-tile reveal" style={index != null ? { ['--i' as string]: index } : undefined}>
      <span className="k">{label}</span>
      <span
        className={`v ${valueClassName} count`.trim()}
        data-count={count != null ? count : undefined}
        data-dec={decimals}
        data-prefix={prefix}
        data-suffix={suffix}
        data-sep={separator ? '1' : undefined}
      >
        {value}
      </span>
      {detail != null ? <span className="d">{detail}</span> : null}
    </div>
  );
}
