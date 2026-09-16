import { useState, type CSSProperties, type ReactNode } from 'react';
import { Icon, type IconName } from '../Icon';
import { initials } from '../../lib/format';
import { fplEndpoints } from '../../api/fpl/endpoints';
import type { Club, Player } from '../../domain/types';

/* ------------------------------------------------------------------- badges */

type Tone = 'pos' | 'neg' | 'warn' | 'info' | 'accent' | '';

export function Badge({
  tone = '',
  children,
  className = '',
  title,
  style,
}: {
  tone?: Tone;
  children: ReactNode;
  className?: string;
  title?: string;
  style?: CSSProperties;
}) {
  return (
    <span className={`badge ${tone} ${className}`.trim()} title={title} style={style}>
      {children}
    </span>
  );
}

export function Chip({
  pressed,
  children,
  className = '',
  ...rest
}: {
  pressed?: boolean;
  children: ReactNode;
  className?: string;
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button className={`chip ${className}`.trim()} aria-pressed={pressed} {...rest}>
      {children}
    </button>
  );
}

export function Fdr({ value, title }: { value: number | null | undefined; title?: string }) {
  const k = Math.max(1, Math.min(5, Math.round(value ?? 3)));
  return (
    <span className={`fdr fdr-${k}`} title={title ?? `Fixture difficulty ${k} of 5`}>
      {k}
    </span>
  );
}

export function FdrRail({ values, titles }: { values: (number | null)[]; titles?: string[] }) {
  if (!values.length) return <span className="faint tiny">No fixtures</span>;
  return (
    <span className="fdr-rail">
      {values.map((v, i) => {
        const k = Math.max(1, Math.min(5, Math.round(v ?? 3)));
        return (
          <span key={i} className={`cell fdr-${k}`} title={titles?.[i]}>
            {k}
          </span>
        );
      })}
    </span>
  );
}

const POS_LABEL: Record<string, string> = {
  GK: 'Goalkeeper',
  DEF: 'Defender',
  MID: 'Midfielder',
  FWD: 'Forward',
};

export function PosBadge({ pos }: { pos: string }) {
  return (
    <span className={`pos-badge pos-${pos}`} title={POS_LABEL[pos] ?? pos}>
      {pos}
    </span>
  );
}

export function StatusBadge({ player }: { player: Player }) {
  if (player.status === 'a') {
    return (
      <Badge tone="pos">
        <Icon name="check" size={12} /> Available
      </Badge>
    );
  }
  const tone: Tone = player.status === 'd' ? 'warn' : 'neg';
  const label = player.status === 'd' ? 'Doubtful' : player.status === 's' ? 'Suspended' : 'Injured';
  return (
    <Badge tone={tone}>
      {label}
      {player.status === 'd' ? ` · ${player.chance}%` : ''}
    </Badge>
  );
}

/* -------------------------------------------------------------- avatars/crest */

export function Avatar({
  player,
  size = 40,
  className = '',
}: {
  player: Pick<Player, 'name' | 'code'>;
  size?: number;
  className?: string;
}) {
  const [failed, setFailed] = useState(false);
  const showImg = !failed && player.code > 0;
  return (
    <span
      className={`headshot ${className}`.trim()}
      style={{ width: size, height: size, fontSize: Math.max(9, size * 0.32) }}
      title={player.name}
    >
      {showImg ? (
        <img
          src={fplEndpoints.playerPhoto(player.code, '110x140')}
          alt={player.name}
          loading="lazy"
          style={{ objectPosition: 'top' }}
          onError={() => setFailed(true)}
        />
      ) : (
        initials(player.name)
      )}
    </span>
  );
}

export function Crest({ club, size = 26 }: { club: Club; size?: number }) {
  const [failed, setFailed] = useState(false);
  const showImg = !failed && club.code > 0;
  return (
    <span
      className="crest"
      style={{
        width: size,
        height: size,
        background: showImg ? 'var(--surface-2)' : club.color,
        fontSize: Math.max(8, size * 0.34),
        padding: showImg ? Math.max(2, size * 0.1) : 0,
      }}
      title={club.name}
    >
      {showImg ? (
        <img
          src={fplEndpoints.teamBadge(club.code)}
          alt={club.name}
          loading="lazy"
          onError={() => setFailed(true)}
        />
      ) : (
        club.short
      )}
    </span>
  );
}

/* ---------------------------------------------------------------- feedback */

export function Empty({
  icon = 'search',
  title,
  text,
  action,
}: {
  icon?: IconName;
  title: string;
  text?: string;
  action?: ReactNode;
}) {
  return (
    <div className="empty">
      <Icon name={icon} size={44} className="ico" />
      <div>
        <b>{title}</b>
      </div>
      {text ? <p className="small">{text}</p> : null}
      {action}
    </div>
  );
}

export function Skeleton({
  height = 16,
  width,
  className = '',
  style,
}: {
  height?: number | string;
  width?: number | string;
  className?: string;
  style?: CSSProperties;
}) {
  return (
    <div
      className={`skel ${className}`.trim()}
      style={{ height, width: width ?? '100%', ...style }}
    />
  );
}

export function SkeletonLines({ count = 6 }: { count?: number }) {
  return (
    <div>
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="skel skel-line" style={{ width: `${60 + ((i * 13) % 35)}%` }} />
      ))}
    </div>
  );
}

export function Progress({ value, color }: { value: number; color?: string }) {
  return (
    <div className="progress">
      <i style={{ width: `${Math.max(0, Math.min(100, value))}%`, background: color }} />
    </div>
  );
}

export function Alert({
  tone = '',
  icon = 'info',
  children,
  actions,
  className = '',
}: {
  tone?: Tone;
  icon?: IconName;
  children: ReactNode;
  actions?: ReactNode;
  className?: string;
}) {
  return (
    <div className={`alert ${tone} ${className}`.trim()} role="status">
      <span className="ico">
        <Icon name={icon} size={20} />
      </span>
      <div className="od-fill">{children}</div>
      {actions ? <span className="act">{actions}</span> : null}
    </div>
  );
}

/* ------------------------------------------------------------------ tooltip */

export function Tooltip({ tip, children }: { tip: ReactNode; children: ReactNode }) {
  return (
    <span className="tooltip-host">
      {children}
      <span className="tip">{tip}</span>
    </span>
  );
}

export function InfoTip({ text, label = 'More information' }: { text: string; label?: string }) {
  return (
    <Tooltip tip={text}>
      <button className="btn btn-icon btn-ghost btn-sm" type="button" aria-label={label}>
        <Icon name="info" size={14} />
      </button>
    </Tooltip>
  );
}

/* ----------------------------------------------------------------- segmented */

export function Segmented<T extends string>({
  options,
  value,
  onChange,
  ariaLabel,
}: {
  options: { value: T; label: ReactNode }[];
  value: T;
  onChange: (value: T) => void;
  ariaLabel?: string;
}) {
  return (
    <span className="segmented" role="group" aria-label={ariaLabel}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          aria-pressed={value === o.value}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </span>
  );
}
