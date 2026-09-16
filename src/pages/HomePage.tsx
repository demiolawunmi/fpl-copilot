import { Link } from 'react-router-dom';
import { Icon } from '../components/Icon';
import { Card, StatTile } from '../components/ds/Card';
import { Alert, Avatar, Badge, Crest, Skeleton, SkeletonLines } from '../components/ds/atoms';
import { Bars } from '../components/ds/charts';
import { Countdown } from '../components/ds/Countdown';
import { FixturePills } from '../components/shared';
import { useCore } from '../context/CoreContext';
import { useSquad } from '../context/SquadContext';
import { initials, num, plus } from '../lib/format';
import { xiXpts } from '../domain/projection';
import { outlookLabel, teamGwDifficulty } from '../domain/model';

export default function HomePage() {
  const core = useCore();
  const { xi, captainId, byId } = useSquad();

  if (core.loading) {
    return (
      <div className="od-stack" style={{ gap: 24 }}>
        <Skeleton height={132} />
        <div className="grid grid-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} height={104} />
          ))}
        </div>
        <div className="grid grid-2">
          <Skeleton height={280} />
          <Skeleton height={280} />
        </div>
      </div>
    );
  }

  if (core.error && !core.manager) {
    return (
      <Alert tone="neg" icon="alert" actions={<button className="btn btn-ghost" type="button" onClick={core.refresh}><Icon name="refresh" size={16} /> Retry</button>}>
        <b>We couldn’t load your squad</b>
        <div className="small muted">{core.error}</div>
      </Alert>
    );
  }

  const manager = core.manager;
  const cap = captainId != null ? byId.get(captainId) : xi.length ? byId.get(xi[0]) : undefined;
  const totalXpts = xiXpts(xi, byId, captainId);
  const rec = core.recommendedTransfers[0];
  const deadline = core.nextDeadline;

  const squadForm = xi
    .map((id) => byId.get(id))
    .filter((p): p is NonNullable<typeof p> => Boolean(p))
    .map((p) => ({
      k: p.name.split(' ').slice(-1)[0],
      v: p.last4?.points ?? Math.round(p.form * 4),
      label: `${p.last4?.points ?? Math.round(p.form * 4)} pts`,
      color: p.id === captainId ? 'var(--accent)' : 'var(--accent-2)',
    }))
    .sort((a, b) => b.v - a.v)
    .slice(0, 6);

  const clubsAhead = core.clubs
    .map((c) => ({
      club: c,
      avg: core.fixtureIndex ? teamGwDifficulty(core.fixtureIndex, c.id, core.nextGW, 5) : 3,
    }))
    .sort((a, b) => a.avg - b.avg)
    .slice(0, 8);

  return (
    <div className="od-stack page-enter" style={{ gap: 24 }}>
      <div className="od-row" style={{ gap: 16, alignItems: 'flex-end', flexWrap: 'wrap' }}>
        <div>
          <div className="tiny faint" style={{ textTransform: 'uppercase', letterSpacing: '.14em' }}>
            Welcome back
          </div>
          <h2>
            {manager?.manager ?? core.entry?.manager ?? 'Manager'}, your XI is projected for {num(totalXpts, 1)} xPts
            in GW{core.nextGW}.
          </h2>
        </div>
        <span className="spacer" />
        <Link className="btn btn-primary" to="/command">
          <Icon name="target" size={16} /> Open Command Center
        </Link>
      </div>

      <div className="hero-band">
        <Card reveal>
          <div className="od-row" style={{ gap: 20, flexWrap: 'wrap', alignItems: 'flex-start' }}>
            <span className="avatar" style={{ width: 64, height: 64, fontSize: 'var(--fs-xl)' }}>
              {initials(manager?.name ?? 'FC')}
            </span>
            <div className="od-fill">
              <div className="od-row" style={{ gap: 10, flexWrap: 'wrap' }}>
                <h3>{manager?.name ?? core.entry?.name ?? 'My Team'}</h3>
                <Badge className="mono">ID {manager?.teamId ?? '—'}</Badge>
                <Badge tone="accent">GW{core.currentGW}</Badge>
              </div>
              <div className="muted small">{manager?.manager ?? core.entry?.manager ?? '—'}</div>
              <div className="chips-row" style={{ marginTop: 12 }}>
                {(manager?.chips ?? []).map((c) => (
                  <Badge key={c.key} tone={c.used ? '' : 'accent'} title={c.note}>
                    <Icon name={c.used ? 'x' : 'check'} size={12} /> {c.name}
                    {c.used ? ` used GW${c.usedGw ?? ''}` : ''}
                  </Badge>
                ))}
              </div>
            </div>
            <div className="od-stack" style={{ gap: 4, minWidth: 180 }}>
              <span className="k tiny faint" style={{ textTransform: 'uppercase', letterSpacing: '.1em' }}>
                Next deadline
              </span>
              <span className="mono" style={{ fontSize: 'var(--fs-xl)' }}>
                <Countdown target={deadline} />
              </span>
              <span className="tiny muted">GW{core.nextGW}</span>
            </div>
          </div>
        </Card>

        <Card
          reveal
          index={1}
          title="This week’s call"
          actions={<Badge tone="info"><Icon name="cpu" size={12} /> Copilot</Badge>}
        >
          {cap ? (
            <div className="od-stack" style={{ gap: 14 }}>
              <div className="od-row" style={{ gap: 12 }}>
                <Avatar player={cap} size={44} />
                <div className="od-fill">
                  <div style={{ fontWeight: 700 }}>Captain {cap.name}</div>
                  <div className="small muted">
                    {num(cap.xpts * 2, 1)} projected xPts with the armband · <Crest club={cap.club} size={18} />{' '}
                    {cap.club.short}
                  </div>
                </div>
              </div>
              <div className="divider" />
              {rec ? (
                <div className="od-row" style={{ gap: 12 }}>
                  <Badge tone="neg">OUT</Badge>
                  <div className="od-fill small">{rec.out.player_name}</div>
                  <Icon name="swap" size={16} className="muted" />
                  <Badge tone="pos">IN</Badge>
                  <div className="od-fill small">{rec.in.player_name}</div>
                  <Badge tone="accent" className="mono">{plus(Math.round(rec.projected_points_delta * 10) / 10)} xPts</Badge>
                </div>
              ) : (
                <div className="small muted">No recommended transfer yet — build one in the Sandbox.</div>
              )}
              <div className="od-row" style={{ gap: 8, flexWrap: 'wrap' }}>
                <Link className="btn btn-primary btn-sm" to="/command?tab=sandbox">
                  <Icon name="sliders" size={14} /> Try it in the Sandbox
                </Link>
                <Link className="btn btn-ghost btn-sm" to="/gw">
                  <Icon name="calendar" size={14} /> Review GW{core.currentGW}
                </Link>
              </div>
            </div>
          ) : (
            <SkeletonLines count={4} />
          )}
        </Card>
      </div>

      <div className="grid grid-4">
        <StatTile
          label="GW points"
          value={manager?.gwPoints ?? 0}
          count={manager?.gwPoints ?? 0}
          detail={`Average ${manager?.average ?? 0}`}
          valueClassName="pos"
          index={0}
        />
        <StatTile
          label="Overall rank"
          value={(manager?.overallRank ?? 0).toLocaleString('en-GB')}
          count={manager?.overallRank ?? 0}
          separator
          detail={
            manager?.overallRankDelta
              ? `+${Math.abs(manager.overallRankDelta).toLocaleString('en-GB')} places this week`
              : undefined
          }
          valueClassName="pos"
          index={1}
        />
        <StatTile
          label="Squad value"
          value={`£${(manager?.squadValue ?? 0).toFixed(1)}m`}
          count={manager?.squadValue ?? 0}
          decimals={1}
          prefix="£"
          suffix="m"
          detail={`Bank £${(manager?.bank ?? 0).toFixed(1)}m`}
          index={2}
        />
        <StatTile
          label="Free transfers"
          value={manager?.freeTransfers ?? 1}
          count={manager?.freeTransfers ?? 1}
          detail="Next hit costs −4"
          index={3}
        />
      </div>

      <div className="cols-2">
        <Card
          reveal
          title="Squad form"
          actions={<Link className="btn btn-ghost btn-sm" to="/gw">Full gameweek</Link>}
        >
          {squadForm.length ? (
            <Bars items={squadForm} highlight />
          ) : (
            <div className="empty">
              <Icon name="trending" size={44} />
              <b>No form data yet</b>
            </div>
          )}
        </Card>

        <Card reveal title="Fixtures ahead">
          <div className="od-stack" style={{ gap: 10 }}>
            {clubsAhead.map(({ club, avg }) => {
              const outlook = outlookLabel(avg);
              return (
                <div
                  key={club.id}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '26px minmax(120px, 1fr) auto auto',
                    alignItems: 'center',
                    gap: 12,
                  }}
                >
                  <Crest club={club} size={24} />
                  <span className="small">{club.name}</span>
                  <FixturePills teamId={club.id} fromGw={core.nextGW} count={5} separated />
                  <span
                    className={`badge ${outlook.tone}`}
                    style={{ justifySelf: 'end', minWidth: 112, justifyContent: 'center' }}
                  >
                    {outlook.label}
                  </span>
                </div>
              );
            })}
          </div>
          <div className="od-row" style={{ marginTop: 14 }}>
            <Link className="btn btn-ghost btn-sm" to="/fixtures">
              <Icon name="shield" size={14} /> Difficulty intelligence
            </Link>
          </div>
        </Card>
      </div>
    </div>
  );
}
