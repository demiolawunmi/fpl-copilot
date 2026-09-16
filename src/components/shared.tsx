import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Icon } from './Icon';
import { Card } from './ds/Card';
import { Alert, Avatar, Badge, Crest, Fdr, FdrRail, Segmented } from './ds/atoms';
import { useCore } from '../context/CoreContext';
import { dayLabel, money, timeLabel } from '../lib/format';
import { nextFixtures, outlookLabel, teamGwDifficulty } from '../domain/model';

export function FixturePills({
  teamId,
  fromGw,
  count,
  separated,
}: {
  teamId: number;
  fromGw: number;
  count: number;
  separated?: boolean;
}) {
  const core = useCore();
  if (!core.fixtureIndex) return null;
  const list = nextFixtures(core.fixtureIndex, teamId, fromGw, count);
  if (separated) {
    return (
      <span className="fdr-rail fdr-rail-sep">
        {list.map((f, i) => {
          const k = Math.max(1, Math.min(5, f.officialFdr));
          return (
            <span key={i} className="od-row" style={{ gap: 6 }}>
              {i > 0 ? <span className="fdr-sep">,</span> : null}
              <span className={`cell fdr-${k}`} title={`${f.opp.short} (${f.home ? 'H' : 'A'})`}>
                {k}
              </span>
            </span>
          );
        })}
      </span>
    );
  }
  return (
    <FdrRail
      values={list.map((f) => f.officialFdr)}
      titles={list.map((f) => `${f.opp.short} (${f.home ? 'H' : 'A'})`)}
    />
  );
}

export function FixturesCard({ gw }: { gw: number }) {
  const core = useCore();
  const [sort, setSort] = useState<'newest' | 'oldest'>('oldest');
  const list = (core.fixtureIndex?.byGw.get(gw) ?? []).slice();
  list.sort((a, b) => {
    const ta = a.kickoff ? Date.parse(a.kickoff) : 0;
    const tb = b.kickoff ? Date.parse(b.kickoff) : 0;
    return sort === 'oldest' ? ta - tb : tb - ta;
  });

  const groups = new Map<string, typeof list>();
  for (const f of list) {
    const key = dayLabel(f.kickoff);
    const arr = groups.get(key) ?? [];
    arr.push(f);
    groups.set(key, arr);
  }

  const past = gw <= core.currentGW;

  return (
    <Card
      reveal
      title={`GW${gw} fixtures`}
      actions={
        <Segmented
          ariaLabel="Sort fixtures"
          value={sort}
          onChange={setSort}
          options={[
            { value: 'newest', label: 'Newest' },
            { value: 'oldest', label: 'Oldest' },
          ]}
        />
      }
    >
      {list.length ? (
        <div>
          {[...groups.entries()].map(([day, fixtures]) => (
            <div key={day} className="od-stack" style={{ gap: 8, marginBottom: 18 }}>
              <div className="tiny faint" style={{ textTransform: 'uppercase', letterSpacing: '.12em' }}>
                {day}
              </div>
              {fixtures.map((f) => {
                const home = core.clubs.find((c) => c.id === f.teamH);
                const away = core.clubs.find((c) => c.id === f.teamA);
                if (!home || !away) return null;
                return (
                  <div
                    key={f.id}
                    style={{
                      display: 'grid',
                      gridTemplateColumns: '56px minmax(0,1fr) auto minmax(0,1fr) auto',
                      alignItems: 'center',
                      gap: 12,
                    }}
                  >
                    <span className="mono tiny muted">{timeLabel(f.kickoff)}</span>
                    <span className="od-row" style={{ gap: 8, justifyContent: 'flex-end' }}>
                      <span className="small">{home.short}</span>
                      <Crest club={home} size={22} />
                    </span>
                    <span
                      className="badge mono"
                      style={{ minWidth: 56, justifyContent: 'center', flex: 'none', whiteSpace: 'nowrap' }}
                    >
                      {past && f.finished && f.homeScore != null && f.awayScore != null
                        ? `${f.homeScore} – ${f.awayScore}`
                        : 'vs'}
                    </span>
                    <span className="od-row" style={{ gap: 8 }}>
                      <Crest club={away} size={22} />
                      <span className="small">{away.short}</span>
                    </span>
                    <span style={{ justifySelf: 'end' }}>
                      <Fdr value={f.difficultyH} />
                    </span>
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      ) : (
        <div className="empty">
          <Icon name="calendar" size={44} />
          <b>No fixtures scheduled</b>
        </div>
      )}
    </Card>
  );
}

export function InjuriesCard({ limit = 10 }: { limit?: number }) {
  const core = useCore();
  const navigate = useNavigate();
  const rows = core.players
    .filter((p) => p.status !== 'a')
    .filter((p) => p.absenceType !== 'loan_out' && p.absenceType !== 'transfer_out')
    .filter((p) => !/\b(loan|joined|permanently|departed|free agent|signing)\b/i.test(p.news))
    .sort((a, b) => {
      const order = (s: string) => (s === 'i' ? 0 : s === 's' ? 1 : 2);
      return order(a.status) - order(b.status);
    })
    .slice(0, limit);

  const returnText = (status: string) => {
    if (status === 'i') return 'Out';
    if (status === 'd') return 'Assessed before deadline';
    if (status === 's') return 'Back next GW';
    return '—';
  };

  return (
    <Card reveal title="Injuries & suspensions">
      {rows.length ? (
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>Player</th>
                <th>Team</th>
                <th>Status</th>
                <th>Return</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((p) => (
                <tr key={p.id} className="clickable" onClick={() => navigate(`/player/${p.id}?from=gw`)}>
                  <td>
                    <div className="cell-player">
                      <Avatar player={p} size={28} />
                      <div>
                        <div className="nm">{p.name}</div>
                        <div className="meta">
                          {p.club.short} · {p.pos}
                        </div>
                        {p.news ? (
                          <div className="meta od-clamp-2" style={{ maxWidth: '34ch' }}>
                            {p.news}
                          </div>
                        ) : null}
                      </div>
                    </div>
                  </td>
                  <td>
                    <Crest club={p.club} size={22} />
                  </td>
                  <td>
                    {p.status === 'd' ? (
                      <span className="badge warn">Doubtful · {p.chance}%</span>
                    ) : (
                      <span className={`badge ${p.status === 's' ? 'warn' : 'neg'}`}>
                        {p.status === 's' ? 'Suspended' : 'Injured'}
                      </span>
                    )}
                  </td>
                  <td className="small muted">{returnText(p.status)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="empty">
          <Icon name="shieldCheck" size={44} />
          <b>No flagged players</b>
        </div>
      )}
    </Card>
  );
}

export function FixturesSnapshotCard({ teamIds, gw }: { teamIds: number[]; gw: number }) {
  const core = useCore();
  const unique = [...new Set(teamIds)];
  return (
    <Card reveal title="Fixtures snapshot">
      <div className="od-stack" style={{ gap: 10 }}>
        {unique.map((id) => {
          const club = core.clubs.find((c) => c.id === id);
          if (!club) return null;
          const avg = core.fixtureIndex
            ? teamGwDifficulty(core.fixtureIndex, id, gw, 5)
            : 3;
          const outlook = outlookLabel(avg);
          return (
            <div
              key={id}
              style={{ display: 'grid', gridTemplateColumns: '24px 40px 1fr auto', alignItems: 'center', gap: 12 }}
            >
              <Crest club={club} size={22} />
              <span className="small">{club.short}</span>
              <FixturePills teamId={id} fromGw={gw} count={5} />
              <span
                className={`badge ${outlook.tone}`}
                style={{ justifySelf: 'end', minWidth: 96, justifyContent: 'center' }}
              >
                {outlook.label}
              </span>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

export function PlayerCell({
  id,
  onClick,
}: {
  id: number;
  onClick?: () => void;
}) {
  const core = useCore();
  const p = core.playersById.get(id);
  if (!p) return null;
  return (
    <button
      type="button"
      onClick={onClick}
      style={{ background: 'none', border: 0, padding: 0, cursor: onClick ? 'pointer' : 'default', textAlign: 'left' }}
    >
      <div className="cell-player">
        <Avatar player={p} size={28} />
        <Crest club={p.club} size={20} />
        <div>
          <div className="nm">{p.name}</div>
          <div className="meta">
            {p.club.short} · {p.pos} · {money(p.price)}
          </div>
        </div>
      </div>
    </button>
  );
}

export function AiSummaryCard({ title = 'AI Summary' }: { title?: string }) {
  const core = useCore();
  const result = core.blendResult;

  if (!result) {
    return (
      <Card
        reveal
        title={title}
        actions={<Badge tone="info"><Icon name="cpu" size={12} /> AIrsenal + Copilot</Badge>}
      >
        <div className="empty">
          <Icon name="cpu" size={44} />
          <b>No blend has been run yet</b>
          <p className="small">
            Open the Command Center, set your model weights and apply a blend to generate a grounded
            summary of your squad.
          </p>
          <Link className="btn btn-primary btn-sm" to="/command?tab=sandbox">
            <Icon name="sliders" size={14} /> Go to the Sandbox
          </Link>
        </div>
      </Card>
    );
  }

  const confidence = Math.round(result.core.confidence * 100);
  return (
    <Card
      reveal
      title={title}
      actions={
        <>
          <Badge tone="info">
            <Icon name="cpu" size={12} /> AIrsenal + Copilot
          </Badge>
          <Badge className="mono">{confidence}% confidence</Badge>
        </>
      }
    >
      <div className="od-stack" style={{ gap: 14 }}>
        <p style={{ fontWeight: 600 }}>{result.core.summary}</p>
        {result.ask_copilot.answer ? <p className="small muted">{result.ask_copilot.answer}</p> : null}
        <div className="od-stack" style={{ gap: 10 }}>
          {result.ask_copilot.rationale.map((r, i) => (
            <div key={i} className="od-row-top reveal" style={{ ['--i' as string]: i, gap: 10 }}>
              <span style={{ color: 'var(--info)', flex: 'none', marginTop: 2 }}>
                <Icon name="info" size={16} />
              </span>
              <div className="od-fill small">{r}</div>
            </div>
          ))}
        </div>
        {result.degraded_mode.is_degraded ? (
          <Alert tone="warn" icon="alert">
            <b>Degraded mode</b>
            <div className="small muted">{result.degraded_mode.message ?? result.degraded_mode.code}</div>
          </Alert>
        ) : null}
      </div>
    </Card>
  );
}
