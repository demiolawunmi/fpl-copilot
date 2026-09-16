import { useEffect, useState } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { Icon } from '../components/Icon';
import { Card, StatTile } from '../components/ds/Card';
import { Alert, Avatar, Badge, Crest, Empty, PosBadge, StatusBadge } from '../components/ds/atoms';
import { LineTrend } from '../components/ds/charts';
import { useCore } from '../context/CoreContext';
import { getElementSummary, type ElementSummaryHistory } from '../api/fpl/fpl';
import { nextFixtures, outlookLabel, teamGwDifficulty } from '../domain/model';
import { kick, money, num } from '../lib/format';

export default function PlayerDetailPage() {
  const { playerId } = useParams();
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const core = useCore();
  const from = params.get('from') ?? 'players';

  const id = Number(playerId);
  const player = Number.isFinite(id) ? core.playersById.get(id) : undefined;

  const [history, setHistory] = useState<ElementSummaryHistory[] | null>(null);

  useEffect(() => {
    if (!player) return;
    let cancelled = false;
    getElementSummary(player.id)
      .then((res) => {
        if (!cancelled) setHistory(res.history);
      })
      .catch(() => {
        /* history optional */
      });
    return () => {
      cancelled = true;
    };
  }, [player]);

  const backTo: Record<string, string> = {
    players: '/players',
    gw: '/gw',
    command: '/command',
    fixtures: '/fixtures',
  };
  const backLabel: Record<string, string> = {
    players: 'Players',
    gw: `GW${core.currentGW}`,
    command: 'Command Center',
    fixtures: 'Fixtures',
  };

  if (core.loading) {
    return <Card><div className="skel" style={{ height: 240 }} /></Card>;
  }

  if (!player) {
    return (
      <Empty
        title="Player not found"
        text={`No player with id ${playerId}.`}
        action={
          <button className="btn btn-primary" type="button" onClick={() => navigate('/players')}>
            Back to players
          </button>
        }
      />
    );
  }

  const fx = core.fixtureIndex ? nextFixtures(core.fixtureIndex, player.teamId, core.nextGW, 5) : [];
  const avg = core.fixtureIndex ? teamGwDifficulty(core.fixtureIndex, player.teamId, core.nextGW, 5) : 3;
  const outlook = outlookLabel(avg);

  const insight = `${player.name} is projected for ${player.xpts.toFixed(1)} xPts in GW${core.nextGW}, with ${player.own.toFixed(1)}% ownership and a ${player.status === 'a' ? 'available' : 'flagged'} status. ${
    avg < 2.8
      ? 'The next five fixtures are among the kindest in the league.'
      : avg < 3.6
        ? 'The next five fixtures are about average difficulty.'
        : 'The next five fixtures are a tough run — manage minutes risk.'
  }`;

  const series = (history ?? [])
    .filter((h) => h.minutes > 0 || h.total_points > 0)
    .slice(-5)
    .map((h) => h.total_points);
  const seriesLabels = (history ?? [])
    .filter((h) => h.minutes > 0 || h.total_points > 0)
    .slice(-5)
    .map((h) => `GW${h.round}`);

  const tiles: { k: string; v: string | number }[] =
    player.pos === 'GK'
      ? [
          { k: 'Saves', v: player.saves },
          { k: 'Clean sheets', v: player.cs },
          { k: 'Save rate', v: `${player.saveRate}%` },
          { k: 'Saves/90', v: player.saves90.toFixed(1) },
          { k: 'xGC', v: player.xgc.toFixed(1) },
        ]
      : player.pos === 'DEF'
        ? [
            { k: 'Clean sheets', v: player.cs },
            { k: 'Bonus', v: player.bonus },
            { k: 'xGI', v: player.xgi.toFixed(1) },
            { k: 'xGI/90', v: player.xgi90.toFixed(1) },
            { k: 'Threat', v: player.threat },
          ]
        : [
            { k: 'Goals', v: player.g },
            { k: 'xG', v: player.xg.toFixed(1) },
            { k: 'Threat', v: player.threat },
            { k: 'Form', v: player.form },
            { k: 'Creativity', v: player.creativity },
          ];

  type Tile = { label: string; value: string | number; count?: number };
  const summaryTiles: Tile[] =
    player.pos === 'GK'
      ? [
          { label: 'Total points', value: player.pts, count: player.pts },
          { label: 'Minutes', value: player.mins.toLocaleString('en-GB'), count: player.mins },
          { label: 'Goals conceded', value: player.gc, count: player.gc },
          { label: 'Penalties saved', value: player.penaltiesSaved, count: player.penaltiesSaved },
        ]
      : player.pos === 'DEF'
        ? [
            { label: 'Total points', value: player.pts, count: player.pts },
            { label: 'Minutes', value: player.mins.toLocaleString('en-GB'), count: player.mins },
            { label: 'Clean sheets', value: player.cs, count: player.cs },
            { label: 'Bonus', value: player.bonus, count: player.bonus },
          ]
        : [
            { label: 'Total points', value: player.pts, count: player.pts },
            { label: 'Minutes', value: player.mins.toLocaleString('en-GB'), count: player.mins },
            { label: 'Goals', value: player.g, count: player.g },
            { label: 'Assists', value: player.a, count: player.a },
          ];

  const extraTiles: Tile[] =
    player.pos === 'GK'
      ? [
          { label: 'Bonus', value: player.bonus, count: player.bonus },
          { label: 'Pts/start', value: player.ppg.toFixed(1) },
          { label: 'Form', value: player.form },
        ]
      : player.pos === 'DEF'
        ? [
            { label: 'xGI', value: player.xgi.toFixed(1) },
            { label: 'xGC', value: player.xgc.toFixed(1) },
            { label: 'Threat', value: player.threat },
          ]
        : [
            { label: 'xG', value: player.xg.toFixed(1) },
            { label: 'xGI', value: player.xgi.toFixed(1) },
            { label: 'Bonus', value: player.bonus, count: player.bonus },
          ];

  return (
    <div className="od-stack page-enter" style={{ gap: 20 }}>
      <div className="od-row" style={{ gap: 10 }}>
        <button className="btn btn-ghost btn-sm" type="button" onClick={() => navigate(backTo[from] ?? '/players')}>
          <Icon name="chevronLeft" size={14} /> {backLabel[from] ?? 'Back'}
        </button>
      </div>

      <Card reveal>
        <div className="od-row-top" style={{ gap: 24, flexWrap: 'wrap' }}>
          <div className="od-row" style={{ gap: 16, alignItems: 'center' }}>
            <Avatar player={player} size={84} />
            <div>
              <div className="od-row" style={{ gap: 10, flexWrap: 'wrap' }}>
                <h2>{player.name}</h2>
                <PosBadge pos={player.pos} />
              </div>
              <div className="od-row" style={{ gap: 8, marginTop: 6, flexWrap: 'wrap' }}>
                <Crest club={player.club} size={26} />
                <span className="muted">{player.club.name}</span>
                <Badge className="mono">{money(player.price)}</Badge>
                <Badge className="mono">{num(player.own, 1)}% owned</Badge>
              </div>
              <div className="od-row" style={{ gap: 8, marginTop: 10, flexWrap: 'wrap' }}>
                <StatusBadge player={player} />
                <Badge tone="info">
                  {player.xptsSource === 'airsenal' ? 'AIrsenal xPts' : player.xptsSource === 'fpl' ? 'FPL xPts' : 'Estimated xPts'}
                </Badge>
              </div>
            </div>
          </div>
          <span className="spacer" />
          <div className="od-stat" style={{ textAlign: 'right' }}>
            <span className="k tiny faint">GW{core.nextGW} projection</span>
            <span
              className="v mono"
              style={{ fontSize: 'var(--fs-3xl)', fontWeight: 700, color: 'var(--accent)' }}
              data-count={player.xpts}
              data-dec="1"
            >
              {player.xpts.toFixed(1)}
            </span>
            <span className="tiny muted">xPts · {player.ppg.toFixed(1)} pts/start</span>
          </div>
        </div>
        <div className="divider" />
        <p className="small">{insight}</p>
        {player.news ? (
          <Alert tone={player.status === 'i' ? 'neg' : 'warn'} icon="alert" className="reveal" >
            <span className="small">
              <b>
                {player.status === 'd' ? 'Doubtful' : player.status === 's' ? 'Suspended' : 'Injured'}
                {player.status === 'd' ? ` · ${player.chance}% chance of playing` : ''}
              </b>
              <div>{player.news}</div>
            </span>
          </Alert>
        ) : null}
      </Card>

      <div className="grid grid-5">
        {tiles.map((t, i) => (
          <StatTile key={t.k} label={t.k} value={t.v} index={i} />
        ))}
      </div>

      <div className="grid grid-4">
        {summaryTiles.map((t, i) => (
          <StatTile key={t.label} label={t.label} value={t.value} count={t.count} index={i} />
        ))}
      </div>

      <div className="grid grid-3">
        {extraTiles.map((t, i) => (
          <StatTile key={t.label} label={t.label} value={t.value} count={t.count} index={i} />
        ))}
      </div>

      <div className="cols-2">
        <Card reveal title="Form trend" actions={<Badge>Last 5 GWs</Badge>}>
          {series.length ? (
            <LineTrend points={series} labels={seriesLabels} />
          ) : (
            <div className="empty">
              <Icon name="trending" size={44} />
              <b>No per-gameweek history yet</b>
            </div>
          )}
        </Card>
        <Card
          reveal
          title="Upcoming fixtures"
          actions={<span className={`badge ${outlook.tone}`}>{outlook.label} · avg {avg.toFixed(1)}</span>}
        >
          {fx.length ? (
            <div className="od-stack" style={{ gap: 8 }}>
              {fx.map((f) => (
                <div className="od-row" key={f.gw} style={{ gap: 12 }}>
                  <span className="mono tiny muted" style={{ width: 44 }}>
                    GW{f.gw}
                  </span>
                  <Crest club={f.opp} size={24} />
                  <span className="od-fill small">{f.opp.name}</span>
                  <Badge>{f.home ? 'Home' : 'Away'}</Badge>
                  <span className="tiny faint">{kick(f.kickoff)}</span>
                  <span className={`fdr fdr-${f.officialFdr}`}>{f.officialFdr}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty">
              <Icon name="calendar" size={44} />
              <b>No upcoming fixtures</b>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
