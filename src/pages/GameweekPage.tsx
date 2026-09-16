import { useState, type ReactNode } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Icon } from '../components/Icon';
import { Card, StatTile } from '../components/ds/Card';
import { Avatar, Badge, Crest, Skeleton, SkeletonLines } from '../components/ds/atoms';
import { Segmented } from '../components/ds/atoms';
import { BenchRow, Pitch, type StatOf } from '../components/ds/Pitch';
import { AiSummaryCard, FixturesCard, InjuriesCard, PlayerCell } from '../components/shared';
import { PlayerActionDialog } from '../components/player/PlayerActionDialog';
import { useCore } from '../context/CoreContext';
import { useSquad } from '../context/SquadContext';
import { useTeamId } from '../context/TeamIdContext';
import { useToast } from '../context/ToastContext';
import { useGameweekPicks } from '../hooks/useGameweekPicks';
import { findPlayerIdByName, nextFixtures } from '../domain/model';
import { kick, money, plus } from '../lib/format';
import type { Player } from '../domain/types';

export default function GameweekPage() {
  const core = useCore();
  const { teamId } = useTeamId();
  const { xi, bench, captainId, viceId, byId } = useSquad();
  const toast = useToast();
  const navigate = useNavigate();
  const [gwOverride, setGwOverride] = useState<number | null>(null);
  const [tab, setTab] = useState<'pitch' | 'table'>('pitch');
  const [dialogPlayer, setDialogPlayer] = useState<Player | null>(null);

  const gw = gwOverride ?? core.currentGW;
  const event = core.gameweeks.find((g) => g.id === gw);
  const myHistory = core.history.find((h) => h.event === gw);
  const manager = core.manager;

  // Actual picks + live points for the selected gameweek (falls back to the
  // current squad when the gameweek hasn't been played).
  const hist = useGameweekPicks(teamId, gw);
  const usingHistory = hist.hasPicks && hist.xi.length > 0;
  const xiIds = usingHistory ? hist.xi : xi;
  const benchIds = usingHistory ? hist.bench : bench;
  const capId = usingHistory ? hist.captainId : captainId;
  const viceIdFinal = usingHistory ? hist.viceId : viceId;

  const statOf: StatOf | undefined = usingHistory
    ? (p, ctx) => {
        const pts = hist.pointsById.get(p.id) ?? 0;
        const mult = hist.multiplierById.get(p.id) ?? 1;
        const value = ctx.bench ? pts : pts * mult;
        return { value, suffix: 'pts', decimals: 0 };
      }
    : undefined;

  const fixtureOf = (p: Player) => {
    if (!core.fixtureIndex) return null;
    return nextFixtures(core.fixtureIndex, p.teamId, gw, 1)[0] ?? null;
  };

  const stats = {
    average: event?.average ?? manager?.average ?? 0,
    highest: event?.highest ?? manager?.highest ?? 0,
    myPoints: myHistory?.points ?? 0,
    myRank: myHistory?.rank ?? 0,
    overallRank: myHistory?.overall_rank ?? manager?.overallRank ?? 0,
  };

  const squad = [...xiIds, ...benchIds];
  const gwPointsOf = (id: number) => {
    const pts = hist.pointsById.get(id) ?? 0;
    const mult = hist.multiplierById.get(id) ?? 1;
    return benchIds.includes(id) ? pts : pts * mult;
  };

  return (
    <div className="od-stack page-enter" style={{ gap: 24 }}>
      <div className="od-row" style={{ gap: 16, flexWrap: 'wrap' }}>
        <div>
          <div className="tiny faint" style={{ textTransform: 'uppercase', letterSpacing: '.14em' }}>
            Review
          </div>
          <h2>Gameweek {gw}</h2>
        </div>
        <span className="spacer" />
        <div className="od-row" style={{ gap: 8 }}>
          <button
            className="btn btn-icon btn-ghost"
            type="button"
            aria-label="Previous gameweek"
            disabled={gw <= 1}
            onClick={() => setGwOverride(Math.max(1, gw - 1))}
          >
            <Icon name="chevronLeft" />
          </button>
          <div className="od-stat" style={{ minWidth: 190, textAlign: 'center' }}>
            <span style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 'var(--fs-lg)' }}>
              Gameweek {gw}
            </span>
            <span className="tiny muted">{event?.deadline ? kick(event.deadline) : ''}</span>
          </div>
          <button
            className="btn btn-icon btn-ghost"
            type="button"
            aria-label="Next gameweek"
            disabled={gw >= (core.gameweeks.length || 38)}
            onClick={() => setGwOverride(Math.min(core.gameweeks.length || 38, gw + 1))}
          >
            <Icon name="chevronRight" />
          </button>
        </div>
      </div>

      <div className="od-row" style={{ gap: 12, flexWrap: 'wrap' }}>
        <Badge className="mono">
          {manager?.name ?? core.entry?.name ?? '—'} • {manager?.manager ?? core.entry?.manager ?? '—'} • ID{' '}
          {manager?.teamId ?? '—'}
        </Badge>
        {event?.deadline ? <Badge>Deadline {kick(event.deadline)}</Badge> : null}
      </div>

      {core.loading ? (
        <div className="grid grid-5">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} height={88} />
          ))}
        </div>
      ) : (
        <div className="stat-strip">
          <StatTile label="Average" value={stats.average} count={stats.average} index={0} />
          <StatTile label="Highest" value={stats.highest} count={stats.highest} index={1} />
          <StatTile
            label="Your points"
            value={stats.myPoints}
            count={stats.myPoints}
            valueClassName={stats.myPoints >= stats.average ? 'pos' : 'neg'}
            detail={stats.myPoints >= stats.average ? 'Above average' : 'Below average'}
            index={2}
          />
          <StatTile
            label="Your GW rank"
            value={stats.myRank.toLocaleString('en-GB')}
            count={stats.myRank}
            separator
            index={3}
          />
          <StatTile
            label="Overall rank"
            value={stats.overallRank.toLocaleString('en-GB')}
            count={stats.overallRank}
            separator
            index={4}
          />
        </div>
      )}

      <div className="cols-3">
        <div className="od-stack" style={{ gap: 16 }}>
          <Card
            reveal
            title={`Squad · ${manager?.name ?? 'My Team'}`}
            actions={
              <Segmented
                ariaLabel="Squad view"
                value={tab}
                onChange={setTab}
                options={[
                  { value: 'pitch', label: <><Icon name="grid" size={14} /> Pitch</> },
                  { value: 'table', label: <><Icon name="table" size={14} /> Table</> },
                ]}
              />
            }
          >
            {core.loading ? (
              <SkeletonLines count={8} />
            ) : tab === 'table' ? (
              <div className="table-wrap">
                <table className="data">
                  <thead>
                    <tr>
                      <th>Player</th>
                      <th>£</th>
                      <th>{usingHistory ? 'GW pts' : 'xPts'}</th>
                      <th>Pts</th>
                      <th>Form</th>
                      <th>Next</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {squad.map((id) => {
                      const p = byId.get(id);
                      if (!p) return null;
                      const f = fixtureOf(p);
                      return (
                        <tr key={id} className="clickable" onClick={() => setDialogPlayer(p)}>
                          <td>
                            <div className="cell-player">
                              <Avatar player={p} size={30} />
                              <div>
                                <div className="nm">
                                  {p.name}
                                  {id === capId ? ' ' : ''}
                                  {id === capId ? <span className="badge accent">C</span> : null}
                                  {id === viceIdFinal ? <span className="badge info">V</span> : null}
                                </div>
                                <div className="meta">{p.club.name}</div>
                              </div>
                            </div>
                          </td>
                          <td className="mono">{money(p.price)}</td>
                          <td className="mono">
                            {usingHistory ? (
                              <span className={gwPointsOf(id) >= 6 ? 'pos' : ''}>{gwPointsOf(id)}</span>
                            ) : (
                              p.xpts.toFixed(1)
                            )}
                          </td>
                          <td className="mono">{p.pts}</td>
                          <td className={`mono ${p.form >= 6 ? 'pos' : ''}`}>{p.form}</td>
                          <td>
                            {f ? (
                              <span className="od-row" style={{ gap: 6 }}>
                                <Crest club={f.opp} size={20} />
                                <span className="tiny muted">{f.home ? 'H' : 'A'}</span>
                                <span className={`fdr fdr-${f.officialFdr}`}>{f.officialFdr}</span>
                              </span>
                            ) : (
                              <span className="faint">—</span>
                            )}
                          </td>
                          <td>
                            <button
                              className="btn btn-ghost btn-sm"
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                navigate(`/player/${p.id}?from=gw`);
                              }}
                            >
                              <Icon name="eye" size={14} /> View
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <Pitch
                ids={xiIds}
                byId={byId}
                captainId={capId}
                viceId={viceIdFinal}
                fixtureOf={fixtureOf}
                statOf={statOf}
                onChipClick={setDialogPlayer}
              />
            )}
            {tab === 'pitch' ? (
              <BenchRow
                ids={benchIds}
                byId={byId}
                captainId={capId}
                viceId={viceIdFinal}
                fixtureOf={fixtureOf}
                statOf={statOf}
                onChipClick={setDialogPlayer}
              />
            ) : null}
          </Card>

          <AiSummaryCard />
          <RecommendedTransfersCard onApplied={(msg) => toast(msg, 'pos')} />
          <TransfersMadeCard gw={gw} />
        </div>

        <div className="od-stack" style={{ gap: 16 }}>
          <FixturesCard gw={gw} />
          <InjuriesCard />
        </div>
      </div>

      {dialogPlayer ? <PlayerActionDialog player={dialogPlayer} onClose={() => setDialogPlayer(null)} /> : null}
    </div>
  );
}

function RecommendedTransfersCard({ onApplied }: { onApplied: (msg: ReactNode) => void }) {
  const core = useCore();
  const { applyRecommendation } = useSquad();
  const navigate = useNavigate();
  const recs = core.recommendedTransfers;

  return (
    <Card
      reveal
      title="Recommended transfers"
      actions={<Badge tone="accent">{recs.length} suggestions</Badge>}
    >
      {recs.length ? (
        <div className="od-stack" style={{ gap: 12 }}>
          {recs.map((t) => {
            const gain = Math.round(t.projected_points_delta * 10) / 10;
            return (
              <div key={t.transfer_id} className="od-stack" style={{ gap: 8, paddingBottom: 12, borderBottom: '1px solid var(--border)' }}>
                <div className="od-row" style={{ gap: 8, flexWrap: 'wrap' }}>
                  <Badge tone="neg">OUT</Badge>
                  <span className="small od-fill">{t.out.player_name}</span>
                  <Icon name="swap" size={16} className="muted" />
                  <Badge tone="pos">IN</Badge>
                  <span className="small od-fill">{t.in.player_name}</span>
                  <Badge tone="accent" className="mono">{plus(gain)} xPts</Badge>
                </div>
                <p className="tiny muted">{t.reason}</p>
                <div className="od-row" style={{ gap: 8 }}>
                  <button
                    className="btn btn-primary btn-sm"
                    type="button"
                    onClick={() => {
                      const outId = findPlayerIdByName(core.players, t.out.player_name);
                      const inId = findPlayerIdByName(core.players, t.in.player_name);
                      if (outId && inId && applyRecommendation(outId, inId)) {
                        onApplied(<>Added <b>{t.in.player_name}</b> to the sandbox.</>);
                        navigate('/command?tab=sandbox');
                      } else {
                        navigate('/command?tab=sandbox');
                      }
                    }}
                  >
                    Apply in sandbox
                  </button>
                  <Link className="btn btn-ghost btn-sm" to="/command?tab=sandbox">
                    Sandbox it
                  </Link>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="small muted">
          Run a model blend in the Command Center to generate transfer recommendations.
        </div>
      )}
    </Card>
  );
}

function TransfersMadeCard({ gw }: { gw: number }) {
  const core = useCore();
  const rows = core.transfersLatest
    .filter((r) => r.event === gw)
    .slice()
    .sort((a, b) => (a.time ?? '').localeCompare(b.time ?? ''));

  const hit = core.history.find((h) => h.event === gw)?.hit ?? 0;
  const hitCount = Math.round(hit / 4);

  if (!rows.length) {
    return (
      <Card reveal title={`Transfers · GW${gw}`}>
        <div className="small muted">No transfers made in GW{gw}.</div>
      </Card>
    );
  }

  return (
    <Card
      reveal
      title={`Transfers · GW${gw}`}
      actions={
        <>
          <Badge>{rows.length} move{rows.length === 1 ? '' : 's'}</Badge>
          {hit ? <Badge tone="neg" className="mono">−{hit} pts</Badge> : <Badge tone="pos">No hits</Badge>}
        </>
      }
    >
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>In</th>
              <th>Out</th>
              <th>Cost</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => {
              const charged = i >= rows.length - hitCount;
              return (
                <tr key={`${r.element_in}-${r.element_out}-${i}`}>
                  <td>
                    <PlayerCell id={r.element_in ?? -1} />
                  </td>
                  <td>
                    <PlayerCell id={r.element_out ?? -1} />
                  </td>
                  <td>
                    {charged ? (
                      <Badge tone="neg" className="mono">−4</Badge>
                    ) : (
                      <Badge tone="pos">Free</Badge>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
