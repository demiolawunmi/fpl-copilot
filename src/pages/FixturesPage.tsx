import { useEffect, useState } from 'react';
import { Icon } from '../components/Icon';
import { Card } from '../components/ds/Card';
import { Badge, Crest, SkeletonLines } from '../components/ds/atoms';
import { FixturesCard } from '../components/shared';
import { useCore } from '../context/CoreContext';
import { getFdrEloSnapshot } from '../api/backend/fdr';
import { outlookLabel, teamGwDifficulty } from '../domain/model';
import { clamp, round1 } from '../lib/format';
import type { Club } from '../domain/types';

/** Copilot's squad-aware Elo FDR, normalized against the league distribution. */
function copilotFdr(oppElo: number, home: boolean, mean: number, std: number): number {
  const z = std > 0 ? (oppElo - mean) / std : 0;
  return clamp(3 + z * 0.9 + (home ? -0.7 : 0.7), 1, 5);
}

export default function FixturesPage() {
  const core = useCore();
  const [gwOverride, setGwOverride] = useState<number | null>(null);
  const [eloById, setEloById] = useState<Map<number, number>>(new Map());
  const [eloError, setEloError] = useState<string | null>(null);

  const gw = gwOverride ?? core.nextGW;

  useEffect(() => {
    let cancelled = false;
    getFdrEloSnapshot()
      .then((snap) => {
        if (!cancelled) setEloById(new Map(snap.ratings.map((r) => [r.team_id, r.elo])));
      })
      .catch((e: unknown) => {
        if (!cancelled) setEloError(e instanceof Error ? e.message : 'Elo unavailable');
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const gwIds: number[] = [];
  for (let i = 0; i < 5; i++) {
    const id = gw + i;
    if (id <= (core.gameweeks.length || 38)) gwIds.push(id);
  }

  const elos = [...eloById.values()];
  const eloMean = elos.length ? elos.reduce((a, b) => a + b, 0) / elos.length : 0;
  const eloStd = elos.length
    ? Math.sqrt(elos.reduce((a, b) => a + (b - eloMean) ** 2, 0) / elos.length)
    : 0;

  const clubRows = core.clubs
    .slice()
    .sort((a, b) => a.name.localeCompare(b.name))
    .map((club) => {
      const official: (number | null)[] = gwIds.map(() => null);
      const copilot: (number | null)[] = gwIds.map(() => null);
      gwIds.forEach((g, i) => {
        const entry = core.fixtureIndex?.teamGw.get(club.id)?.get(g);
        if (!entry) return;
        official[i] = entry.officialFdr;
        const oppElo = eloById.get(entry.opp.id);
        copilot[i] = oppElo != null ? round1(copilotFdr(oppElo, entry.home, eloMean, eloStd)) : null;
      });
      const vals = copilot.filter((v): v is number => v != null);
      const avg = vals.length ? round1(vals.reduce((a, b) => a + b, 0) / vals.length) : null;
      return { club, official, copilot, avg, elo: eloById.get(club.id) ?? null };
    });

  const hasMatrix = core.clubs.length > 0 && core.fixtureIndex != null;
  const eloRows = clubRows
    .filter((r) => r.elo != null || r.avg != null)
    .sort((a, b) => (b.elo ?? 0) - (a.elo ?? 0));

  return (
    <div className="od-stack page-enter" style={{ gap: 24 }}>
      <div className="od-row" style={{ gap: 16, flexWrap: 'wrap' }}>
        <div>
          <div className="tiny faint" style={{ textTransform: 'uppercase', letterSpacing: '.14em' }}>
            Intelligence
          </div>
          <h2>Fixture difficulty</h2>
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
          <div className="od-stat" style={{ minWidth: 160, textAlign: 'center' }}>
            <span style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 'var(--fs-lg)' }}>
              GW{gw}
            </span>
          </div>
          <button
            className="btn btn-icon btn-ghost"
            type="button"
            aria-label="Next gameweek"
            disabled={gw >= 38}
            onClick={() => setGwOverride(Math.min(38, gw + 1))}
          >
            <Icon name="chevronRight" />
          </button>
        </div>
      </div>

      <p className="muted small" style={{ maxWidth: '70ch' }}>
        Two ratings side by side: the official FPL FDR, and Copilot’s injury- and squad-aware Elo rating. Cells
        are colour-tiered from 1 (easiest) to 5 (hardest).
      </p>

      <FixturesCard gw={gw} />

      {!hasMatrix ? (
        <Card>
          <SkeletonLines count={12} />
        </Card>
      ) : (
        <>
          <div className="cols-2">
            <Card reveal title={`Official FPL FDR — GW${gw} onwards`}>
              <HeatMatrix kind="official" gwIds={gwIds} rows={clubRows} />
            </Card>
            <Card reveal title={`Copilot Elo FDR — GW${gw} onwards`}>
              {eloError && eloById.size === 0 ? (
                <div className="small muted">Copilot Elo ratings are unavailable — {eloError}</div>
              ) : (
                <HeatMatrix kind="copilot" gwIds={gwIds} rows={clubRows} />
              )}
            </Card>
          </div>

          <Card
            reveal
            title="Team ratings"
            actions={
              <Badge tone="info">
                <Icon name="cpu" size={12} /> Club Elo
              </Badge>
            }
          >
            {eloRows.length ? (
              <div className="table-wrap">
                <table className="data">
                  <thead>
                    <tr>
                      <th>Team</th>
                      <th>Club Elo</th>
                      <th>Copilot FDR (next 5)</th>
                      <th>Run</th>
                      <th>Outlook</th>
                    </tr>
                  </thead>
                  <tbody>
                    {eloRows.map((r) => {
                      const avg = r.avg ?? 3;
                      const outlook = outlookLabel(avg);
                      return (
                        <tr key={r.club.id}>
                          <td>
                            <div className="cell-player">
                              <Crest club={r.club} size={24} />
                              <div className="nm">{r.club.name}</div>
                            </div>
                          </td>
                          <td className="mono">{r.elo != null ? Math.round(r.elo) : '—'}</td>
                          <td className="mono">{avg.toFixed(1)}</td>
                          <td>
                            <span className="fdr-rail">
                              {r.copilot.map((v, i) => {
                                const k = Math.max(1, Math.min(5, Math.round(v ?? 3)));
                                return (
                                  <span key={i} className={`cell fdr-${k}`}>
                                    {v != null ? v.toFixed(1) : '–'}
                                  </span>
                                );
                              })}
                            </span>
                          </td>
                          <td>
                            <span className={`badge ${outlook.tone}`}>{outlook.label}</span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="small muted">Elo ratings are unavailable — is the backend running?</div>
            )}
          </Card>
        </>
      )}

      <Card reveal title="Best upcoming runs">
        <div className="od-stack" style={{ gap: 8 }}>
          {core.clubs
            .map((c) => ({
              club: c,
              avg: core.fixtureIndex ? teamGwDifficulty(core.fixtureIndex, c.id, gw, 5) : 3,
            }))
            .sort((a, b) => a.avg - b.avg)
            .slice(0, 6)
            .map(({ club, avg }) => {
              const outlook = outlookLabel(avg);
              return (
                <div key={club.id} className="od-row" style={{ gap: 12 }}>
                  <Crest club={club} size={24} />
                  <span className="od-fill small">{club.name}</span>
                  <span className="mono small muted">{avg.toFixed(1)}</span>
                  <span className={`badge ${outlook.tone}`}>{outlook.label}</span>
                </div>
              );
            })}
        </div>
      </Card>
    </div>
  );
}

function HeatMatrix({
  kind,
  gwIds,
  rows,
}: {
  kind: 'official' | 'copilot';
  gwIds: number[];
  rows: { club: Club; official: (number | null)[]; copilot: (number | null)[] }[];
}) {
  return (
    <div className="heat-scroll">
      <table className="heat">
        <thead>
          <tr>
            <th className="teamcol">Team</th>
            {gwIds.map((g) => (
              <th key={g}>GW{g}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const values = kind === 'official' ? r.official : r.copilot;
            return (
              <tr key={r.club.id}>
                <th className="teamcol">
                  <span className="od-row" style={{ gap: 8 }}>
                    <Crest club={r.club} size={20} />
                    <span className="small">{r.club.short}</span>
                  </span>
                </th>
                {gwIds.map((g, i) => {
                  const v = values[i];
                  if (v == null) {
                    return (
                      <td key={g} className="faint">
                        –
                      </td>
                    );
                  }
                  const k = Math.max(1, Math.min(5, Math.round(v)));
                  return (
                    <td
                      key={g}
                      className={`heat-cell fdr-${k} ${k >= 4 ? 'hi' : ''}`}
                      style={{ animationDelay: `${i * 18}ms` }}
                    >
                      {kind === 'official' ? Math.round(v) : v.toFixed(1)}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
