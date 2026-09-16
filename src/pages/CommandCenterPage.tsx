import { useEffect, useRef, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { Icon } from '../components/Icon';
import { Card } from '../components/ds/Card';
import {
  Alert,
  Avatar,
  Badge,
  Crest,
  Empty,
  Segmented,
  SkeletonLines,
} from '../components/ds/atoms';
import { ColChart, Bars } from '../components/ds/charts';
import { Countdown } from '../components/ds/Countdown';
import { BenchRow, Pitch } from '../components/ds/Pitch';
import { AiSummaryCard, FixturesSnapshotCard, InjuriesCard } from '../components/shared';
import { PlayerActionDialog } from '../components/player/PlayerActionDialog';
import { useCore } from '../context/CoreContext';
import { useSquad, type SandboxDelta } from '../context/SquadContext';
import { useToast } from '../context/ToastContext';
import { useTeamId } from '../context/TeamIdContext';
import {
  runAirsenal,
  submitCopilotBlendJob,
  getCopilotBlendJobStatus,
  postCopilotChat,
  type CopilotBlendSubmitRequest,
  type CopilotChatTurn,
} from '../api/backend';
import { findPlayerIdByName, nextFixtures } from '../domain/model';
import { clamp, compactCount, money, num, plus } from '../lib/format';
import { xiXpts } from '../domain/projection';
import type { Player } from '../domain/types';

const DEFAULT_WEIGHTS = { official: 25, elo: 25, airsenal: 25, copilot: 25 };

/**
 * The backend blend accepts exactly two model sources (Club Elo + AIrsenal)
 * whose weights must sum to 1.0. The UI exposes four models for context, so
 * the two blendable sources are normalized to a valid pair.
 */
function normalizeSourceWeights(elo: number, airsenal: number): { elo: number; airsenal: number } {
  const sum = elo + airsenal;
  if (sum <= 0) return { elo: 0.5, airsenal: 0.5 };
  return {
    elo: Math.round((elo / sum) * 1000) / 1000,
    airsenal: Math.round(((sum - elo) / sum) * 1000) / 1000,
  };
}

function parseWeights(raw: string | null) {
  const out = { ...DEFAULT_WEIGHTS };
  if (!raw) return out;
  for (const pair of raw.split(',')) {
    const [k, v] = pair.split(':');
    if (k in out && Number.isFinite(Number(v))) out[k as keyof typeof out] = clamp(Number(v), 0, 100);
  }
  return out;
}

export default function CommandCenterPage() {
  const core = useCore();
  const [params, setParams] = useSearchParams();
  const tab = params.get('tab') === 'sandbox' ? 'sandbox' : 'pick';

  const [dismissedBanner, setDismissedBanner] = useState(false);

  return (
    <div className="od-stack page-enter" style={{ gap: 20 }}>
      <div className="od-row" style={{ gap: 16, flexWrap: 'wrap' }}>
        <div>
          <div className="tiny faint" style={{ textTransform: 'uppercase', letterSpacing: '.14em' }}>
            Plan
          </div>
          <h2>Command Center</h2>
          <div className="small muted">
            Gameweek {core.nextGW} • {core.manager?.name ?? core.entry?.name ?? '—'} • ID{' '}
            {core.manager?.teamId ?? '—'}
          </div>
        </div>
        <span className="spacer" />
        <OptimizeButton />
      </div>

      {core.seasonStatus && !core.seasonStatus.is_current && !dismissedBanner ? (
        <Alert
          tone="warn"
          icon="alert"
          actions={
            <button className="btn btn-ghost btn-sm" type="button" onClick={() => setDismissedBanner(true)}>
              Dismiss
            </button>
          }
        >
          <b>Some pricing data may be stale.</b>
          <div className="small">
            Live season is {core.seasonStatus.fpl_season ?? 'unknown'} but local exports are{' '}
            {core.seasonStatus.data_season ?? 'unknown'}. Projections still run, but transfer prices may shift
            before the deadline.
          </div>
        </Alert>
      ) : null}

      <StatusStrip />

      <div className="tabs" role="tablist" aria-label="Command Center sections">
        <button
          className="tab"
          role="tab"
          aria-selected={tab === 'pick'}
          onClick={() => setParams({ tab: 'pick' })}
        >
          <Icon name="crown" size={14} /> Pick Team
        </button>
        <button
          className="tab"
          role="tab"
          aria-selected={tab === 'sandbox'}
          onClick={() => setParams({ tab: 'sandbox' })}
        >
          <Icon name="sliders" size={14} /> AI Sandbox
        </button>
      </div>

      {core.loading ? (
        <Card>
          <SkeletonLines count={10} />
        </Card>
      ) : tab === 'pick' ? (
        <PickTeamTab />
      ) : (
        <SandboxTab weights={parseWeights(params.get('w'))} />
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ status */

function StatusStrip() {
  const core = useCore();
  const manager = core.manager;
  const usedChips = manager?.chips.filter((c) => c.used).length ?? 0;

  return (
    <Card reveal>
      <div className="od-row" style={{ gap: 20, flexWrap: 'wrap' }}>
        <div className="od-stack" style={{ gap: 6 }}>
          <span className="tiny faint" style={{ textTransform: 'uppercase', letterSpacing: '.14em' }}>
            Chips
          </span>
          <div className="od-row" style={{ gap: 8, flexWrap: 'wrap' }}>
            {(manager?.chips ?? []).map((c) => (
              <span className="tooltip-host" key={c.key}>
                <span className="badge" style={{ opacity: c.used ? 0.6 : undefined, color: c.used ? undefined : 'var(--accent)', borderColor: c.used ? undefined : 'color-mix(in srgb, var(--accent) 40%, transparent)' }}>
                  <Icon name={c.used ? 'lock' : 'check'} size={12} /> {c.name}
                  {c.used ? ` · GW${c.usedGw ?? ''}` : ''}
                </span>
                <span className="tip">
                  <b>{c.name}</b>
                  <br />
                  {c.note}
                  <br />
                  {c.used ? `Used in GW${c.usedGw ?? '?'}` : 'Available'}
                </span>
              </span>
            ))}
            {usedChips ? <Badge>{usedChips}/4 chips used</Badge> : null}
          </div>
        </div>
        <span className="spacer" />
        <div className="od-row" style={{ gap: 24, flexWrap: 'wrap' }}>
          <div className="od-stat">
            <span className="k tiny faint">Free transfers</span>
            <span className="v mono" style={{ fontSize: 'var(--fs-xl)' }}>
              {manager?.freeTransfers ?? '—'}
            </span>
          </div>
          <div className="od-stat">
            <span className="k tiny faint">Bank</span>
            <span className="v mono" style={{ fontSize: 'var(--fs-xl)' }}>
              {money(manager?.bank ?? 0)}
            </span>
          </div>
          <div className="od-stat">
            <span className="k tiny faint">Squad value</span>
            <span className="v mono" style={{ fontSize: 'var(--fs-xl)' }}>
              {money(manager?.squadValue ?? 0)}
            </span>
          </div>
          <div className="od-stat">
            <span className="k tiny faint">Deadline in</span>
            <span className="v mono" style={{ fontSize: 'var(--fs-xl)' }}>
              <Countdown target={core.nextDeadline} />
            </span>
            <span className="tiny muted">GW{core.nextGW}</span>
          </div>
        </div>
      </div>
    </Card>
  );
}

/* ---------------------------------------------------------------- pick team */

function PickTeamTab() {
  const core = useCore();
  const { xi, bench, captainId, viceId, byId, quickCaptain, quickBench, autoPickXi } = useSquad();
  const toast = useToast();
  const [, setParams] = useSearchParams();
  const [dialogPlayer, setDialogPlayer] = useState<Player | null>(null);

  const fixtureOf = (p: Player) => {
    if (!core.fixtureIndex) return null;
    return nextFixtures(core.fixtureIndex, p.teamId, core.nextGW, 1)[0] ?? null;
  };

  const inForm = core.players
    .filter((p) => p.last4)
    .sort((a, b) => (b.last4?.points ?? 0) - (a.last4?.points ?? 0))
    .slice(0, 5);

  const bandwagons = [...core.players].sort((a, b) => b.transfersNet - a.transfersNet).slice(0, 5);

  return (
    <div className="cols-3">
      <div className="od-stack" style={{ gap: 16 }}>
        <Card
          reveal
          title={`Starting XI · GW${core.nextGW}`}
          actions={
            <>
              <Badge className="mono">{num(xiXpts(xi, byId, captainId), 1)} xPts</Badge>
              <Badge tone="info">
                <Icon name="cpu" size={12} /> Copilot
              </Badge>
            </>
          }
        >
          <Pitch
            ids={xi}
            byId={byId}
            captainId={captainId}
            viceId={viceId}
            fixtureOf={fixtureOf}
            onChipClick={setDialogPlayer}
          />
          <BenchRow
            ids={bench}
            byId={byId}
            captainId={captainId}
            viceId={viceId}
            fixtureOf={fixtureOf}
            onChipClick={setDialogPlayer}
          />
        </Card>

        <div className="grid grid-2">
          <Card reveal title="Quick actions">
            <div className="od-stack" style={{ gap: 10 }}>
              <QuickAction
                icon="crown"
                label="Auto-pick captain"
                desc="Highest projected xPts"
                onClick={() => {
                  quickCaptain();
                  toast('Captain auto-picked by projected xPts.');
                }}
              />
              <QuickAction
                icon="check"
                label="Auto-pick best XI"
                desc="Highest projected legal lineup"
                onClick={() => {
                  autoPickXi();
                  toast('Best starting XI selected by projected xPts.');
                }}
              />
              <QuickAction
                icon="swap"
                label="Auto-pick bench order"
                desc="By next-GW xPts"
                onClick={() => {
                  quickBench();
                  toast('Bench order re-ordered by projected xPts.');
                }}
              />
              <OptimizeButton variant="quick" />
              <QuickAction
                icon="sliders"
                label="Open the Sandbox"
                desc="Plan transfers as a what-if"
                onClick={() => setParams({ tab: 'sandbox' })}
              />
            </div>
          </Card>

          <Card reveal title="In form" actions={<Badge>Last 4 GWs</Badge>}>
            <div className="table-wrap">
              <table className="data">
                <thead>
                  <tr>
                    <th>Player</th>
                    <th>Pts</th>
                    <th>Mins</th>
                    <th>xGI</th>
                  </tr>
                </thead>
                <tbody>
                  {inForm.map((p) => (
                    <tr key={p.id} className="clickable" onClick={() => setDialogPlayer(p)}>
                      <td>
                        <div className="cell-player">
                          <Avatar player={p} size={26} />
                          <Crest club={p.club} size={18} />
                          <div className="nm small">{p.name}</div>
                        </div>
                      </td>
                      <td className="mono pos">{p.last4?.points ?? 0}</td>
                      <td className="mono">{p.last4?.minutes ?? 0}</td>
                      <td className="mono">{p.last4?.xgi.toFixed(1) ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>

        <AiSummaryCard title="AI Command Summary" />
        <BandwagonsCard bandwagons={bandwagons} onOpen={setDialogPlayer} />
      </div>

      <div className="od-stack" style={{ gap: 16 }}>
        <FixturesSnapshotCard teamIds={xi.map((id) => byId.get(id)?.teamId ?? 0)} gw={core.nextGW} />
        <InjuriesCard />
      </div>

      {dialogPlayer ? <PlayerActionDialog player={dialogPlayer} onClose={() => setDialogPlayer(null)} /> : null}
    </div>
  );
}

function QuickAction({
  icon,
  label,
  desc,
  onClick,
}: {
  icon: Parameters<typeof Icon>[0]['name'];
  label: string;
  desc: string;
  onClick: () => void;
}) {
  return (
    <button className="btn btn-ghost" style={{ justifyContent: 'flex-start', textAlign: 'left' }} type="button" onClick={onClick}>
      <Icon name={icon} size={16} />
      <span className="od-stack" style={{ gap: 0, alignItems: 'flex-start' }}>
        <span>{label}</span>
        <span className="tiny faint" style={{ fontWeight: 400 }}>
          {desc}
        </span>
      </span>
    </button>
  );
}

function BandwagonsCard({ bandwagons, onOpen }: { bandwagons: Player[]; onOpen: (p: Player) => void }) {
  return (
    <Card reveal title="Bandwagons" actions={<Badge>Most transferred</Badge>}>
      <div className="od-stack" style={{ gap: 12 }}>
        {bandwagons.map((p) => (
          <button
            key={p.id}
            type="button"
            onClick={() => onOpen(p)}
            style={{
              display: 'grid',
              gridTemplateColumns: '28px 20px minmax(0,1fr) auto auto',
              alignItems: 'center',
              gap: 12,
              width: '100%',
              textAlign: 'left',
              background: 'transparent',
              border: 0,
              cursor: 'pointer',
              padding: '8px 0',
            }}
          >
            <Avatar player={p} size={28} />
            <Crest club={p.club} size={20} />
            <div style={{ minWidth: 0 }}>
              <div className="small" style={{ fontWeight: 600 }}>
                {p.name}
              </div>
              <div className="tiny faint">
                {p.club.short} · {money(p.price)}
              </div>
            </div>
            <div style={{ display: 'grid', gap: 2, textAlign: 'right', minWidth: 58 }}>
              <span className="tiny pos mono">
                <Icon name="arrowUp" size={11} /> {compactCount(p.transfersIn)}
              </span>
              <span className="tiny neg mono">
                <Icon name="arrowDown" size={11} /> {compactCount(p.transfersOut)}
              </span>
            </div>
            <span
              className={`badge ${p.transfersNet >= 0 ? 'pos' : 'neg'} mono`}
              style={{ minWidth: 66, justifyContent: 'center' }}
            >
              {p.transfersNet >= 0 ? '+' : ''}
              {compactCount(p.transfersNet)}
            </span>
          </button>
        ))}
      </div>
    </Card>
  );
}

/* ------------------------------------------------------------------ sandbox */

function SandboxTab({ weights: initialWeights }: { weights: typeof DEFAULT_WEIGHTS }) {
  const core = useCore();
  const squad = useSquad();
  const {
    sandbox,
    byId,
    toggleSandbox,
    resetSandbox,
    undo,
    applyToTeam,
    sandboxDelta,
  } = squad;
  const toast = useToast();
  const [dialogPlayer, setDialogPlayer] = useState<Player | null>(null);

  const d = sandboxDelta();

  const fixtureOf = (p: Player) => {
    if (!core.fixtureIndex) return null;
    return nextFixtures(core.fixtureIndex, p.teamId, core.nextGW, 1)[0] ?? null;
  };

  return (
    <div className="od-stack" style={{ gap: 16 }}>
      <Card reveal>
        <div className="od-row" style={{ gap: 12, flexWrap: 'wrap' }}>
          <label className="od-row" style={{ gap: 8, cursor: 'pointer' }}>
            <span
              className="switch"
              role="switch"
              tabIndex={0}
              aria-checked={sandbox.on}
              onClick={toggleSandbox}
              onKeyDown={(e) => {
                if (e.key === ' ' || e.key === 'Enter') {
                  e.preventDefault();
                  toggleSandbox();
                }
              }}
            />
            <span className="small">
              <b>Sandbox mode</b> <span className="muted">— changes never touch your real team</span>
            </span>
          </label>
          <span className="spacer" />
          <button className="btn btn-ghost btn-sm" type="button" disabled={!sandbox.history.length} onClick={undo}>
            <Icon name="undo" size={14} /> Undo
          </button>
          <button className="btn btn-ghost btn-sm" type="button" disabled={!sandbox.transfers.length} onClick={resetSandbox}>
            <Icon name="refresh" size={14} /> Reset
          </button>
          <button
            className="btn btn-primary btn-sm"
            type="button"
            disabled={!sandbox.transfers.length}
            onClick={() => {
              applyToTeam();
              toast('Sandbox applied to your team for this plan.');
            }}
          >
            <Icon name="check" size={14} /> Apply to team
          </button>
        </div>
      </Card>

      <DeltaStrip delta={d} />

      <div className="cols-2">
        <RecommendedTransfersCard />
        <Card reveal title="Sandbox charts" actions={<Badge tone="info"><Icon name="cpu" size={12} /> Projected</Badge>}>
          <div className="od-stack" style={{ gap: 18 }}>
            <div>
              <div className="tiny faint" style={{ textTransform: 'uppercase', letterSpacing: '.12em', marginBottom: 8 }}>
                Starting XI xPts by position
              </div>
              <ColChart
                items={(['GK', 'DEF', 'MID', 'FWD'] as const).map((pos) => {
                  const sum = sandbox.xi
                    .filter((id) => byId.get(id)?.pos === pos)
                    .reduce((t, id) => t + (byId.get(id)?.xpts ?? 0), 0);
                  const colors = { GK: 'var(--warn)', DEF: 'var(--info)', MID: 'var(--accent)', FWD: 'var(--neg)' };
                  return { k: pos, v: Math.round(sum * 10) / 10, color: colors[pos] };
                })}
              />
            </div>
            <div>
              <div className="tiny faint" style={{ textTransform: 'uppercase', letterSpacing: '.12em', marginBottom: 8 }}>
                Transfer impact (xPts gain)
              </div>
              {sandbox.transfers.length ? (
                <Bars
                  items={sandbox.transfers.map((t) => ({
                    k: t.inName.split(' ').slice(-1)[0],
                    v: Math.max(0.1, t.gain),
                    label: `${t.gain >= 0 ? '+' : ''}${t.gain}`,
                    color: t.gain >= 0 ? 'var(--pos)' : 'var(--neg)',
                  }))}
                  max={Math.max(2, ...sandbox.transfers.map((t) => Math.abs(t.gain)))}
                />
              ) : (
                <Empty icon="sliders" title="No transfers in the sandbox" text="Add a transfer to see its projected xPts impact." />
              )}
            </div>
          </div>
        </Card>
      </div>

      <Card
        reveal
        title={`Sandbox pitch · GW${core.nextGW}`}
        actions={<Badge className="mono">{num(xiXpts(sandbox.xi, byId, sandbox.captainId), 1)} xPts</Badge>}
      >
        <Pitch
          ids={sandbox.xi}
          byId={byId}
          captainId={sandbox.captainId}
          viceId={sandbox.viceId}
          swapId={sandbox.outId}
          fixtureOf={fixtureOf}
          onChipClick={setDialogPlayer}
        />
        <BenchRow
          ids={sandbox.bench}
          byId={byId}
          captainId={sandbox.captainId}
          viceId={sandbox.viceId}
          swapId={sandbox.outId}
          fixtureOf={fixtureOf}
          onChipClick={setDialogPlayer}
        />
      </Card>

      <div className="cols-2">
        <TransferBuilder />
        <BlendPanel initialWeights={initialWeights} />
      </div>

      <ChatPanel />

      {dialogPlayer ? <PlayerActionDialog player={dialogPlayer} onClose={() => setDialogPlayer(null)} /> : null}
    </div>
  );
}

function DeltaStrip({ delta }: { delta: SandboxDelta }) {
  if (!delta.count) {
    return (
      <Alert tone="info" icon="info">
        <b>No changes yet.</b>{' '}
        <span className="small">The sandbox matches your real team. Build a transfer below to see the projected impact.</span>
      </Alert>
    );
  }
  const item = (k: string, real: number, sand: number, d: number, dec: number, prefix = '', suffix = '') => (
    <div className="stat-tile">
      <span className="k">{k}</span>
      <span className="v mono sm">
        {prefix}
        {real.toFixed(dec)}
        {suffix} <span className="muted" style={{ fontSize: 'var(--fs-base)' }}>→</span> {prefix}
        {sand.toFixed(dec)}
        {suffix}
      </span>
      <span className={`d ${d >= 0 ? 'pos' : 'neg'} mono`}>
        {d >= 0 ? '+' : ''}
        {d.toFixed(dec)}
        {suffix}
      </span>
    </div>
  );
  return (
    <div className="stat-strip" style={{ gridTemplateColumns: 'repeat(5, minmax(0,1fr))' }}>
      {item('GW xPts', delta.realGw, delta.sandGw, delta.gwDelta, 1)}
      {item('Next 5 GW xPts', delta.real5, delta.sand5, delta.d5, 0)}
      {item('Bank', 0, delta.bank, delta.bankDelta, 1, '£', 'm')}
      <div className="stat-tile">
        <span className="k">Transfers</span>
        <span className="v mono sm">{delta.count}</span>
        <span className="d muted">
          {delta.free} free · {delta.hits} hit{delta.hits === 1 ? '' : 's'}
        </span>
      </div>
      <div className="stat-tile">
        <span className="k">Hit cost</span>
        <span className={`v mono sm ${delta.hitCost ? 'neg' : ''}`}>{delta.hitCost ? `−${delta.hitCost}` : '0'}</span>
        <span className="d muted">−4 per extra transfer</span>
      </div>
    </div>
  );
}

function RecommendedTransfersCard() {
  const core = useCore();
  const { applyRecommendation, sandbox } = useSquad();
  const toast = useToast();
  const recs = core.recommendedTransfers;

  return (
    <Card reveal title="Recommended transfers" actions={<Badge tone="accent">{recs.length}</Badge>}>
      {recs.length ? (
        <div className="od-stack" style={{ gap: 12 }}>
          {recs.map((t) => {
            const gain = Math.round(t.projected_points_delta * 10) / 10;
            const done = sandbox.transfers.some((x) => x.inName === t.in.player_name);
            const outId = findPlayerIdByName(core.players, t.out.player_name);
            const inId = findPlayerIdByName(core.players, t.in.player_name);
            const outP = outId ? core.playersById.get(outId) : undefined;
            const inP = inId ? core.playersById.get(inId) : undefined;
            return (
              <div key={t.transfer_id} className="od-stack" style={{ gap: 8, paddingBottom: 12, borderBottom: '1px solid var(--border)' }}>
                <div className="od-row" style={{ gap: 8, flexWrap: 'wrap' }}>
                  <Badge tone="neg">OUT</Badge>
                  {outP ? <Avatar player={outP} size={24} /> : null}
                  <span className="small od-fill">{t.out.player_name}</span>
                  <Icon name="swap" size={14} className="muted" />
                  <Badge tone="pos">IN</Badge>
                  {inP ? <Avatar player={inP} size={24} /> : null}
                  <span className="small od-fill">{t.in.player_name}</span>
                  <Badge tone="accent" className="mono">{plus(gain)} xPts</Badge>
                </div>
                <p className="tiny muted">{t.reason}</p>
                <div className="od-row" style={{ gap: 8 }}>
                  <button
                    className={`btn ${done ? 'btn-ghost' : 'btn-primary'} btn-sm`}
                    type="button"
                    disabled={done}
                    onClick={() => {
                      const outId = findPlayerIdByName(core.players, t.out.player_name);
                      const inId = findPlayerIdByName(core.players, t.in.player_name);
                      if (outId && inId && applyRecommendation(outId, inId)) {
                        toast('Transfer added to the sandbox.', 'pos');
                      } else {
                        toast('That move isn’t valid for your squad.', 'neg');
                      }
                    }}
                  >
                    {done ? <><Icon name="check" size={14} /> Added</> : 'Apply in sandbox'}
                  </button>
                  <Link className="btn btn-ghost btn-sm" to={`/player/${findPlayerIdByName(core.players, t.in.player_name) ?? ''}?from=command`}>
                    <Icon name="eye" size={14} /> Profile
                  </Link>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="small muted">
          Run a model blend to generate transfer recommendations grounded in your squad.
        </div>
      )}
    </Card>
  );
}

function TransferBuilder() {
  const core = useCore();
  const { sandbox, byId, pickOut, pickIn, clearPair, applySandboxTransfer } = useSquad();
  const toast = useToast();
  const [search, setSearch] = useState('');
  const [posFilter, setPosFilter] = useState<'all' | 'GK' | 'DEF' | 'MID' | 'FWD'>('all');
  const [sortKey, setSortKey] = useState<'xpts' | 'price' | 'form' | 'own'>('xpts');

  const squadIds = [...sandbox.xi, ...sandbox.bench];
  const outPlayer = sandbox.outId != null ? byId.get(sandbox.outId) : undefined;

  const available = core.players
    .filter((p) => !squadIds.includes(p.id) && p.status !== 'i')
    .filter((p) => (posFilter === 'all' ? true : p.pos === posFilter))
    .filter((p) => (outPlayer && posFilter === 'all' ? p.pos === outPlayer.pos : true))
    .filter((p) => (search ? p.name.toLowerCase().includes(search.toLowerCase()) : true))
    .sort((a, b) => {
      if (sortKey === 'price') return b.price - a.price;
      if (sortKey === 'form') return b.form - a.form;
      if (sortKey === 'own') return b.own - a.own;
      return b.xpts - a.xpts;
    })
    .slice(0, 60);

  return (
    <Card
      reveal
      title="Custom transfer builder"
      actions={
        <Badge tone={sandbox.inId ? 'pos' : sandbox.outId ? 'info' : ''}>
          {sandbox.outId ? 'Step 2 of 2 · pick replacement' : 'Step 1 of 2 · pick a player to sell'}
        </Badge>
      }
    >
      <div className="od-stack" style={{ gap: 14 }}>
        <div className="od-row" style={{ gap: 8, flexWrap: 'wrap' }}>
          <Segmented
            ariaLabel="Position filter"
            value={posFilter}
            onChange={(v) => setPosFilter(v)}
            options={[
              { value: 'all', label: 'All' },
              { value: 'GK', label: 'GK' },
              { value: 'DEF', label: 'DEF' },
              { value: 'MID', label: 'MID' },
              { value: 'FWD', label: 'FWD' },
            ]}
          />
          <span className="spacer" />
          <label className="od-row" style={{ gap: 6 }}>
            <span className="tiny faint">Sort</span>
            <select
              className="select"
              aria-label="Sort available players"
              style={{ minHeight: 34, padding: '0 32px 0 10px', fontSize: 'var(--fs-xs)' }}
              value={sortKey}
              onChange={(e) => setSortKey(e.target.value as typeof sortKey)}
            >
              <option value="xpts">xPts</option>
              <option value="price">Price</option>
              <option value="form">Form</option>
              <option value="own">Ownership</option>
            </select>
          </label>
        </div>

        <div className="grid grid-2" style={{ gap: 16 }}>
          <div className="od-stack" style={{ gap: 8 }}>
            <div className="tiny faint" style={{ textTransform: 'uppercase', letterSpacing: '.12em' }}>
              Your squad — pick OUT
            </div>
            <div className="od-scroll" style={{ maxHeight: 340, display: 'grid', gap: 2 }}>
              {sandbox.xi.concat(sandbox.bench).map((id) => {
                const p = byId.get(id);
                if (!p) return null;
                return <BuilderRow key={id} player={p} selected={sandbox.outId === p.id} bench={sandbox.bench.includes(id)} onClick={() => pickOut(p.id)} />;
              })}
            </div>
          </div>
          <div className="od-stack" style={{ gap: 8 }}>
            <div className="tiny faint" style={{ textTransform: 'uppercase', letterSpacing: '.12em' }}>
              {outPlayer ? `Available ${outPlayer.pos}s — pick IN` : 'Available players'}
            </div>
            <input
              className="input"
              placeholder="Search players…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              aria-label="Search available players"
              style={{ minHeight: 40 }}
            />
            <div className="od-scroll" style={{ maxHeight: 296, display: 'grid', gap: 2 }}>
              {available.length ? (
                available.map((p) => (
                  <BuilderRow key={p.id} player={p} selected={sandbox.inId === p.id} onClick={() => pickIn(p.id)} />
                ))
              ) : (
                <Empty title="No players match" text="Try another position or clear the search." />
              )}
            </div>
          </div>
        </div>

        {sandbox.outId && sandbox.inId ? (
          <Alert tone="pos" icon="swap">
            <span className="small">
              <b>{byId.get(sandbox.outId)?.name}</b> → <b>{byId.get(sandbox.inId)?.name}</b> ·{' '}
              {money(Math.max(0, (byId.get(sandbox.inId)?.price ?? 0) - (byId.get(sandbox.outId)?.price ?? 0)))} spend
            </span>
            <div className="od-row" style={{ gap: 8, marginTop: 8 }}>
              <button className="btn btn-ghost btn-sm" type="button" onClick={clearPair}>
                Clear
              </button>
              <button
                className="btn btn-primary btn-sm"
                type="button"
                onClick={() => {
                  if (applySandboxTransfer()) toast('Transfer added to the sandbox.', 'pos');
                  else toast('That move isn’t valid.', 'neg');
                }}
              >
                Make transfer
              </button>
            </div>
          </Alert>
        ) : sandbox.outId ? (
          <Alert tone="info" icon="info">
            <span className="small">
              Now pick a replacement for <b>{outPlayer?.name}</b> from the available list.
            </span>
          </Alert>
        ) : null}
      </div>
    </Card>
  );
}

function BuilderRow({
  player,
  selected,
  bench,
  onClick,
}: {
  player: Player;
  selected: boolean;
  bench?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className="od-row"
      onClick={onClick}
      style={{
        gap: 10,
        width: '100%',
        textAlign: 'left',
        padding: '8px 10px',
        borderRadius: 'var(--r-sm)',
        border: `1px solid ${selected ? 'var(--accent)' : 'transparent'}`,
        background: selected ? 'color-mix(in srgb, var(--accent) 12%, transparent)' : 'transparent',
        cursor: 'pointer',
      }}
    >
      <Avatar player={player} size={28} />
      <Crest club={player.club} size={18} />
      <span className="od-fill">
        <span className="small" style={{ fontWeight: 600, display: 'block' }}>
          {player.name} {bench ? <span className="badge">Bench</span> : null}{' '}
          {player.status !== 'a' ? <span className="badge warn">Flagged</span> : null}
        </span>
        <span className="tiny faint">
          {player.pos} · {player.club.short} · {money(player.price)} · form {player.form}
        </span>
      </span>
      <span className={`pos-badge pos-${player.pos}`}>{player.pos}</span>
      <span className="od-stat" style={{ textAlign: 'right', minWidth: 64 }}>
        <span className="tiny muted">
          xPts <b className="mono" style={{ color: 'var(--text)' }}>{player.xpts.toFixed(1)}</b>
        </span>
        <span className="tiny faint">{num(player.own, 1)}% own</span>
      </span>
    </button>
  );
}

/* ------------------------------------------------------------------- blend */

function BlendPanel({ initialWeights }: { initialWeights: typeof DEFAULT_WEIGHTS }) {
  const core = useCore();
  const squad = useSquad();
  const toast = useToast();
  const { teamId } = useTeamId();
  const [weights, setWeights] = useState(initialWeights);
  const [status, setStatus] = useState<'idle' | 'pending' | 'running' | 'completed' | 'failed'>('idle');
  const [progress, setProgress] = useState(0);

  const total = weights.official + weights.elo + weights.airsenal + weights.copilot;
  const over = total > 100;
  const exact = total === 100;

  const models: { key: keyof typeof weights; name: string; desc: string }[] = [
    { key: 'official', name: 'Official FPL', desc: 'Form, ownership and FPL’s own ICT index.' },
    { key: 'elo', name: 'Club Elo', desc: 'Team-strength ratings built from historical results.' },
    { key: 'airsenal', name: 'AIrsenal', desc: 'Gradient-boosted expected-points model.' },
    { key: 'copilot', name: 'Copilot', desc: 'Injury- and minutes-aware hybrid of the above.' },
  ];

  const apply = async () => {
    if (total !== 100) return;
    setStatus('pending');
    setProgress(8);
    try {
      const currentSquad = [...squad.sandbox.xi, ...squad.sandbox.bench]
        .map((id) => core.playersById.get(id))
        .filter((p): p is Player => Boolean(p))
        .map((p) => ({
          fpl_api_id: p.id,
          player_name: p.name,
          team: p.club.short,
          position: p.pos,
          price: p.price,
          x_pts: p.xpts,
        }));

      const request: CopilotBlendSubmitRequest = {
        schema_version: '1.0',
        correlation_id: crypto.randomUUID(),
        source_weights: normalizeSourceWeights(weights.elo, weights.airsenal),
        gameweek: core.nextGW,
        bank: core.manager?.bank ?? 0,
        free_transfers: core.manager?.freeTransfers ?? 1,
        current_squad: currentSquad,
        fpl_team_id: teamId ? Number(teamId) : undefined,
        task: 'hybrid',
      };

      const accepted = await submitCopilotBlendJob(request);
      setStatus('running');
      setProgress(30);

      const poll = window.setInterval(async () => {
        try {
          const s = await getCopilotBlendJobStatus(accepted.job_id);
          setProgress((p) => Math.min(95, p + 8));
          if (s.status === 'completed') {
            window.clearInterval(poll);
            setProgress(100);
            setStatus('completed');
            toast('Blend applied. Ask Copilot is now grounded in your squad.', 'pos');
            core.refresh();
          } else if (s.status === 'failed') {
            window.clearInterval(poll);
            setStatus('failed');
            toast('Blend failed. Your previous weights are unchanged.', 'neg');
          }
        } catch {
          window.clearInterval(poll);
          setStatus('failed');
          toast('Blend failed while polling the job.', 'neg');
        }
      }, 1500);
    } catch {
      setStatus('failed');
      toast('Blend failed. The model service may be unavailable.', 'neg');
    }
  };

  return (
    <Card
      reveal
      title="Model blending"
      actions={
        <Badge tone={over ? 'neg' : exact ? 'pos' : 'warn'}>
          {total}% {over ? '· over budget' : exact ? '· ready' : `· ${100 - total}% left`}
        </Badge>
      }
    >
      <div className="od-stack" style={{ gap: 16 }}>
        {models.map((m) => (
          <div className="od-stack" key={m.key} style={{ gap: 4 }}>
            <div className="od-row" style={{ gap: 8 }}>
              <span className="small" style={{ fontWeight: 600 }}>
                {m.name}
              </span>
              <span className="spacer" />
              <span className={`mono small ${over ? 'neg' : ''}`}>{weights[m.key]}%</span>
            </div>
            <input
              type="range"
              min={0}
              max={100}
              step={5}
              value={weights[m.key]}
              aria-label={`${m.name} weight`}
              onChange={(e) =>
                setWeights((w) => ({ ...w, [m.key]: Number(e.target.value) }))
              }
            />
            <span className="tiny faint">{m.desc}</span>
          </div>
        ))}
        <div className="progress">
          <i
            style={{
              width: `${Math.min(100, total)}%`,
              background: over ? 'var(--neg)' : exact ? 'var(--accent)' : 'var(--warn)',
            }}
          />
        </div>
        <p className="tiny faint">
          Club Elo and AIrsenal are the two blendable model sources; their weights are normalized for the
          run. Official FPL and Copilot are shown for context.
        </p>
        {over ? (
          <Alert tone="neg" icon="alert">
            <span className="small">
              Weights add up to {total}%. Blending is blocked above 100% — reduce a model by {total - 100} points.
            </span>
          </Alert>
        ) : null}
        {status === 'pending' || status === 'running' ? (
          <div className="od-stack" style={{ gap: 6 }}>
            <div className="od-row small">
              <span className="mono">
                <Icon name="cpu" size={14} /> Blending models…
              </span>
              <span className="spacer" />
              <span className="mono">{Math.round(progress)}%</span>
            </div>
            <div className="progress">
              <i style={{ width: `${progress}%` }} />
            </div>
          </div>
        ) : null}
        {status === 'completed' ? (
          <Alert tone="pos" icon="check">
            <span className="small">
              <b>Blend applied.</b> Projections now use {weights.official}/{weights.elo}/{weights.airsenal}/
              {weights.copilot} (official / Elo / AIrsenal / Copilot). Ask Copilot is unlocked.
            </span>
          </Alert>
        ) : null}
        {status === 'failed' ? (
          <Alert
            tone="neg"
            icon="alert"
            actions={
              <button className="btn btn-ghost btn-sm" type="button" onClick={apply}>
                Retry
              </button>
            }
          >
            <span className="small">
              <b>Blend failed.</b> The model service timed out. Your previous weights are unchanged.
            </span>
          </Alert>
        ) : null}
        <div className="od-row" style={{ gap: 8 }}>
          <button
            className="btn btn-primary"
            type="button"
            disabled={over || !exact || status === 'pending' || status === 'running'}
            onClick={apply}
          >
            <Icon name="cpu" size={16} /> Apply Blend
          </button>
          <button
            className="btn btn-ghost"
            type="button"
            onClick={() => {
              setWeights(DEFAULT_WEIGHTS);
              setStatus('idle');
            }}
          >
            Reset to default
          </button>
          <span className="spacer" />
          <button
            className="btn btn-ghost btn-sm"
            type="button"
            title="Copy a shareable link to these weights"
            onClick={() => {
              const url = `${location.origin}${location.pathname}?tab=sandbox&w=official:${weights.official},elo:${weights.elo},airsenal:${weights.airsenal},copilot:${weights.copilot}`;
              void navigator.clipboard?.writeText(url).catch(() => {});
              toast('Weights link copied. Shareable and deep-linkable.', 'info');
            }}
          >
            <Icon name="send" size={14} /> Share weights
          </button>
        </div>
      </div>
    </Card>
  );
}

/* -------------------------------------------------------------------- chat */

function ChatPanel() {
  const core = useCore();
  const toast = useToast();
  const [messages, setMessages] = useState<{ role: 'user' | 'assistant'; content: string }[]>([
    {
      role: 'assistant',
      content:
        'I’m grounded in your squad and the live model output. Ask me about captaincy, transfers, chip timing or a specific fixture run.',
    },
  ]);
  const [input, setInput] = useState('');
  const [thinking, setThinking] = useState(false);
  const threadRef = useRef<HTMLDivElement>(null);

  const enabled = Boolean(core.blendResult && core.blendInput);

  useEffect(() => {
    if (threadRef.current) threadRef.current.scrollTop = threadRef.current.scrollHeight;
  }, [messages, thinking]);

  const send = async () => {
    const q = input.trim();
    if (!q || !enabled || !core.blendResult || !core.blendInput) return;
    const next = [...messages, { role: 'user' as const, content: q }];
    setMessages(next);
    setInput('');
    setThinking(true);
    try {
      const turns: CopilotChatTurn[] = next
        .slice(0, -1)
        .map((m) => ({ role: m.role === 'assistant' ? 'assistant' : 'user', content: m.content }));
      const res = await postCopilotChat({
        schema_version: '1.0',
        correlation_id: crypto.randomUUID(),
        message: q,
        messages: turns,
        blend_input: core.blendInput,
        blend_result: core.blendResult,
      });
      setMessages((cur) => [...cur, { role: 'assistant', content: res.answer }]);
    } catch {
      toast('Copilot could not answer right now.', 'neg');
      setMessages((cur) => [
        ...cur,
        { role: 'assistant', content: 'I couldn’t reach the model service. Try again in a moment.' },
      ]);
    } finally {
      setThinking(false);
    }
  };

  return (
    <Card
      reveal
      title="Ask Copilot"
      actions={
        <Badge tone={enabled ? 'accent' : ''}>
          {enabled ? (
            <>
              <Icon name="check" size={12} /> Grounded in your squad
            </>
          ) : (
            <>
              <Icon name="lock" size={12} /> Apply a blend first
            </>
          )}
        </Badge>
      }
      bodyClassName="tight"
    >
      <div className="chat">
        <div className="chat-thread" ref={threadRef}>
          {messages.map((m, i) => (
            <div key={i} className={`msg ${m.role === 'assistant' ? 'ai' : 'user'}`}>
              {m.content}
            </div>
          ))}
          {thinking ? (
            <div className="msg ai thinking">
              <i />
              <i />
              <i />
              <span className="sr">Copilot is thinking</span>
            </div>
          ) : null}
        </div>
        <form
          className="chat-input"
          onSubmit={(e) => {
            e.preventDefault();
            void send();
          }}
        >
          <input
            className="input"
            placeholder={enabled ? 'Ask about captaincy, transfers, chips…' : 'Apply a model blend to unlock the assistant'}
            value={input}
            disabled={!enabled}
            aria-label="Ask Copilot a question"
            autoComplete="off"
            onChange={(e) => setInput(e.target.value)}
          />
          <button className="btn btn-primary btn-icon" type="submit" disabled={!enabled} aria-label="Send">
            <Icon name="send" size={18} />
          </button>
        </form>
      </div>
    </Card>
  );
}

/* ---------------------------------------------------------------- optimize */

function OptimizeButton({ variant = 'full' }: { variant?: 'full' | 'quick' }) {
  const core = useCore();
  const toast = useToast();
  const { teamId } = useTeamId();
  const [open, setOpen] = useState(false);
  const [weeks, setWeeks] = useState(5);
  const [running, setRunning] = useState(false);

  const run = async () => {
    setRunning(true);
    try {
      await runAirsenal({
        action: 'optimize',
        weeks_ahead: weeks,
        fpl_team_id: teamId ? Number(teamId) : null,
      });
      await runAirsenal({ action: 'export', fpl_team_id: teamId ? Number(teamId) : null }).catch(() => {});
      toast(`Optimization complete across ${weeks} GWs.`, 'pos');
      core.refresh();
      setOpen(false);
    } catch {
      toast('Optimization failed. Check the backend AIrsenal logs.', 'neg');
    } finally {
      setRunning(false);
    }
  };

  const label = variant === 'quick' ? 'Run AIrsenal optimization' : 'Run AIrsenal optimization';

  return (
    <>
      <button className={variant === 'quick' ? 'btn btn-ghost' : 'btn btn-ghost'} style={variant === 'quick' ? { justifyContent: 'flex-start', textAlign: 'left' } : undefined} type="button" onClick={() => setOpen(true)}>
        <Icon name="optimize" size={16} />
        {variant === 'quick' ? (
          <span className="od-stack" style={{ gap: 0, alignItems: 'flex-start' }}>
            <span>{label}</span>
            <span className="tiny faint" style={{ fontWeight: 400 }}>
              Multi-gameweek solver
            </span>
          </span>
        ) : (
          label
        )}
      </button>

      {open ? (
        <div className="scrim" onClick={() => !running && setOpen(false)}>
          <div className="dialog" role="dialog" aria-modal="true" aria-label="AIrsenal optimization" onClick={(e) => e.stopPropagation()}>
            <div className="dialog-head">
              <h3>AIrsenal optimization</h3>
              <span className="spacer" />
              <button className="btn btn-icon btn-ghost" type="button" onClick={() => !running && setOpen(false)} aria-label="Close">
                <Icon name="x" />
              </button>
            </div>
            <div className="dialog-body od-stack" style={{ gap: 16 }}>
              <p className="small muted">
                The solver searches transfer combinations across the next N gameweeks and returns the highest
                projected total, accounting for free transfers and −4 hits.
              </p>
              <div className="field">
                <span className="label">Weeks ahead</span>
                <div className="od-row" style={{ gap: 12 }}>
                  <button className="btn btn-icon btn-ghost" type="button" disabled={weeks <= 1} onClick={() => setWeeks((w) => Math.max(1, w - 1))} aria-label="Fewer weeks">
                    <Icon name="minus" />
                  </button>
                  <span className="mono" style={{ fontSize: 'var(--fs-2xl)', minWidth: 64, textAlign: 'center' }}>
                    {weeks}
                  </span>
                  <button className="btn btn-icon btn-ghost" type="button" disabled={weeks >= 38} onClick={() => setWeeks((w) => Math.min(38, w + 1))} aria-label="More weeks">
                    <Icon name="plus" />
                  </button>
                  <input type="range" min={1} max={38} value={weeks} aria-label="Weeks ahead" style={{ flex: 1 }} onChange={(e) => setWeeks(Number(e.target.value))} />
                </div>
                <span className="help">
                  Covers GW{core.nextGW}–GW{Math.min(38, core.nextGW + weeks - 1)}
                </span>
              </div>
              {running ? (
                <div className="od-stack" style={{ gap: 10, alignItems: 'center' }}>
                  <span style={{ color: 'var(--accent)', animation: 'spin 1.1s linear infinite', display: 'inline-block' }}>
                    <Icon name="optimize" size={36} />
                  </span>
                  <b>Solving…</b>
                  <div className="progress" style={{ width: '100%' }}>
                    <i style={{ width: '60%' }} />
                  </div>
                  <span className="tiny muted">This can take several minutes. Keep this tab open.</span>
                </div>
              ) : (
                <Alert tone="info" icon="info">
                  <span className="small">Longer horizons find better chip timing but take noticeably longer to solve.</span>
                </Alert>
              )}
            </div>
            <div className="dialog-foot">
              <button className="btn btn-ghost" type="button" disabled={running} onClick={() => setOpen(false)}>
                Cancel
              </button>
              <button className="btn btn-primary" type="button" disabled={running} onClick={() => void run()} aria-busy={running}>
                <Icon name="optimize" size={16} /> {running ? 'Running…' : 'Run optimization'}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
