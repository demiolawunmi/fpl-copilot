import { useNavigate } from 'react-router-dom';
import { Icon } from '../Icon';
import { Avatar, Badge, Crest, StatusBadge } from '../ds/atoms';
import { useSquad } from '../../context/SquadContext';
import { useToast } from '../../context/ToastContext';
import { useCore } from '../../context/CoreContext';
import { money, num } from '../../lib/format';
import type { Player } from '../../domain/types';

export function PlayerActionDialog({
  player,
  onClose,
}: {
  player: Player;
  onClose: () => void;
}) {
  const { xi, bench, captainId, viceId, setCaptain, setVice, swap, sandbox } = useSquad();
  const core = useCore();
  const toast = useToast();
  const navigate = useNavigate();

  const target = sandbox.on
    ? { xi: sandbox.xi, bench: sandbox.bench, captainId: sandbox.captainId, viceId: sandbox.viceId }
    : { xi, bench, captainId, viceId };

  const inXi = target.xi.includes(player.id);
  const inBench = target.bench.includes(player.id);
  const isCap = target.captainId === player.id;
  const isVice = target.viceId === player.id;

  const swapPool = (inXi ? target.bench : target.xi)
    .map((id) => core.playersById.get(id))
    .filter((p): p is Player => Boolean(p))
    .filter((p) => p.pos === player.pos || (player.pos !== 'GK' && p.pos !== 'GK'));

  return (
    <div className="scrim" onClick={onClose}>
      <div className="dialog" role="dialog" aria-modal="true" aria-label={player.name} onClick={(e) => e.stopPropagation()}>
        <div className="dialog-head">
          <Avatar player={player} size={40} />
          <Crest club={player.club} size={24} />
          <div className="od-fill">
            <h3 style={{ fontSize: 'var(--fs-base)' }}>{player.name}</h3>
            <div className="tiny muted">
              {player.club.name} · {player.pos} · {money(player.price)}
            </div>
          </div>
          <button className="btn btn-icon btn-ghost" type="button" onClick={onClose} aria-label="Close">
            <Icon name="x" />
          </button>
        </div>
        <div className="dialog-body od-stack" style={{ gap: 10 }}>
          <div className="od-row" style={{ gap: 8, flexWrap: 'wrap' }}>
            <StatusBadge player={player} />
            <Badge className="mono">{player.xpts.toFixed(1)} xPts</Badge>
            <Badge className="mono">{num(player.own, 1)}% owned</Badge>
          </div>
          <div className="grid grid-2" style={{ gap: 8 }}>
            <button
              className={`btn ${isCap ? 'btn-primary' : 'btn-ghost'}`}
              type="button"
              disabled={isCap}
              onClick={() => {
                setCaptain(player.id);
                toast(<b>{player.name}</b>, 'pos');
                onClose();
              }}
            >
              <Icon name="crown" size={15} /> {isCap ? 'Captain' : 'Set captain'}
            </button>
            <button
              className={`btn ${isVice ? 'btn-primary' : 'btn-ghost'}`}
              type="button"
              disabled={isVice}
              onClick={() => {
                setVice(player.id);
                toast(<b>{player.name}</b>, 'pos');
                onClose();
              }}
            >
              <Icon name="star" size={15} /> {isVice ? 'Vice-captain' : 'Set vice'}
            </button>
          </div>

          {inXi || inBench ? (
            <div className="od-stack" style={{ gap: 6, marginTop: 4 }}>
              <div className="tiny faint" style={{ textTransform: 'uppercase', letterSpacing: '.12em' }}>
                Swap with
              </div>
              {swapPool.length ? (
                swapPool.map((p) => (
                  <button
                    key={p.id}
                    className="od-row"
                    type="button"
                    style={{
                      gap: 12,
                      width: '100%',
                      textAlign: 'left',
                      padding: 10,
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--r-sm)',
                      background: 'transparent',
                      cursor: 'pointer',
                    }}
                    onClick={() => {
                      swap(player.id, p.id);
                      toast(
                        <>
                          Swapped <b>{player.name}</b> with <b>{p.name}</b>.
                        </>,
                      );
                      onClose();
                    }}
                  >
                    <Avatar player={p} size={22} />
                    <Crest club={p.club} size={16} />
                    <span className="od-fill">
                      <span className="small" style={{ fontWeight: 600, display: 'block' }}>
                        {p.name}
                      </span>
                      <span className="tiny faint">
                        {p.pos} · {money(p.price)} · {p.xpts.toFixed(1)} xPts
                      </span>
                    </span>
                    <span className={`pos-badge pos-${p.pos}`}>{p.pos}</span>
                  </button>
                ))
              ) : (
                <div className="tiny muted">No compatible swap available.</div>
              )}
            </div>
          ) : null}

          <button
            className="btn btn-primary"
            type="button"
            onClick={() => {
              onClose();
              navigate(`/player/${player.id}?from=command`);
            }}
          >
            <Icon name="eye" size={15} /> Open full profile
          </button>
        </div>
      </div>
    </div>
  );
}
