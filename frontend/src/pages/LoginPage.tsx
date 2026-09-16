import { useState, type FormEvent } from 'react';
import { useNavigate, Navigate } from 'react-router-dom';
import { Icon, LogoMark } from '../components/Icon';
import { useTeamId } from '../context/TeamIdContext';

export default function LoginPage() {
  const { teamId, setTeamId } = useTeamId();
  const navigate = useNavigate();
  const [value, setValue] = useState('');
  const [error, setError] = useState(false);
  const [loading, setLoading] = useState(false);

  if (teamId) {
    return <Navigate to="/home" replace />;
  }

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const val = value.trim();
    if (!/^\d{5,9}$/.test(val)) {
      setError(true);
      return;
    }
    setError(false);
    setLoading(true);
    window.setTimeout(() => {
      setTeamId(val);
      navigate('/home');
    }, 600);
  };

  return (
    <div className="login">
      <div className="login-hero">
        <div className="od-row" style={{ gap: 12 }}>
          <LogoMark size={56} />
          <div>
            <div
              className="word"
              style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 'var(--fs-xl)' }}
            >
              FPL <span style={{ color: 'var(--accent)' }}>Copilot</span>
            </div>
            <div className="tiny faint">Your AI co-manager</div>
          </div>
        </div>
        <div className="od-stack" style={{ gap: 20, maxWidth: 560 }}>
          <h1>Every point you leave on the bench is a point someone else took.</h1>
          <p className="muted">
            Copilot reads your squad against official FPL data and the AIrsenal expected-points models,
            then tells you what to do — and shows its working when you want it.
          </p>
          <div className="od-stack" style={{ gap: 12 }}>
            <div className="feat">
              <span className="dot" />
              <div>
                <b>One clear call per screen</b>
                <div className="small muted">Captain, transfers and chip timing, ranked by expected points.</div>
              </div>
            </div>
            <div className="feat">
              <span className="dot" />
              <div>
                <b>Evidence on demand</b>
                <div className="small muted">Every recommendation expands into the model reasoning behind it.</div>
              </div>
            </div>
            <div className="feat">
              <span className="dot" />
              <div>
                <b>Sandbox the future</b>
                <div className="small muted">Run transfers as a what-if before you spend a free transfer or take a hit.</div>
              </div>
            </div>
          </div>
        </div>
        <div className="od-row" style={{ gap: 32, flexWrap: 'wrap' }}>
          <div className="od-stat">
            <span className="mono" style={{ fontSize: 'var(--fs-2xl)', fontWeight: 700 }}>
              2
            </span>
            <span className="tiny muted">data brains</span>
          </div>
          <div className="od-stat">
            <span className="mono" style={{ fontSize: 'var(--fs-2xl)', fontWeight: 700 }}>
              4
            </span>
            <span className="tiny muted">blendable models</span>
          </div>
          <div className="od-stat">
            <span className="mono" style={{ fontSize: 'var(--fs-2xl)', fontWeight: 700 }}>
              38
            </span>
            <span className="tiny muted">gameweeks planned</span>
          </div>
        </div>
      </div>
      <div className="login-form">
        <div className="login-card">
          <div className="od-row" style={{ gap: 12 }}>
            <LogoMark size={40} />
            <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 'var(--fs-lg)' }}>
              FPL <span style={{ color: 'var(--accent)' }}>Copilot</span>
            </div>
          </div>
          <div className="od-stack" style={{ gap: 6 }}>
            <h2>Connect your team</h2>
            <p className="muted small">
              Sign in with your FPL Team ID. Copilot loads your squad, your bank and your chip status
              straight from the official game.
            </p>
          </div>
          <form onSubmit={submit} noValidate className="od-stack" style={{ gap: 16 }}>
            <div className="field">
              <label htmlFor="tid">FPL Team ID</label>
              <input
                id="tid"
                name="tid"
                className="input mono-input"
                inputMode="numeric"
                autoComplete="off"
                placeholder="0000000"
                value={value}
                aria-invalid={error}
                onChange={(e) => {
                  setValue(e.target.value);
                  setError(false);
                }}
              />
              {error ? (
                <span className="error-text">
                  <Icon name="alert" size={14} /> That doesn’t look like a Team ID. Enter the 6–9 digit
                  number from your FPL profile URL.
                </span>
              ) : (
                <span className="help">
                  Find it in the URL of your points page: fantasy.premierleague.com/entry/
                  <b>3714256</b>/event/12
                </span>
              )}
            </div>
            <button className={`btn btn-primary ${loading ? 'loading' : ''}`} type="submit" disabled={loading}>
              {loading ? 'Connecting' : 'Connect team'}
            </button>
          </form>
          <button className="btn btn-ghost" type="button" onClick={() => setValue('3714256')}>
            Use the demo team
          </button>
          <p className="tiny faint">
            Copilot is a planning tool. It never makes changes to your FPL team for you — you always
            confirm in the official game.
          </p>
        </div>
      </div>
    </div>
  );
}
