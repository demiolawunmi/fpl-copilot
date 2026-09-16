import { createContext, useContext, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { Icon } from '../components/Icon';

type ToastTone = 'pos' | 'neg' | 'warn' | 'info';

interface ToastItem {
  id: number;
  message: ReactNode;
  tone?: ToastTone;
}

type PushToast = (message: ReactNode, tone?: ToastTone) => void;

const ToastContext = createContext<PushToast>(() => {});

const TONE_ICON = { neg: 'alert', warn: 'alert', info: 'info', pos: 'check' } as const;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const seq = useRef(0);

  const push: PushToast = (message, tone) => {
    const id = ++seq.current;
    setItems((list) => [...list, { id, message, tone }]);
    window.setTimeout(() => {
      setItems((list) => list.filter((t) => t.id !== id));
    }, 3800);
  };

  const root = typeof document !== 'undefined' ? document.getElementById('toast-root') : null;

  return (
    <ToastContext.Provider value={push}>
      {children}
      {root
        ? createPortal(
            <>
              {items.map((t) => (
                <div
                  key={t.id}
                  className="toast"
                  role="status"
                  style={{
                    borderLeftColor:
                      t.tone === 'neg'
                        ? 'var(--neg)'
                        : t.tone === 'warn'
                          ? 'var(--warn)'
                          : t.tone === 'info'
                            ? 'var(--info)'
                            : 'var(--accent)',
                  }}
                >
                  <Icon name={t.tone ? TONE_ICON[t.tone] : 'check'} size={18} />
                  <div>{t.message}</div>
                </div>
              ))}
            </>,
            root,
          )
        : null}
    </ToastContext.Provider>
  );
}

export function useToast(): PushToast {
  return useContext(ToastContext);
}
