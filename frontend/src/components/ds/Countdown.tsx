import { useEffect, useState } from 'react';
import { countdown } from '../../lib/format';

/**
 * Live countdown. The formatted string is held in state and recomputed inside
 * the timer callback — computing it during render would let React Compiler
 * memoize the result (its inputs never change) and the value would freeze.
 */
export function Countdown({ target, className }: { target: string | number | null; className?: string }) {
  const [text, setText] = useState(() => countdown(target));

  useEffect(() => {
    let id = 0;
    const tick = () => {
      setText(countdown(target));
      id = window.setTimeout(tick, 1000);
    };
    id = window.setTimeout(tick, 1000);
    return () => window.clearTimeout(id);
  }, [target]);

  return <span className={className}>{text}</span>;
}
