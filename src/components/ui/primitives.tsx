import type { HTMLAttributes, ElementType, ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface DashboardCardProps extends HTMLAttributes<HTMLElement> {
  as?: ElementType;
}

export function DashboardCard({ as: Comp = 'div', className, ...props }: DashboardCardProps) {
  return (
    <Comp
      className={cn('overflow-hidden rounded-2xl border border-white/8 bg-slate-900 shadow-xl', className)}
      {...props}
    />
  );
}

interface DashboardHeaderProps {
  title: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
  className?: string;
}

export function DashboardHeader({ title, description, action, className }: DashboardHeaderProps) {
  return (
      <div
        className={cn(
          'flex items-start justify-between gap-3 border-b border-white/6 px-5 py-4',
          className,
        )}
      >
        <div className="min-w-0">
          {typeof title === 'string' ? (
            <h3 className="text-sm font-bold uppercase leading-[1.33] tracking-[0.1em] text-white">
              {title}
            </h3>
          ) : (
            title
          )}
          {description ? (
            <p className="mt-1 line-clamp-2 text-xs text-slate-400">{description}</p>
          ) : null}
        </div>
        {action ? <div className="shrink-0">{action}</div> : null}
      </div>
  );
}
