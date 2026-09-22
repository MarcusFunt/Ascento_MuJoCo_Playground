import type { HTMLAttributes } from 'react'
import { cn } from '../../lib/utils'

export function Badge({ className, ...props }: HTMLAttributes<HTMLSpanElement>) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full border border-border-strong bg-raised px-2.5 py-1 text-[11px] font-bold uppercase tracking-[0.08em] text-secondary',
        className,
      )}
      {...props}
    />
  )
}
