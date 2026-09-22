import { cn } from '../../lib/utils'

export function Progress({
  value,
  className,
}: {
  value: number
  className?: string
}) {
  const safe = Math.max(0, Math.min(100, Number.isFinite(value) ? value : 0))
  return (
    <div className={cn('h-1.5 overflow-hidden rounded-full bg-border', className)}>
      <div
        className="h-full rounded-full bg-foreground transition-[width] duration-300"
        style={{ width: `${safe}%` }}
      />
    </div>
  )
}
