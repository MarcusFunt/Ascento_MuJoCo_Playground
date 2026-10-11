import { CircleHelp } from 'lucide-react'
import { Sparkline } from './Sparkline'
import { Progress } from './ui/progress'
import { Tooltip, TooltipContent, TooltipTrigger } from './ui/tooltip'
import { cn } from '../lib/utils'

export function MetricCard({
  label,
  value,
  secondary,
  values = [],
  help,
  tone = 'neutral',
  progress,
}: {
  label: string
  value: string
  secondary?: string
  values?: Array<number | null | undefined>
  help?: string
  tone?: 'neutral' | 'good' | 'warning' | 'danger'
  progress?: number
}) {
  const hasTrend = values.filter((value) => Number.isFinite(Number(value))).length >= 2
  const toneClass = {
    neutral: 'text-foreground',
    good: 'text-success',
    warning: 'text-warning',
    danger: 'text-danger',
  }[tone]
  return (
    <section className="min-w-0 border-t border-border pt-4">
      <div className="flex flex-col items-start gap-1 sm:flex-row sm:justify-between sm:gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-[0.09em] text-muted">
            {label}
            {help ? (
              <Tooltip>
                <TooltipTrigger className="control-focus rounded p-0.5 text-subtle hover:text-secondary" aria-label={`About ${label}`}>
                  <CircleHelp size={13} />
                </TooltipTrigger>
                <TooltipContent>{help}</TooltipContent>
              </Tooltip>
            ) : null}
          </div>
          <div className={cn('numeric mt-2 text-[25px] font-semibold leading-none tracking-[-0.04em] sm:text-[32px]', toneClass)}>{value}</div>
        </div>
        {secondary ? <div className="max-w-full text-left text-[11px] leading-relaxed text-muted sm:max-w-[42%] sm:text-right sm:text-xs">{secondary}</div> : null}
      </div>
      {progress !== undefined ? (
        <Progress value={progress} className="mt-5 h-1" />
      ) : hasTrend ? (
        <Sparkline values={values} className="mt-3 text-secondary" />
      ) : null}
    </section>
  )
}
