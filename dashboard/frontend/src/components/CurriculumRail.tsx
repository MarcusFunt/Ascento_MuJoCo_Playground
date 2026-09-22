import { Check, LockKeyhole } from 'lucide-react'
import type { Curriculum, HorizonCurriculum, RecoveryDifficulty } from '../types'
import { fmtNumber, fmtRatioPercent } from '../lib/utils'
import { Progress } from './ui/progress'

function ThresholdRow({
  label,
  value,
  threshold,
  passWhenAbove = true,
}: {
  label: string
  value?: number | null
  threshold: number
  passWhenAbove?: boolean
}) {
  const hasValue = value !== null && value !== undefined && Number.isFinite(value)
  const pass = hasValue && (passWhenAbove ? value >= threshold : value <= threshold)
  return (
    <div className="grid grid-cols-[1fr_auto_auto] items-center gap-3 py-2 text-sm">
      <span className="text-secondary">{label}</span>
      <span className="numeric font-semibold text-foreground">{hasValue ? fmtRatioPercent(value) : '—'}</span>
      <span className={pass ? 'text-success' : hasValue ? 'text-warning' : 'text-muted'}>
        {pass ? '✓' : hasValue ? `${passWhenAbove ? '≥' : '≤'} ${fmtRatioPercent(threshold)}` : 'waiting'}
      </span>
    </div>
  )
}

function RecoveryRamp({ value }: { value: RecoveryDifficulty }) {
  const degrees = value.pitch_max_rad * 180 / Math.PI
  return (
    <div className="mt-6 border-t border-border pt-5">
      <div className="flex items-end justify-between gap-4">
        <div>
          <div className="text-xs font-bold uppercase tracking-[0.09em] text-muted">{value.label}</div>
          <div className="numeric mt-2 text-2xl font-semibold">{fmtRatioPercent(value.hard_fraction)}</div>
        </div>
        <div className="text-right text-xs text-muted">
          {fmtNumber(value.control_steps, 0)} / {fmtNumber(value.ramp_control_steps, 0)} control steps
        </div>
      </div>
      <Progress value={value.progress * 100} className="mt-4" />
      <div className="mt-4 grid grid-cols-3 gap-3 text-sm">
        <div><span className="block text-xs text-muted">Pitch max</span><strong className="numeric mt-1 block">{fmtNumber(degrees, 1)}°</strong></div>
        <div><span className="block text-xs text-muted">Linear vx</span><strong className="numeric mt-1 block">±{fmtNumber(value.linear_x_max_m_s, 2)} m/s</strong></div>
        <div><span className="block text-xs text-muted">Angular</span><strong className="numeric mt-1 block">±{fmtNumber(value.angular_max_rad_s, 2)} rad/s</strong></div>
      </div>
    </div>
  )
}

function HorizonRail({ curriculum }: { curriculum: HorizonCurriculum }) {
  const promotion = curriculum.promotion
  return (
    <>
      <div className="mt-6 grid grid-cols-4">
        {curriculum.stages.map((stage, index) => (
          <div key={stage.value} className="relative">
            {index < curriculum.stages.length - 1 ? (
              <div className={`absolute left-1/2 right-[-50%] top-[11px] h-px ${stage.state === 'complete' ? 'bg-success/65' : 'bg-border-strong'}`} />
            ) : null}
            <div className="relative z-10 flex flex-col items-center text-center">
              <div className={`flex h-[23px] w-[23px] items-center justify-center rounded-full border text-[10px] ${stage.state === 'complete' ? 'border-success bg-success text-background' : stage.state === 'current' ? 'border-foreground bg-foreground text-background' : 'border-border-strong bg-panel text-muted'}`}>
                {stage.state === 'complete' ? <Check size={13} strokeWidth={3} /> : stage.index}
              </div>
              <strong className={`mt-2 text-sm ${stage.state === 'current' ? 'text-foreground' : 'text-secondary'}`}>{stage.label}</strong>
              <span className="mt-0.5 text-[11px] uppercase tracking-[0.08em] text-muted">
                {stage.state === 'complete' ? 'complete' : stage.state === 'current' ? 'current' : 'upcoming'}
              </span>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-7 grid gap-5 lg:grid-cols-[1.2fr_1fr]">
        <div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-secondary">Promotion streak</span>
            <strong className="numeric">{promotion.qualified_windows} / {promotion.required_windows} windows</strong>
          </div>
          <Progress value={(promotion.qualified_windows / Math.max(1, promotion.required_windows)) * 100} className="mt-3" />
          <div className="mt-3 divide-y divide-border">
            <ThresholdRow label="Timeout success" value={promotion.timeout_fraction} threshold={promotion.timeout_threshold} />
            <ThresholdRow label="Quality pass" value={promotion.quality_fraction} threshold={promotion.quality_threshold} />
          </div>
        </div>
        <div className="rounded-lg border border-border bg-background/35 p-4">
          <div className="text-xs font-bold uppercase tracking-[0.09em] text-muted">Regression guard</div>
          <div className="mt-3 flex items-end justify-between">
            <div>
              <span className="numeric text-2xl font-semibold">{curriculum.demotion.failed_windows}</span>
              <span className="text-secondary"> / {curriculum.demotion.required_windows}</span>
            </div>
            {curriculum.protected ? (
              <span className="flex items-center gap-1.5 text-xs font-semibold text-success"><LockKeyhole size={13} /> protected stage</span>
            ) : (
              <span className="text-xs text-muted">severe-failure windows</span>
            )}
          </div>
          <p className="mt-3 text-xs leading-relaxed text-muted">
            A severe window is below {fmtRatioPercent(curriculum.demotion.severe_timeout_threshold)} timeout completion.
          </p>
          {curriculum.transition && curriculum.transition !== 'held' ? (
            <div className="mt-3 rounded-md border border-warning/35 bg-warning/10 px-3 py-2 text-xs font-semibold text-warning">
              Last transition: {curriculum.transition}
            </div>
          ) : null}
        </div>
      </div>
      {curriculum.secondary ? <RecoveryRamp value={curriculum.secondary} /> : null}
    </>
  )
}

export function CurriculumRail({ curriculum }: { curriculum?: Curriculum | null }) {
  if (!curriculum) {
    return (
      <section className="rounded-xl border border-border bg-panel p-6">
        <div className="text-xs font-bold uppercase tracking-[0.1em] text-muted">Curriculum</div>
        <p className="mt-3 text-sm text-secondary">No curriculum state is available yet.</p>
      </section>
    )
  }

  return (
    <section className="rounded-xl border border-border bg-panel p-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <div className="text-xs font-bold uppercase tracking-[0.1em] text-muted">Curriculum</div>
          <h2 className="mt-1 text-xl font-semibold tracking-[-0.02em]">{curriculum.label}</h2>
        </div>
        {curriculum.kind === 'horizon' ? (
          <div className="text-right">
            <span className="text-xs text-muted">Stage</span>
            <div className="numeric text-lg font-semibold">{curriculum.stage} / {curriculum.stage_count}</div>
          </div>
        ) : null}
      </div>

      {curriculum.kind === 'horizon' ? <HorizonRail curriculum={curriculum} /> : null}

      {curriculum.kind === 'sequence' ? (
        <>
          <div className="mt-6 grid gap-2 md:grid-cols-5">
            {curriculum.stages.map((stage, index) => (
              <div key={stage.label} className="relative rounded-lg border border-border bg-background/35 p-4">
                <span className="numeric text-xs text-muted">0{index + 1}</span>
                <strong className="mt-3 block text-sm">{stage.label}</strong>
                <span className="mt-1 block text-xs leading-relaxed text-muted">{stage.detail}</span>
              </div>
            ))}
          </div>
          {curriculum.note ? <p className="mt-4 text-xs text-muted">{curriculum.note}</p> : null}
        </>
      ) : null}

      {curriculum.kind === 'static' ? (
        <p className="mt-5 text-sm leading-relaxed text-secondary">{curriculum.note}</p>
      ) : null}
    </section>
  )
}
