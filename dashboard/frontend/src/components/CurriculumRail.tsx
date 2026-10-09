import { Check, LockKeyhole } from 'lucide-react'
import type { Curriculum, GeneralistCurriculum, HorizonCurriculum, RecoveryDifficulty } from '../types'
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


function HorizonCompact({ curriculum }: { curriculum: HorizonCurriculum }) {
  const promotion = curriculum.promotion
  const recovery = curriculum.secondary
  return (
    <>
      <div className="mt-5 grid grid-cols-4">
        {curriculum.stages.map((stage, index) => (
          <div key={stage.value} className="relative">
            {index < curriculum.stages.length - 1 ? (
              <div className={`absolute left-1/2 right-[-50%] top-[9px] h-px ${stage.state === 'complete' ? 'bg-success/65' : 'bg-border-strong'}`} />
            ) : null}
            <div className="relative z-10 flex flex-col items-center text-center">
              <div className={`flex h-[19px] w-[19px] items-center justify-center rounded-full border text-[9px] ${stage.state === 'complete' ? 'border-success bg-success text-background' : stage.state === 'current' ? 'border-foreground bg-foreground text-background' : 'border-border-strong bg-panel text-muted'}`}>
                {stage.state === 'complete' ? <Check size={11} strokeWidth={3} /> : stage.index}
              </div>
              <strong className={`mt-1.5 text-xs ${stage.state === 'current' ? 'text-foreground' : 'text-secondary'}`}>{stage.label}</strong>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-5 grid overflow-hidden rounded-lg border border-border bg-border sm:grid-cols-2 xl:grid-cols-4">
        {[
          ['Promotion', `${promotion.qualified_windows} / ${promotion.required_windows}`, 'qualified windows'],
          ['Timeout', fmtRatioPercent(promotion.timeout_fraction), `target ≥ ${fmtRatioPercent(promotion.timeout_threshold)}`],
          ['Quality', fmtRatioPercent(promotion.quality_fraction), `target ≥ ${fmtRatioPercent(promotion.quality_threshold)}`],
          recovery
            ? ['Reset difficulty', fmtRatioPercent(recovery.hard_fraction), `${fmtNumber(recovery.control_steps, 0)} control steps`]
            : ['Regression', `${curriculum.demotion.failed_windows} / ${curriculum.demotion.required_windows}`, 'severe windows'],
        ].map(([label, value, note]) => (
          <div key={String(label)} className="bg-panel px-4 py-3">
            <span className="block text-[10px] font-bold uppercase tracking-[0.08em] text-muted">{label}</span>
            <strong className="numeric mt-1 block text-lg">{value}</strong>
            <span className="mt-0.5 block text-[11px] text-muted">{note}</span>
          </div>
        ))}
      </div>
    </>
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
      {curriculum.candidate_checkpoint ? (
        <div className="mt-5 rounded-lg border border-border bg-background/35 p-4 text-sm text-secondary">
          Long-horizon candidate <code className="text-foreground">{curriculum.candidate_checkpoint}</code>.
          {' '}Evaluate it deterministically with <code className="text-foreground">balance_gate_v5</code> before selecting it.
        </div>
      ) : null}
      {curriculum.secondary ? <RecoveryRamp value={curriculum.secondary} /> : null}
    </>
  )
}

function GeneralistRail({ curriculum, compact }: { curriculum: GeneralistCurriculum; compact: boolean }) {
  const stage = curriculum.stage
  const fraction = (value?: number | null) => value === null || value === undefined ? '—' : fmtRatioPercent(value)
  return (
    <div className={compact ? 'mt-4' : 'mt-6'}>
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="text-sm text-secondary">
          {stage === null || stage === undefined ? 'Waiting for curriculum telemetry' : `Goal-mix stage ${stage + 1} of ${curriculum.stage_count}`}
          <span className="mx-2 text-muted">·</span>
          short {fraction(curriculum.goal_fractions.short)} / medium {fraction(curriculum.goal_fractions.medium)} / long {fraction(curriculum.goal_fractions.long)}
        </div>
        <div className="text-xs text-muted">Gate-like episodes {fraction(curriculum.gate_like_fraction)}</div>
      </div>
      <div className="mt-4 grid grid-cols-3 gap-2" aria-label="Generalist curriculum stage">
        {[0, 1, 2].map((index) => {
          const state = stage === null || stage === undefined ? 'waiting' : index < stage ? 'complete' : index === stage ? 'current' : 'upcoming'
          const longGoal = index === 0 ? 0.10 : index === 1 ? 0.25 : 0.40
          return <div key={index} className={`rounded-md border px-3 py-2 text-center text-xs ${state === 'current' ? 'border-foreground bg-foreground/10 text-foreground' : state === 'complete' ? 'border-success/40 text-success' : 'border-border text-muted'}`}><strong className="block">Stage {index + 1}</strong><span className="mt-0.5 block">long goal share {fmtRatioPercent(longGoal)}</span><span className="mt-0.5 block uppercase tracking-wide">{state}</span></div>
        })}
      </div>

      <div className="mt-5 grid gap-3 lg:grid-cols-3">
        {curriculum.gates.map((gate) => (
          <article key={gate.id} className="rounded-lg border border-border bg-background/35 p-3">
            <div className="text-xs font-semibold text-secondary">{gate.label}</div>
            <div className="mt-2 flex items-baseline justify-between gap-2">
              <strong className="numeric text-lg">{fraction(gate.estimate)}</strong>
              <span className={gate.state === 'pass' ? 'text-xs text-success' : gate.state === 'below_threshold' ? 'text-xs text-warning' : 'text-xs text-muted'}>{gate.state === 'pass' ? 'meets gate' : gate.state === 'below_threshold' ? 'below gate' : 'waiting'}</span>
            </div>
            <div className="mt-1 text-[11px] text-muted">Wilson LCB ≥ {fraction(gate.threshold)} · {gate.samples ?? '—'} / {gate.minimum_samples} {gate.grain}</div>
            {gate.successes !== null && gate.successes !== undefined ? <div className="mt-1 text-[11px] text-muted">{gate.successes} observed successes in the latest emitted window</div> : null}
          </article>
        ))}
      </div>

      <div className="mt-5">
        <h3 className="text-xs font-bold uppercase tracking-[0.08em] text-muted">Fixed-cohort episode outcomes</h3>
        <div className="mt-2 grid gap-2 md:grid-cols-3">
          {curriculum.cohorts.map((cohort) => <article key={cohort.id} className="rounded-lg border border-border bg-background/35 p-3">
            <div className="text-sm font-semibold">{cohort.label}</div>
            <div className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-xs text-secondary">
              <span>Episodes</span><strong className="numeric text-right">{cohort.episode_count ?? '—'}</strong>
              <span>Episode arrivals</span><strong className="numeric text-right">{cohort.arrival_count ?? '—'}{cohort.arrival_rate === null || cohort.arrival_rate === undefined ? '' : ` · ${fraction(cohort.arrival_rate)}`}</strong>
              <span>Recovery / settled</span><strong className="numeric text-right">{cohort.recovery_count ?? '—'} / {cohort.settled_stop_count ?? '—'}</strong>
              <span>Falls / timeouts</span><strong className="numeric text-right">{cohort.fall_count ?? '—'} / {cohort.timeout_count ?? '—'}</strong>
            </div>
            <div className="mt-2 text-[10px] text-muted">Episode-level counters</div>
          </article>)}
        </div>
      </div>

      <div className="mt-5">
        <h3 className="text-xs font-bold uppercase tracking-[0.08em] text-muted">Navigation target-attempt outcomes · generalist cohort</h3>
        <div className="mt-2 grid gap-2 xl:grid-cols-3">
          {curriculum.attempt_bands.map((band) => <article key={band.id} className="rounded-lg border border-border bg-background/35 p-3">
            <div className="flex items-baseline justify-between gap-2"><strong className="text-sm">{band.label}</strong><span className="text-xs text-muted">{band.attempts ?? '—'} attempts</span></div>
            <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
              <span className="text-muted">Arrived</span><strong className="numeric text-right">{band.arrivals ?? '—'} · {fraction(band.arrival_rate)}</strong>
              <span className="text-muted">Settled stops</span><strong className="numeric text-right">{band.settled_stops ?? '—'} · {fraction(band.settled_stop_rate)}</strong>
              <span className="text-muted">Final target error (mean / p95)</span><strong className="numeric text-right">{fmtNumber(band.mean_final_target_error_m, 3)} / {fmtNumber(band.p95_final_target_error_m, 3)} m</strong>
              <span className="text-muted">Arrival time (mean / p95)</span><strong className="numeric text-right">{fmtNumber(band.mean_time_to_arrival_s, 2)} / {fmtNumber(band.p95_time_to_arrival_s, 2)} s</strong>
              <span className="text-muted">Heading-valid / interruptions</span><strong className="numeric text-right">{band.heading_valid_at_settle ?? '—'} / {band.fall_interruptions ?? '—'} fall · {band.timeout_interruptions ?? '—'} timeout</strong>
            </div>
            <div className="mt-2 text-[10px] text-muted">Target-attempt denominators; missing distributions are not zero.</div>
          </article>)}
        </div>
      </div>
      {curriculum.telemetry.iteration !== null && curriculum.telemetry.iteration !== undefined ? <p className="mt-4 text-[11px] text-muted">{curriculum.telemetry.source} · iteration {fmtNumber(curriculum.telemetry.iteration, 0)} · {curriculum.telemetry.window_label}</p> : <p className="mt-4 text-xs text-muted">{curriculum.note}</p>}
    </div>
  )
}

export function CurriculumRail({
  curriculum,
  compact = false,
}: {
  curriculum?: Curriculum | null
  compact?: boolean
}) {
  if (!curriculum) {
    return (
      <section className={`rounded-xl border border-border bg-panel ${compact ? 'p-5' : 'p-6'}`}>
        <div className="text-xs font-bold uppercase tracking-[0.1em] text-muted">Curriculum</div>
        <p className="mt-3 text-sm text-secondary">No curriculum state is available yet.</p>
      </section>
    )
  }

  return (
    <section className={`rounded-xl border border-border bg-panel ${compact ? 'p-5' : 'p-6'}`}>
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <div className="text-xs font-bold uppercase tracking-[0.1em] text-muted">Curriculum</div>
          <h2 className={`mt-1 font-semibold tracking-[-0.02em] ${compact ? 'text-lg' : 'text-xl'}`}>{curriculum.label}</h2>
        </div>
        {curriculum.kind === 'horizon' ? (
          <div className="text-right">
            <span className="text-xs text-muted">Stage</span>
            <div className="numeric text-lg font-semibold">{curriculum.stage} / {curriculum.stage_count}</div>
          </div>
        ) : null}
        {curriculum.kind === 'generalist' && curriculum.stage !== null && curriculum.stage !== undefined ? (
          <div className="text-right"><span className="text-xs text-muted">Stage</span><div className="numeric text-lg font-semibold">{curriculum.stage + 1} / {curriculum.stage_count}</div></div>
        ) : null}
      </div>

      {curriculum.kind === 'horizon' ? (
        compact ? <HorizonCompact curriculum={curriculum} /> : <HorizonRail curriculum={curriculum} />
      ) : null}

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

      {curriculum.kind === 'generalist' ? <GeneralistRail curriculum={curriculum} compact={compact} /> : null}

      {curriculum.kind === 'static' ? (
        <p className="mt-5 text-sm leading-relaxed text-secondary">{curriculum.note}</p>
      ) : null}
    </section>
  )
}
