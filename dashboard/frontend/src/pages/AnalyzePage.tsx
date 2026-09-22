import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate, useParams } from '@tanstack/react-router'
import { api } from '../api'
import { CurriculumRail } from '../components/CurriculumRail'
import { LiveLog } from '../components/LiveLog'
import { PageHeader } from '../components/PageHeader'
import { TelemetryChart } from '../components/TelemetryChart'
import { SelectInput } from '../components/ui/input'
import { fmtNumber } from '../lib/utils'

const CORE = [
  ['reward', 'Reward', 'Task reward over training. Interpret it with task success and curriculum progression.'],
  ['episode_length', 'Episode length', 'Mean episode duration in environment steps.'],
  ['ppo_loss', 'PPO / surrogate loss', 'Optimization diagnostic; the absolute value is not a score.'],
  ['entropy', 'Entropy', 'Policy exploration and stochasticity.'],
  ['kl', 'KL divergence', 'Magnitude of policy updates relative to the previous policy.'],
  ['clip_fraction', 'Clip fraction', 'Fraction of PPO samples whose update reached the clipping boundary.'],
] as const

const ADVANCED = [
  ['controller_request_saturation_fraction', 'Controller request saturation'],
  ['physical_saturation_fraction', 'Actuator output saturation'],
  ['effort_rms', 'Controller request RMS'],
  ['recovery_success', 'Recovery success'],
  ['recovery_time_s', 'Recovery time'],
  ['max_recovery_hold_s', 'Recovery hold duration'],
  ['recovery_dwell', 'Recovery dwell shaping'],
  ['post_landing_stability', 'Post-landing stability'],
] as const

export function AnalyzePage() {
  const params = useParams({ strict: false }) as { runId?: string }
  const navigate = useNavigate()
  const runs = useQuery({ queryKey: ['runs'], queryFn: api.runs, refetchInterval: 10_000 })
  const selectedId = params.runId || runs.data?.runs.find((run) => ['starting', 'running', 'stopping'].includes(run.state))?.id || runs.data?.runs[0]?.id || ''
  const detail = useQuery({ queryKey: ['run', selectedId], queryFn: () => api.run(selectedId), enabled: Boolean(selectedId), refetchInterval: 10_000 })
  const telemetry = useQuery({ queryKey: ['telemetry', selectedId], queryFn: () => api.telemetry(selectedId, 900), enabled: Boolean(selectedId), refetchInterval: 15_000 })
  const curriculum = useQuery({ queryKey: ['curriculum', selectedId], queryFn: () => api.curriculum(selectedId), enabled: Boolean(selectedId), refetchInterval: 10_000 })

  const records = telemetry.data?.records || []
  const availableAdvanced = useMemo(
    () => ADVANCED.filter(([metric]) => records.some((record) => Number.isFinite(Number(record.canonical_metrics?.[metric] ?? record.metrics?.[metric])))),
    [records],
  )

  const selectRun = (id: string) => {
    if (!id) return
    void navigate({ to: '/analyze/$runId', params: { runId: id } })
  }

  return (
    <>
      <PageHeader
        eyebrow="Deep telemetry"
        title="Analyze"
        description="Zoomable training history, curriculum context and the live trainer log."
        actions={
          <div className="min-w-[280px]">
            <SelectInput value={selectedId} onChange={(event) => selectRun(event.target.value)}>
              {!selectedId ? <option value="">No runs</option> : null}
              {(runs.data?.runs || []).map((run) => <option key={run.id} value={run.id}>{run.display_name}</option>)}
            </SelectInput>
          </div>
        }
      />

      {!selectedId ? (
        <div className="rounded-xl border border-border bg-panel p-10 text-center text-muted">No training runs are available.</div>
      ) : (
        <div className="space-y-6">
          <section className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-border bg-panel p-5">
            <div>
              <div className="text-xs font-bold uppercase tracking-[0.08em] text-muted">{String(detail.data?.run_info?.task || detail.data?.stage || 'Run')}</div>
              <h2 className="mt-1 text-xl font-semibold">{detail.data?.display_name || detail.data?.name || 'Loading…'}</h2>
            </div>
            <div className="flex flex-wrap gap-x-6 gap-y-2 text-sm text-muted">
              <span>Iteration <strong className="numeric text-foreground">{fmtNumber(detail.data?.telemetry?.iteration, 0)}</strong></span>
              <span>Source points <strong className="numeric text-foreground">{fmtNumber(telemetry.data?.source_records ?? records.length, 0)}</strong></span>
            </div>
          </section>

          <CurriculumRail curriculum={curriculum.data?.curriculum} />

          {telemetry.error ? <div className="rounded-xl border border-danger/40 bg-danger/10 p-4 text-sm text-danger">{telemetry.error.message}</div> : null}
          <div className="grid gap-5 xl:grid-cols-2">
            {CORE.map(([metric, title, description]) => <TelemetryChart key={metric} records={records} metric={metric} title={title} description={description} />)}
          </div>

          {availableAdvanced.length ? (
            <section>
              <div className="mb-4">
                <div className="text-xs font-bold uppercase tracking-[0.1em] text-muted">Advanced signals</div>
                <h2 className="mt-1 text-xl font-semibold">Task-specific diagnostics</h2>
              </div>
              <div className="grid gap-5 xl:grid-cols-2">
                {availableAdvanced.map(([metric, title]) => <TelemetryChart key={metric} records={records} metric={metric} title={title} />)}
              </div>
            </section>
          ) : null}

          <LiveLog runId={selectedId} />
        </div>
      )}
    </>
  )
}
