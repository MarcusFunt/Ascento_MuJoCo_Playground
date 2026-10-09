import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import { Activity, ArrowRight, Clock3, Cpu, Gauge, Plus, RefreshCw, ShieldAlert } from 'lucide-react'
import { api } from '../api'
import { CurriculumRail } from '../components/CurriculumRail'
import { MetricCard } from '../components/MetricCard'
import { NewRunDialog } from '../components/NewRunDialog'
import { PageHeader } from '../components/PageHeader'
import { StateBadge } from '../components/StateBadge'
import { Button } from '../components/ui/button'
import { Progress } from '../components/ui/progress'
import { fmtDuration, fmtNumber, fmtPercent, fmtRatioPercent, shortCommit } from '../lib/utils'
import type { HorizonCurriculum } from '../types'

export function OverviewPage() {
  const [newRunOpen, setNewRunOpen] = useState(false)
  const overview = useQuery({
    queryKey: ['overview'],
    queryFn: api.overview,
    refetchInterval: 5_000,
    staleTime: 2_000,
  })
  const health = useQuery({ queryKey: ['health'], queryFn: api.health, refetchInterval: 15_000 })
  const activity = useQuery({ queryKey: ['activity'], queryFn: api.activity, refetchInterval: 15_000 })
  const assessments = useQuery({ queryKey: ['assessments'], queryFn: api.assessments, refetchInterval: 15_000 })
  const overviewRunId = overview.data?.active_run?.id || overview.data?.recent_run?.id
  const checkpointEvidence = useQuery({
    queryKey: ['checkpoint-evidence', overviewRunId],
    queryFn: () => api.checkpointEvidence(overviewRunId!),
    enabled: Boolean(overviewRunId),
    staleTime: 30_000,
    refetchInterval: overview.data?.active_run ? 30_000 : false,
  })
  const evaluations = useQuery({
    queryKey: ['evaluations', 'overview-latest'],
    queryFn: () => api.evaluations(),
    staleTime: 30_000,
    refetchInterval: 30_000,
  })

  const data = overview.data
  const run = data?.active_run
  const series = data?.series || []
  const rewardValues = series.map((point) => point.reward)
  const episodeValues = series.map((point) => point.episode_length)
  const klValues = series.map((point) => point.kl)
  const entropyValues = series.map((point) => point.entropy)
  const ppoValues = series.map((point) => point.ppo_loss)
  const curriculum = data?.curriculum
  const horizon = curriculum?.kind === 'horizon' ? curriculum as HorizonCurriculum : null
  const activityUnverified = health.data?.components?.supervisor?.status === 'unavailable'

  if (overview.isLoading) {
    return <OverviewSkeleton />
  }

  return (
    <>
      <PageHeader
        eyebrow="Control room"
        title="Overview"
        description="The current training state, curriculum and critical learning signals — visible immediately."
        actions={
          <>
            <Button variant="ghost" onClick={() => overview.refetch()} disabled={overview.isFetching}><RefreshCw size={16} /> Refresh</Button>
            <Button variant="primary" onClick={() => setNewRunOpen(true)}><Plus size={16} /> Start training</Button>
          </>
        }
      />

      {overview.error ? (
        <div className="mb-6 rounded-xl border border-danger/45 bg-danger/10 p-4 text-sm text-danger">{overview.error.message}</div>
      ) : null}

      <section className="mb-6 grid gap-3 sm:grid-cols-2 xl:grid-cols-5" aria-label="Current system activity" aria-live="polite">
        <ActivityMetric
          label="Trainer"
          status={activity.data?.trainer.status || 'checking'}
          detail={activity.data?.trainer.verified
            ? `${activity.data.trainer.active_runs?.length || 0} managed run(s) · process state checked`
            : activity.data?.trainer.indexed_active_runs?.length
              ? `${activity.data.trainer.indexed_active_runs.length} indexed active marker(s); host state unverified`
              : activity.data?.trainer.message || 'Host process state is not verified'}
        />
        <ActivityMetric label="Evaluator" status={activity.data?.evaluator.status || 'checking'} detail={activity.data?.evaluator.message || 'No evaluator state received'} />
        <ActivityMetric
          label="Viewer"
          status={activity.data?.viewer.status || 'checking'}
          detail={activity.data?.viewer.items?.[0]?.checkpoint || activity.data?.viewer.message || (activity.data?.viewer.verified ? 'Managed viewer status verified' : 'Viewer state unverified')}
        />
        <ActivityMetric
          label="GPU"
          status={activity.data?.gpu.available ? 'available' : 'unknown'}
          detail={activity.data?.gpu.gpus?.[0]
            ? `${activity.data.gpu.gpus[0].name} · ${fmtNumber(activity.data.gpu.gpus[0].utilization_percent, 0)}% · ${fmtNumber(activity.data.gpu.gpus[0].memory_used_mb, 0)} MB`
            : activity.data?.gpu.error || 'GPU telemetry is unavailable'}
        />
        <ActivityMetric
          label="Index"
          status={activity.data?.index.status || 'checking'}
          detail={`${activity.data?.index.source || 'source pending'} · ${activity.data?.index.indexed_run_count ?? '—'} runs${activity.data?.index.source_conflicts ? ` · ${activity.data.index.source_conflicts} source conflict(s)` : ''}`}
        />
      </section>
      {activity.error ? <div className="mb-6 rounded-xl border border-warning/35 bg-warning/5 p-3 text-sm text-warning">Activity snapshot unavailable: {activity.error.message}</div> : null}
      {assessments.data?.assessments.length ? (
        <section className="mb-6 rounded-xl border border-border bg-panel p-5" aria-label="Evidence based assessments" aria-live="polite">
          <div className="flex items-center justify-between gap-3">
            <div>
              <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.1em] text-muted"><ShieldAlert size={14} /> Evidence based assessments</div>
              <p className="mt-1 text-xs text-muted">Read-only findings from current system and saved artifacts.</p>
            </div>
            <span className="text-xs text-muted">Updated {new Date(assessments.data.assessed_at * 1000).toLocaleTimeString()}</span>
          </div>
          <ul className="mt-4 divide-y divide-border">
            {assessments.data.assessments.slice(0, 6).map((item) => (
              <li key={item.id} className="grid gap-2 py-3 first:pt-0 md:grid-cols-[minmax(0,1fr)_auto] md:items-start">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className={`rounded border px-2 py-0.5 text-[10px] font-bold uppercase ${item.severity === 'critical' ? 'border-danger/40 bg-danger/10 text-danger' : item.severity === 'warning' ? 'border-warning/40 bg-warning/10 text-warning' : 'border-border-strong bg-raised text-secondary'}`}>{item.severity}</span>
                    <strong className="text-sm">{item.headline}</strong>
                  </div>
                  <p className="mt-1 text-sm text-secondary">{item.explanation}</p>
                  <p className="mt-1 text-xs text-muted">Next: {item.recommended_action}</p>
                </div>
                {item.evidence_refs[0] ? <a href={item.evidence_refs[0].href} className="control-focus inline-flex min-h-9 items-center gap-1 text-xs font-semibold text-secondary hover:text-foreground">{item.evidence_refs[0].label}<ArrowRight size={13} /></a> : null}
              </li>
            ))}
          </ul>
        </section>
      ) : assessments.error ? (
        <div className="mb-6 rounded-xl border border-warning/35 bg-warning/5 p-3 text-sm text-warning">Assessments unavailable: {assessments.error.message}</div>
      ) : null}

      {!run ? (
        <section className="rounded-xl border border-border bg-panel p-6 lg:p-7">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.1em] text-muted"><Gauge size={15} /> Control room</div>
              <h2 className="mt-2 text-2xl font-semibold tracking-[-0.03em]">{activityUnverified ? 'Training state unverified' : 'Training is idle'}</h2>
              <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted">
                {activityUnverified
                  ? 'No indexed active run is present. The host supervisor is unavailable, so host process activity cannot be confirmed.'
                  : 'Use the latest checkpoint and saved evaluation as separate evidence. Reward and recency alone do not establish policy quality.'}
              </p>
            </div>
            <div className="flex gap-2">
              <Button variant="primary" onClick={() => setNewRunOpen(true)}><Plus size={16} /> Start training</Button>
              <Link to="/runs" className="control-focus inline-flex h-10 items-center gap-2 rounded-lg border border-border-strong bg-raised px-4 text-sm font-semibold text-foreground hover:bg-hover">Browse runs <ArrowRight size={15} /></Link>
            </div>
          </div>

          <div className="mt-6 grid gap-4 xl:grid-cols-3">
            <section className="rounded-lg border border-border bg-background/35 p-4">
              <div className="text-xs font-bold uppercase tracking-[0.08em] text-muted">Most recent run</div>
              {data?.recent_run ? (
                <>
                  <Link to="/runs/$runId" params={{ runId: data.recent_run.id }} className="control-focus mt-3 block truncate font-semibold hover:underline">{data.recent_run.display_name || data.recent_run.name}</Link>
                  <div className="mt-1 text-xs text-muted">{data.recent_run.task || data.recent_run.stage || 'Unknown task'} · source {shortCommit(data.recent_run.repository_version?.run_commit)}</div>
                  <StateBadge state={data.recent_run.state} stale={data.recent_run.stale} />
                </>
              ) : <p className="mt-3 text-sm text-muted">No indexed runs yet.</p>}
            </section>

            <section className="rounded-lg border border-border bg-background/35 p-4">
              <div className="text-xs font-bold uppercase tracking-[0.08em] text-muted">Newest stable checkpoint</div>
              {checkpointEvidence.data ? (
                <>
                  <Link to="/runs/$runId" params={{ runId: checkpointEvidence.data.run_id }} className="control-focus mt-3 block truncate font-semibold hover:underline">{checkpointEvidence.data.relative_path}</Link>
                  <div className="mt-1 text-xs text-muted">Iteration {checkpointEvidence.data.iteration ?? '—'} · SHA {shortCommit(checkpointEvidence.data.sha256)}</div>
                  <div className="mt-2 text-xs text-secondary">Selection: {checkpointEvidence.data.selection_status.replaceAll('_', ' ')} · visual review: not recorded</div>
                  <div className="mt-2 text-xs text-secondary">{checkpointEvidence.data.evaluations.length ? `${checkpointEvidence.data.evaluations.length} matching evaluation(s) by checkpoint hash` : 'No matching evaluation by checkpoint hash'}</div>
                </>
              ) : <p className="mt-3 text-sm text-muted">{checkpointEvidence.isLoading ? 'Checking stable checkpoints…' : checkpointEvidence.error ? 'No stable checkpoint could be verified.' : 'No stable checkpoint is available.'}</p>}
            </section>

            <section className="rounded-lg border border-border bg-background/35 p-4">
              <div className="text-xs font-bold uppercase tracking-[0.08em] text-muted">Newest saved evaluation</div>
              {evaluations.data?.evaluations[0] ? (
                <>
                  <Link to="/evaluations/$evaluationId" params={{ evaluationId: evaluations.data.evaluations[0].evaluation_id }} className="control-focus mt-3 block truncate font-semibold hover:underline">{evaluations.data.evaluations[0].suite_id}</Link>
                  <div className="mt-1 text-xs text-muted">{evaluations.data.evaluations[0].scenario_count ?? '—'} scenarios · {evaluations.data.evaluations[0].evidence_class.replace('_', ' ')}</div>
                  <div className="mt-2"><StateBadge state={evaluations.data.evaluations[0].status.toLowerCase()} /></div>
                  <p className="mt-2 text-xs text-muted">This may be for a different checkpoint; compare its SHA before drawing a conclusion.</p>
                </>
              ) : <p className="mt-3 text-sm text-muted">No saved evaluation artifacts found.</p>}
            </section>
          </div>
        </section>
      ) : (
        <div className="space-y-6">
          <section className="rounded-xl border border-border bg-panel p-6 lg:p-7">
            <div className="flex flex-wrap items-start justify-between gap-5">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-3">
                  <StateBadge state={run.state} stale={run.stale} />
                  <span className="text-xs font-bold uppercase tracking-[0.09em] text-muted">{run.task || run.stage || 'Training'}</span>
                </div>
                <h2 className="mt-3 max-w-4xl truncate text-[26px] font-semibold tracking-[-0.04em] sm:text-[30px]">{run.display_name}</h2>
                <div className="mt-3 flex flex-wrap gap-x-5 gap-y-2 text-sm text-muted">
                  <span className="flex items-center gap-1.5"><Activity size={14} /> Iteration <strong className="numeric text-secondary">{fmtNumber(run.iteration, 0)}</strong> / {fmtNumber(run.total_iterations, 0)}</span>
                  <span className="flex items-center gap-1.5"><Clock3 size={14} /> ETA <strong className="numeric text-secondary">{fmtDuration(run.eta_seconds)}</strong></span>
                  <span className="flex items-center gap-1.5"><Cpu size={14} /> <strong className="numeric text-secondary">{fmtNumber(run.throughput, 0)}</strong> env steps/s</span>
                </div>
              </div>
              <div className="flex gap-2">
                <Link to="/runs/$runId" params={{ runId: run.id }} className="control-focus inline-flex h-10 items-center gap-2 rounded-lg border border-border-strong bg-raised px-4 text-sm font-semibold hover:bg-hover">Run details</Link>
                <Link to="/analyze/$runId" params={{ runId: run.id }} className="control-focus inline-flex h-10 items-center gap-2 rounded-lg border border-foreground bg-foreground px-4 text-sm font-semibold text-background hover:bg-[#dfe2e5]">
                  Analyze <ArrowRight size={15} />
                </Link>
              </div>
            </div>
            <div className="mt-6 flex items-center justify-between gap-4">
              <Progress value={Number(run.percent_complete || 0)} className="h-2 flex-1" />
              <strong className="numeric whitespace-nowrap text-sm">{fmtPercent(run.percent_complete, 1)}</strong>
            </div>

            <div className="mt-7 grid gap-x-7 gap-y-6 sm:grid-cols-2 xl:grid-cols-5">
              <MetricCard
                label="Reward"
                value={fmtNumber(run.reward, 4)}
                values={rewardValues}
                help="Total task reward. Use the trend with survival and curriculum progression; reward alone is not proof of task success."
              />
              <MetricCard
                label={horizon ? 'Timeout success' : 'Episode length'}
                value={horizon ? fmtRatioPercent(horizon.promotion.timeout_fraction) : fmtNumber(run.episode_length, 1)}
                secondary={horizon ? `target ≥ ${fmtRatioPercent(horizon.promotion.timeout_threshold)}` : 'environment steps'}
                values={horizon ? [] : episodeValues}
                progress={horizon?.promotion.timeout_fraction !== null && horizon?.promotion.timeout_fraction !== undefined ? horizon.promotion.timeout_fraction * 100 : undefined}
                tone={horizon?.promotion.timeout_fraction !== null && horizon?.promotion.timeout_fraction !== undefined && horizon.promotion.timeout_fraction < horizon.promotion.timeout_threshold ? 'warning' : 'neutral'}
                help={horizon ? 'Fraction of completed episodes that reached the current time horizon. This directly gates curriculum promotion.' : 'Mean episode duration. Sudden drops often indicate earlier failures.'}
              />
              <MetricCard
                label="KL"
                value={fmtNumber(run.kl, 5)}
                values={klValues}
                help="How far the updated policy moved from the previous policy. Large persistent spikes can indicate aggressive or unstable updates."
              />
              <MetricCard
                label="Entropy"
                value={fmtNumber(run.entropy, 4)}
                values={entropyValues}
                help="Policy exploration. A gradual decline can be normal; sudden collapse can remove useful exploration."
              />
              <MetricCard
                label="PPO loss"
                value={fmtNumber(run.ppo_loss, 5)}
                values={ppoValues}
                tone={Number(run.invalid_updates || 0) > 0 ? 'danger' : 'neutral'}
                secondary={Number(run.invalid_updates || 0) > 0 ? `${run.invalid_updates} invalid update(s)` : 'surrogate objective'}
                help="The PPO surrogate objective is a training diagnostic, not a score to minimize visually."
              />
            </div>
          </section>

          <section className="rounded-xl border border-border bg-panel p-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <div className="text-xs font-bold uppercase tracking-[0.1em] text-muted">Checkpoint and evidence</div>
                <p className="mt-1 text-sm text-secondary">
                  {checkpointEvidence.data
                    ? `Newest stable ${checkpointEvidence.data.relative_path} · iteration ${checkpointEvidence.data.iteration ?? '—'} · selection ${checkpointEvidence.data.selection_status.replaceAll('_', ' ')}`
                    : checkpointEvidence.isLoading ? 'Checking for the newest stable checkpoint…' : 'No stable checkpoint is currently verified.'}
                </p>
                {checkpointEvidence.data ? <p className="mt-1 break-all font-mono text-[11px] text-muted">SHA-256 {checkpointEvidence.data.sha256}</p> : null}
              </div>
              {checkpointEvidence.data?.evaluations[0] ? (
                <Link to="/evaluations/$evaluationId" params={{ evaluationId: checkpointEvidence.data.evaluations[0].evaluation_id }} className="control-focus inline-flex h-9 items-center gap-2 rounded-lg border border-border-strong bg-raised px-3 text-xs font-semibold">View matching evidence <ArrowRight size={14} /></Link>
              ) : <span className="text-xs text-warning">{checkpointEvidence.data ? 'No evaluation matches this checkpoint hash' : checkpointEvidence.error ? 'Evidence lookup unavailable' : ''}</span>}
            </div>
          </section>

          <CurriculumRail curriculum={curriculum} compact />

          <div className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
            <section className="rounded-xl border border-border bg-panel p-6">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs font-bold uppercase tracking-[0.1em] text-muted">Recent activity</div>
                  <h2 className="mt-1 text-xl font-semibold">Training events</h2>
                </div>
                <Activity size={18} className="text-muted" />
              </div>
              <div className="mt-5 divide-y divide-border">
                {(data?.events || []).length ? data!.events.map((event) => (
                  <div key={event.id} className="flex gap-4 py-3 first:pt-0">
                    <div className="mt-1 h-2 w-2 shrink-0 rounded-full bg-secondary" />
                    <div className="min-w-0 flex-1">
                      <div className="text-sm text-secondary">{event.message}</div>
                      <div className="mt-1 text-xs text-muted">{new Date(event.created_at * 1000).toLocaleString()}</div>
                    </div>
                  </div>
                )) : <p className="py-4 text-sm text-muted">No indexed events yet. State and curriculum transitions will appear here as they occur.</p>}
              </div>
            </section>

            <section className="rounded-xl border border-border bg-panel p-6">
              <div className="text-xs font-bold uppercase tracking-[0.1em] text-muted">Control-room status</div>
              <div className="mt-5 grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-border bg-border">
                {[
                  ['Runs', data?.counts.total ?? '—'],
                  ['Indexed active', data?.counts.active ?? '—'],
                  ['Errors', data?.counts.errors ?? '—'],
                  ['Outdated', data?.counts.outdated ?? '—'],
                ].map(([label, value]) => (
                  <div key={String(label)} className="bg-panel p-4">
                    <span className="block text-xs text-muted">{label}</span>
                    <strong className="numeric mt-1 block text-2xl">{String(value)}</strong>
                  </div>
                ))}
              </div>
              <div className="mt-5 rounded-lg border border-border bg-background/35 p-4">
                <span className="text-xs font-bold uppercase tracking-[0.08em] text-muted">Dashboard index</span>
                <div className="mt-2 flex items-center justify-between gap-3 text-sm">
                  <span className="text-secondary">{data?.database?.backend || 'filesystem fallback'}</span>
                  <span className={data?.database?.available ? 'text-success' : data?.database?.enabled ? 'text-warning' : 'text-muted'}>
                    {data?.database?.available ? 'connected' : data?.database?.enabled ? 'degraded' : 'optional'}
                  </span>
                </div>
              </div>
            </section>
          </div>
        </div>
      )}

      <NewRunDialog open={newRunOpen} onOpenChange={setNewRunOpen} />
    </>
  )
}

function OverviewSkeleton() {
  return (
    <div className="animate-pulse">
      <div className="h-4 w-28 rounded bg-raised" />
      <div className="mt-3 h-10 w-64 rounded bg-raised" />
      <div className="mt-10 h-[330px] rounded-xl border border-border bg-panel" />
      <div className="mt-6 h-[330px] rounded-xl border border-border bg-panel" />
    </div>
  )
}

function ActivityMetric({ label, status, detail }: { label: string; status: string; detail: string }) {
  const tone = ['active', 'healthy', 'available'].includes(status)
    ? 'text-success'
    : ['unknown', 'degraded', 'unavailable', 'stale'].includes(status)
      ? 'text-warning'
      : 'text-secondary'
  return (
    <div className="min-w-0 rounded-lg border border-border bg-panel px-4 py-3">
      <div className="text-[11px] font-bold uppercase tracking-[0.08em] text-muted">{label}</div>
      <strong className={`mt-1 block text-sm capitalize ${tone}`}>{status}</strong>
      <p className="mt-1 truncate text-xs text-muted" title={detail}>{detail}</p>
    </div>
  )
}
