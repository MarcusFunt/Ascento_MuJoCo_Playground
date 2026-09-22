import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import { Activity, ArrowRight, Clock3, Cpu, Gauge, Plus, RefreshCw } from 'lucide-react'
import { api } from '../api'
import { CurriculumRail } from '../components/CurriculumRail'
import { MetricCard } from '../components/MetricCard'
import { NewRunDialog } from '../components/NewRunDialog'
import { PageHeader } from '../components/PageHeader'
import { StateBadge } from '../components/StateBadge'
import { Button } from '../components/ui/button'
import { Progress } from '../components/ui/progress'
import { fmtDuration, fmtNumber, fmtPercent, fmtRatioPercent } from '../lib/utils'
import type { HorizonCurriculum } from '../types'

export function OverviewPage() {
  const [newRunOpen, setNewRunOpen] = useState(false)
  const overview = useQuery({
    queryKey: ['overview'],
    queryFn: api.overview,
    refetchInterval: 5_000,
    staleTime: 2_000,
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

      {!run ? (
        <section className="rounded-xl border border-border bg-panel px-6 py-14 text-center">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl border border-border-strong bg-raised"><Gauge size={22} className="text-secondary" /></div>
          <h2 className="mt-5 text-2xl font-semibold tracking-[-0.03em]">No active training</h2>
          <p className="mx-auto mt-2 max-w-xl text-sm leading-relaxed text-muted">
            Start a managed run, or inspect the most recent archived work from the Runs page.
          </p>
          <div className="mt-6 flex justify-center gap-3">
            <Button variant="primary" onClick={() => setNewRunOpen(true)}><Plus size={16} /> Start training</Button>
            <Link to="/runs" className="control-focus inline-flex h-10 items-center gap-2 rounded-lg border border-border-strong bg-raised px-4 text-sm font-semibold text-foreground hover:bg-hover">
              Browse runs <ArrowRight size={15} />
            </Link>
          </div>
          {data?.recent_run ? (
            <div className="mx-auto mt-10 max-w-2xl border-t border-border pt-5 text-left">
              <span className="text-xs font-bold uppercase tracking-[0.09em] text-muted">Most recent run</span>
              <Link to="/runs/$runId" params={{ runId: data.recent_run.id }} className="control-focus mt-2 flex items-center justify-between rounded-lg border border-border p-4 hover:bg-hover">
                <div>
                  <strong>{data.recent_run.display_name}</strong>
                  <span className="mt-1 block text-xs text-muted">{data.recent_run.task || data.recent_run.stage || 'Unknown task'}</span>
                </div>
                <StateBadge state={data.recent_run.state} stale={data.recent_run.stale} />
              </Link>
            </div>
          ) : null}
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

          <CurriculumRail curriculum={curriculum} />

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
                  ['Active', data?.counts.active ?? '—'],
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
