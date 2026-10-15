import { useQuery } from '@tanstack/react-query'
import { ArrowRight, Activity, Clock3, Cpu, Radio, ShieldCheck } from 'lucide-react'
import { api, type LiveTrainingRun } from '../api'
import { Sparkline } from './Sparkline'

const liveUrl = (id: string) => '/deep-analysis?live=' + encodeURIComponent(id)

export function useLiveTrainingSummary() {
  return useQuery({
    queryKey: ['active-training-summary'],
    queryFn: api.activeTrainingSummary,
    refetchInterval: 10_000,
    staleTime: 6_000,
  })
}

function amount(value: number | null | undefined, decimals = 0) {
  return value == null || !Number.isFinite(value) ? '—' : value.toLocaleString(undefined, { maximumFractionDigits: decimals, minimumFractionDigits: decimals })
}

function duration(value: number | null) {
  if (value == null) return 'Not scheduled'
  if (value <= 0) return 'Due now'
  const hours = Math.floor(value / 3600)
  const minutes = Math.floor((value % 3600) / 60)
  return hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`
}

function LiveHeading({ run }: { run: LiveTrainingRun }) {
  return <div className="min-w-0">
    <div className="flex flex-wrap items-center gap-2 text-xs font-bold uppercase tracking-wider text-success">
      <Radio size={15} aria-hidden="true" /> Streaming PPO
      <span className="rounded-md border border-border px-2 py-0.5 text-[10px] text-secondary">{run.root.toUpperCase()}</span>
    </div>
    <p className="mt-2 break-words text-base font-semibold leading-snug sm:text-lg">{run.label.split('/').slice(-2).join(' / ')}</p>
    <p className="mt-1 break-all font-mono text-[10px] text-muted" title={run.label}>{run.label}</p>
  </div>
}

export function ActiveTrainingPanel() {
  const { data, error } = useLiveTrainingSummary()
  const run = data?.runs[0]
  if (!run && !error) return null

  return <section className="mb-6 rounded-2xl border border-success/30 bg-panel p-4 sm:p-6" aria-label="Active training">
    <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
      <h2 className="flex items-center gap-2 text-lg font-semibold"><Activity size={19} className="text-success" /> Active training</h2>
      {data && data.runs.length > 1 && <span className="text-xs text-muted">{data.runs.length} live runs · newest first</span>}
    </div>
    {error && !run ? <p className="text-sm text-warning">Live training status is unavailable: {error.message}</p> : run && <>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <LiveHeading run={run} />
        <a href={liveUrl(run.id)} className="control-focus inline-flex min-h-11 items-center gap-2 rounded-lg bg-foreground px-4 py-2 text-sm font-semibold text-background hover:opacity-90">
          Open live analysis <ArrowRight size={17} />
        </a>
      </div>
      <div className="mt-5 grid grid-cols-2 gap-3 lg:grid-cols-3 xl:grid-cols-6">
        {[
          ['PPO iteration', amount(run.iteration)],
          ['Mean reward', amount(run.mean_reward, 2)],
          ['Curriculum', run.curriculum_stage == null ? 'Unavailable' : `Stage ${run.curriculum_stage} / ${(run.curriculum_stage_count ?? 3) - 1}`],
          ['GPU utilization', amount(data?.gpu?.utilization_percent) + (data?.gpu ? '%' : '')],
          ['GPU VRAM / temp', data?.gpu ? `${amount(data.gpu.memory_used_mib / 1024, 1)} GB · ${amount(data.gpu.temperature_c)}°C` : 'Unavailable'],
          ['Until final deadline', duration(run.deadline_remaining_seconds)],
        ].map(([label, value]) => <div key={label} className="min-w-0 rounded-xl border border-border bg-raised p-3">
          <div className="text-[11px] font-semibold text-muted">{label}</div>
          <div className="numeric mt-2 truncate text-base font-bold" title={value}>{value}</div>
        </div>)}
      </div>
      <div className="mt-5 grid gap-4 border-t border-border pt-4 lg:grid-cols-[1fr_auto] lg:items-center">
        <div className="min-w-0">
          <div className="flex items-center justify-between gap-3 text-xs text-secondary">
            <span className="font-semibold">Recent mean reward</span>
            <span>{run.reward_trend.length ? `${run.reward_trend.length} PPO updates` : 'History unavailable'}</span>
          </div>
          {run.reward_trend.length > 1 ? <Sparkline values={run.reward_trend.map(x => x.reward)} className="mt-2 h-12 text-info" />
            : <p className="mt-2 text-xs text-muted">No recent trainer history is mounted for this source.</p>}
        </div>
        <div className="flex flex-wrap gap-x-5 gap-y-2 text-xs text-secondary">
          <span className="flex items-center gap-1.5"><Clock3 size={14} /> Training cutoff: {duration(run.training_remaining_seconds)}</span>
          <span className="flex items-center gap-1.5"><Cpu size={14} /> GPU: {data?.gpu ? amount(data.gpu.memory_total_mib / 1024, 1) + ' GB total' : 'unavailable'}</span>
        </div>
      </div>
      {data && data.runs.length > 1 && <div className="mt-4 flex flex-wrap gap-2 border-t border-border pt-4">
        {data.runs.slice(1).map(other => <a key={other.id} href={liveUrl(other.id)} className="control-focus rounded-lg border border-border px-3 py-2 text-xs font-semibold hover:bg-hover">
          <Radio size={13} className="mr-1 inline text-success" /> {other.root.toUpperCase()} · {other.label.split('/').at(-1)} <ArrowRight size={12} className="ml-1 inline" />
        </a>)}
      </div>}
    </>}
  </section>
}

export function LiveRunCards({ onChoose }: { onChoose: (id: string) => void }) {
  const { data, error, isPending } = useLiveTrainingSummary()
  return <section aria-label="Live training runs" className="space-y-3">
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div>
        <h2 className="flex items-center gap-2 text-lg font-semibold"><Radio size={19} className="text-success" /> Live training</h2>
        <p className="mt-1 text-xs text-secondary">Fresh PPO streams from every mounted worktree. Select a run to inspect it immediately.</p>
      </div>
      <span className="text-xs font-medium text-muted">{data ? `${data.runs.length} streaming` : isPending ? 'Checking…' : 'Unavailable'}</span>
    </div>
    {error && <p role="alert" className="rounded-lg border border-warning/40 p-3 text-sm text-warning">Live sources could not be refreshed: {error.message}</p>}
    {data?.runs.length ? <div className="grid gap-3 lg:grid-cols-2">{data.runs.map(run =>
      <button key={run.id} type="button" onClick={() => onChoose(run.id)}
        className="control-focus group min-w-0 rounded-xl border border-success/30 bg-panel p-4 text-left transition-colors hover:border-success/75 hover:bg-hover sm:p-5">
        <div className="flex items-start justify-between gap-3">
          <LiveHeading run={run} />
          <ArrowRight size={18} className="shrink-0 text-secondary transition-transform group-hover:translate-x-1" />
        </div>
        <div className="mt-4 flex flex-wrap gap-4 text-xs text-secondary">
          <span>Iteration <strong className="text-foreground">{amount(run.iteration)}</strong></span>
          <span>Reward <strong className="text-foreground">{amount(run.mean_reward, 2)}</strong></span>
          <span>{run.curriculum_stage == null ? 'Curriculum unavailable' : `Curriculum stage ${run.curriculum_stage}`}</span>
          <span>{run.deadline_remaining_seconds == null ? 'No deadline metadata' : `Ends in ${duration(run.deadline_remaining_seconds)}`}</span>
        </div>
        <div className="mt-4 flex items-center gap-1 text-xs font-semibold text-success"><ShieldCheck size={14} /> Actual live PPO samples · Open analysis</div>
      </button>)}</div> :
      <p className="rounded-xl border border-dashed border-border p-5 text-sm text-muted">
        {isPending ? 'Discovering live streams…' : 'No source is currently emitting fresh PPO data. Saved checkpoint analysis remains available below.'}
      </p>}
  </section>
}
