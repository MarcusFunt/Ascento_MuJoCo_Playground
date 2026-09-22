import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, useParams } from '@tanstack/react-router'
import { ChartNoAxesCombined, Pencil, Square } from 'lucide-react'
import { api } from '../api'
import { CurriculumRail } from '../components/CurriculumRail'
import { EditRunDialog } from '../components/EditRunDialog'
import { PageHeader } from '../components/PageHeader'
import { PolicyArchitectureCard } from '../components/PolicyArchitectureCard'
import { StateBadge } from '../components/StateBadge'
import { ViewerCard } from '../components/ViewerCard'
import { Button } from '../components/ui/button'
import { Progress } from '../components/ui/progress'
import { fmtDate, fmtNumber, fmtPercent, shortCommit } from '../lib/utils'

export function RunDetailPage() {
  const { runId } = useParams({ strict: false }) as { runId: string }
  const [editOpen, setEditOpen] = useState(false)
  const queryClient = useQueryClient()
  const detail = useQuery({ queryKey: ['run', runId], queryFn: () => api.run(runId), refetchInterval: 10_000 })
  const runs = useQuery({ queryKey: ['runs'], queryFn: api.runs, staleTime: 10_000 })
  const curriculum = useQuery({ queryKey: ['curriculum', runId], queryFn: () => api.curriculum(runId), refetchInterval: 10_000 })
  const stop = useMutation({
    mutationFn: () => api.stopRun(runId),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['run', runId] }),
        queryClient.invalidateQueries({ queryKey: ['runs'] }),
        queryClient.invalidateQueries({ queryKey: ['overview'] }),
      ])
    },
  })

  if (detail.isLoading) return <div className="h-[500px] animate-pulse rounded-xl border border-border bg-panel" />
  if (detail.error || !detail.data) return <div className="rounded-xl border border-danger/40 bg-danger/10 p-5 text-danger">{detail.error?.message || 'Run not found.'}</div>

  const run = detail.data
  const info = run.run_info || {}
  const telemetry = run.telemetry || {}
  const active = ['starting', 'running', 'stopping'].includes(String(run.state || ''))
  const percent = Number(telemetry.percent_complete || 0)
  const latest = run.training_health?.latest || {}

  return (
    <>
      <PageHeader
        eyebrow={String(info.task || run.stage || 'Run')}
        title={run.display_name || run.name}
        description={run.notes || run.name}
        actions={
          <>
            <Link to="/analyze/$runId" params={{ runId }} className="control-focus inline-flex h-10 items-center gap-2 rounded-lg border border-border-strong bg-raised px-4 text-sm font-semibold hover:bg-hover"><ChartNoAxesCombined size={16} /> Analyze</Link>
            <Button onClick={() => setEditOpen(true)}><Pencil size={15} /> Edit</Button>
            {active ? <Button variant="danger" disabled={stop.isPending || run.state === 'stopping'} onClick={() => stop.mutate()}><Square size={14} /> {run.state === 'stopping' ? 'Stopping…' : 'Graceful stop'}</Button> : null}
          </>
        }
      />

      <section className="mb-6 rounded-xl border border-border bg-panel p-6">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <StateBadge state={run.state} stale={run.stale} />
          <div className="flex flex-wrap gap-x-6 gap-y-2 text-sm text-muted">
            <span>Iteration <strong className="numeric text-foreground">{fmtNumber(telemetry.iteration, 0)}</strong> / {fmtNumber(telemetry.total_iterations, 0)}</span>
            <span>Started <strong className="text-secondary">{fmtDate(info.started_at)}</strong></span>
            <span>Commit <strong className="font-mono text-secondary">{shortCommit(run.repository_version?.run_commit)}</strong></span>
          </div>
        </div>
        <div className="mt-5 flex items-center gap-4">
          <Progress value={percent} className="h-2 flex-1" />
          <strong className="numeric text-sm">{fmtPercent(percent, 1)}</strong>
        </div>
        <div className="mt-5 grid gap-px overflow-hidden rounded-lg border border-border bg-border sm:grid-cols-2 xl:grid-cols-4">
          {[
            ['Reward', fmtNumber(latest.reward, 4)],
            ['Episode length', fmtNumber(latest.episode_length, 1)],
            ['KL', fmtNumber(latest.kl, 5)],
            ['Entropy', fmtNumber(latest.entropy, 4)],
          ].map(([label, value]) => (
            <div key={label} className="bg-panel px-4 py-3">
              <span className="block text-[10px] font-bold uppercase tracking-[0.08em] text-muted">{label}</span>
              <strong className="numeric mt-1 block text-lg">{value}</strong>
            </div>
          ))}
        </div>
      </section>

      <div className="space-y-6">
        <CurriculumRail curriculum={curriculum.data?.curriculum} />
        <PolicyArchitectureCard
          runId={runId}
          checkpointPath={typeof info.checkpoint_path === 'string' ? info.checkpoint_path : undefined}
        />
        <ViewerCard
          runId={runId}
          checkpointPath={typeof info.checkpoint_path === 'string' ? info.checkpoint_path : undefined}
        />

        <section className="rounded-xl border border-border bg-panel p-6">
          <div className="text-xs font-bold uppercase tracking-[0.1em] text-muted">Provenance & contracts</div>
          <div className="mt-5 grid gap-px overflow-hidden rounded-lg border border-border bg-border sm:grid-cols-2 lg:grid-cols-4">
            {[
              ['Repository', run.repository_version?.status || 'unknown'],
              ['Task topology ABI', run.task_contract?.status || 'legacy'],
              ['Action-controller ABI', run.action_contract?.status || 'legacy'],
              ['Plant ABI', run.plant_contract?.status || 'legacy'],
              ['Device', String(info.device || '—')],
              ['Seed', String(info.seed ?? '—')],
              ['Simulation timestep', String(info.simulation_timestep ?? '—')],
              ['Checkpoint', String(info.checkpoint_path || '—')],
            ].map(([label, value]) => (
              <div key={label} className="min-w-0 bg-panel p-4">
                <span className="block text-xs text-muted">{label}</span>
                <strong className="mt-1 block truncate text-sm">{value}</strong>
              </div>
            ))}
          </div>
          {Array.isArray(info.command) || typeof info.command === 'string' ? (
            <details className="mt-5 rounded-lg border border-border bg-background/35 p-4">
              <summary className="cursor-pointer text-sm font-semibold">Training command</summary>
              <pre className="mt-4 max-h-64 overflow-auto whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-secondary">
                {Array.isArray(info.command) ? info.command.join('\n') : String(info.command)}
              </pre>
            </details>
          ) : null}
        </section>
      </div>

      {stop.error ? <div className="mt-6 rounded-lg border border-danger/40 bg-danger/10 p-4 text-sm text-danger">{stop.error.message}</div> : null}
      <EditRunDialog detail={run} runs={runs.data?.runs || []} open={editOpen} onOpenChange={setEditOpen} />
    </>
  )
}
