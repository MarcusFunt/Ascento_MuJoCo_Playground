import { useMemo, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { GitCompareArrows, Plus, Search } from 'lucide-react'
import { api } from '../api'
import { NewRunDialog } from '../components/NewRunDialog'
import { PageHeader } from '../components/PageHeader'
import { RunTable } from '../components/RunTable'
import { Button } from '../components/ui/button'
import { Input } from '../components/ui/input'
import { activeState, fmtNumber } from '../lib/utils'

export function RunsPage() {
  const [newRunOpen, setNewRunOpen] = useState(false)
  const [filter, setFilter] = useState('')
  const [activeOnly, setActiveOnly] = useState(false)
  const [compareIds, setCompareIds] = useState<string[]>([])
  const runs = useQuery({ queryKey: ['runs'], queryFn: api.runs, refetchInterval: 10_000, staleTime: 4_000 })
  const compare = useMutation({ mutationFn: () => api.compareRuns(compareIds) })

  const visible = useMemo(() => {
    const query = filter.trim().toLowerCase()
    return (runs.data?.runs || []).filter((run) => {
      if (activeOnly && !activeState(run.state)) return false
      if (!query) return true
      return [run.display_name, run.name, run.task, run.stage, run.state, ...(run.tags || [])]
        .filter(Boolean)
        .join(' ')
        .toLowerCase()
        .includes(query)
    })
  }, [activeOnly, filter, runs.data?.runs])

  const toggleCompare = (id: string) => {
    compare.reset()
    setCompareIds((current) => current.includes(id) ? current.filter((value) => value !== id) : current.length >= 8 ? current : [...current, id])
  }

  const comparisonRuns = Array.isArray((compare.data as any)?.runs) ? (compare.data as any).runs : []

  return (
    <>
      <PageHeader
        eyebrow="Experiments"
        title="Runs"
        description="A compact run library. Creation and metadata editing stay out of the way until you ask for them."
        actions={<Button variant="primary" onClick={() => setNewRunOpen(true)}><Plus size={16} /> Start training</Button>}
      />

      <section className="overflow-hidden rounded-xl border border-border bg-panel">
        <div className="flex flex-wrap items-end gap-4 border-b border-border p-5">
          <label className="min-w-[260px] flex-1">
            <span className="mb-1.5 block text-xs font-bold uppercase tracking-[0.06em] text-muted">Find a run</span>
            <div className="relative">
              <Search size={16} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
              <Input className="pl-9" value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="Name, task, stage or tag" />
            </div>
          </label>
          <label className="flex h-11 items-center gap-2 rounded-lg border border-border-strong bg-background px-3 text-sm text-secondary">
            <input type="checkbox" checked={activeOnly} onChange={(event) => setActiveOnly(event.target.checked)} />
            Active only
          </label>
          <div className="h-11 border-l border-border pl-4 text-right">
            <span className="block text-xs text-muted">Showing</span>
            <strong className="numeric text-sm">{visible.length} / {runs.data?.runs.length ?? 0}</strong>
          </div>
        </div>

        {runs.error ? <div className="m-5 rounded-lg border border-danger/40 bg-danger/10 p-4 text-sm text-danger">{runs.error.message}</div> : null}
        <RunTable runs={visible} compareIds={compareIds} onToggleCompare={toggleCompare} />

        <div className="flex flex-wrap items-center gap-3 border-t border-border p-4">
          <span className="mr-auto text-sm text-muted">{compareIds.length} selected for comparison</span>
          {compareIds.length ? <Button size="sm" variant="ghost" onClick={() => { setCompareIds([]); compare.reset() }}>Clear</Button> : null}
          <Button size="sm" disabled={compareIds.length < 2 || compare.isPending} onClick={() => compare.mutate()}>
            <GitCompareArrows size={15} /> Compare selected
          </Button>
        </div>
      </section>

      {compare.error ? <div className="mt-6 rounded-xl border border-danger/40 bg-danger/10 p-4 text-sm text-danger">{compare.error.message}</div> : null}

      {comparisonRuns.length ? (
        <section className="mt-6 rounded-xl border border-border bg-panel p-6">
          <div className="text-xs font-bold uppercase tracking-[0.1em] text-muted">Comparison</div>
          <h2 className="mt-1 text-xl font-semibold">Latest normalized telemetry</h2>
          <div className="mt-5 overflow-x-auto">
            <table className="w-full border-collapse text-left text-sm">
              <thead className="border-b border-border text-xs uppercase tracking-[0.06em] text-muted">
                <tr><th className="px-3 py-3">Run</th><th className="px-3 py-3">Reward</th><th className="px-3 py-3">Δ reward</th><th className="px-3 py-3">Episode length</th><th className="px-3 py-3">PPO loss</th><th className="px-3 py-3">KL</th><th className="px-3 py-3">Iteration</th></tr>
              </thead>
              <tbody className="divide-y divide-border">
                {comparisonRuns.map((run: any) => (
                  <tr key={run.id}>
                    <td className="px-3 py-4 font-semibold">{run.display_name}{run.id === (compare.data as any)?.baseline_id ? <span className="ml-2 text-xs text-muted">baseline</span> : null}</td>
                    <td className="numeric px-3 py-4">{fmtNumber(run.latest_metrics?.reward, 4)}</td>
                    <td className="numeric px-3 py-4">{fmtNumber(run.delta_from_baseline?.reward, 4)}</td>
                    <td className="numeric px-3 py-4">{fmtNumber(run.latest_metrics?.episode_length, 2)}</td>
                    <td className="numeric px-3 py-4">{fmtNumber(run.latest_metrics?.ppo_loss, 6)}</td>
                    <td className="numeric px-3 py-4">{fmtNumber(run.latest_metrics?.kl, 6)}</td>
                    <td className="numeric px-3 py-4">{fmtNumber(run.iteration, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      <NewRunDialog open={newRunOpen} onOpenChange={setNewRunOpen} />
    </>
  )
}
