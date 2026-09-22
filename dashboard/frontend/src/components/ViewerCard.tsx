import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Box, ExternalLink } from 'lucide-react'
import { api } from '../api'
import { fmtNumber } from '../lib/utils'
import { Button } from './ui/button'
import { SelectInput } from './ui/input'
import { StateBadge } from './StateBadge'

export function ViewerCard({ runId }: { runId: string }) {
  const queryClient = useQueryClient()
  const checkpoints = useQuery({
    queryKey: ['checkpoints', runId],
    queryFn: () => api.checkpoints(runId),
    refetchInterval: 15_000,
  })
  const viewers = useQuery({
    queryKey: ['viewers'],
    queryFn: api.viewers,
    refetchInterval: (query) => {
      const current = query.state.data?.viewers?.[0]
      return current && ['starting', 'running', 'stopping'].includes(current.state) ? 2_000 : 10_000
    },
  })
  const viewer = viewers.data?.viewers?.[0] || null
  const forSelected = viewer?.run_id === runId
  const viewerActive = Boolean(viewer && ['starting', 'running', 'stopping'].includes(viewer.state))
  const [checkpoint, setCheckpoint] = useState('latest')
  const [follow, setFollow] = useState(false)

  useEffect(() => {
    const items = checkpoints.data?.checkpoints || []
    setCheckpoint((current) => {
      if (current === 'latest' || items.some((item) => item.relative_path === current)) return current
      return 'latest'
    })
  }, [checkpoints.data, runId])

  const start = useMutation({
    mutationFn: () => api.startViewer({ run_id: runId, checkpoint: checkpoint || 'latest', follow }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['viewers'] }),
  })
  const stop = useMutation({
    mutationFn: (id: string) => api.stopViewer(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['viewers'] }),
  })

  const viewerUrl = useMemo(() => {
    if (!viewer?.port || viewer.state !== 'running') return ''
    const url = new URL(window.location.href)
    url.port = String(viewer.port)
    url.pathname = '/'
    url.search = ''
    url.hash = ''
    return url.toString()
  }, [viewer?.port, viewer?.state])

  return (
    <section className="rounded-xl border border-border bg-panel p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.1em] text-muted"><Box size={14} /> Policy viewer</div>
          <h2 className="mt-1 text-xl font-semibold tracking-[-0.02em]">Interactive MuJoCo / Viser</h2>
          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted">
            Run one checkpoint in an isolated one-environment viewer with live balance diagnostics and reward terms.
          </p>
        </div>
        {forSelected && viewer ? <StateBadge state={viewer.state} /> : null}
      </div>

      {!forSelected || !viewerActive ? (
        <div className="mt-6">
          {viewerActive && viewer ? (
            <div className="mb-4 rounded-lg border border-warning/35 bg-warning/10 p-4 text-sm text-warning">
              Another run owns the single viewer slot. Stop it before opening this run.
              <Button className="ml-3" size="sm" variant="danger" onClick={() => stop.mutate(viewer.id)} disabled={stop.isPending}>Stop active viewer</Button>
            </div>
          ) : null}
          <div className="grid gap-4 md:grid-cols-[1fr_auto] md:items-end">
            <label>
              <span className="mb-1.5 block text-xs font-bold uppercase tracking-[0.06em] text-muted">Checkpoint</span>
              <SelectInput value={checkpoint} onChange={(event) => setCheckpoint(event.target.value)} disabled={viewerActive || !checkpoints.data?.checkpoints?.length}>
                {!checkpoints.data?.checkpoints?.length ? <option value="latest">No stable checkpoints yet</option> : null}
                {checkpoints.data?.checkpoints?.length ? (
                  <option value="latest">
                    Latest stable{checkpoints.data.latest ? ` · ${checkpoints.data.latest}` : ''}
                  </option>
                ) : null}
                {(checkpoints.data?.checkpoints || []).map((item) => (
                  <option key={item.relative_path} value={item.relative_path}>
                    {item.relative_path}{item.iteration !== null && item.iteration !== undefined ? ` · iteration ${item.iteration}` : ''}
                  </option>
                ))}
              </SelectInput>
            </label>
            <Button variant="primary" size="lg" disabled={viewerActive || !checkpoints.data?.checkpoints?.length || start.isPending} onClick={() => start.mutate()}>
              {start.isPending ? 'Starting…' : 'Visualize checkpoint'}
            </Button>
          </div>
          <label className="mt-4 flex items-center gap-2 text-sm text-secondary">
            <input type="checkbox" checked={follow} onChange={(event) => setFollow(event.target.checked)} disabled={viewerActive} />
            Follow newly completed checkpoints automatically
          </label>
        </div>
      ) : viewer ? (
        <div className="mt-6">
          <div className="grid gap-px overflow-hidden rounded-lg border border-border bg-border sm:grid-cols-3 lg:grid-cols-6">
            {[
              ['Loaded', viewer.checkpoint || '—'],
              ['Policy iteration', fmtNumber(viewer.checkpoint_iteration, 0)],
              ['Training iteration', fmtNumber(viewer.training_iteration, 0)],
              ['Lag', viewer.lag_iterations === null || viewer.lag_iterations === undefined ? '—' : `${fmtNumber(viewer.lag_iterations, 0)} it`],
              ['Mode', viewer.follow ? 'Follow latest' : 'Fixed'],
              ['Port', viewer.port || '—'],
            ].map(([label, value]) => (
              <div key={String(label)} className="bg-panel p-3">
                <span className="block text-[11px] uppercase tracking-[0.06em] text-muted">{label}</span>
                <strong className="numeric mt-1 block truncate text-sm">{String(value)}</strong>
              </div>
            ))}
          </div>
          <div className="mt-4 flex flex-wrap gap-3">
            {viewerUrl ? (
              <a href={viewerUrl} target="_blank" rel="noreferrer" className="control-focus inline-flex h-10 items-center gap-2 rounded-lg border border-foreground bg-foreground px-4 text-sm font-semibold text-background hover:bg-[#dfe2e5]">
                Open 3D viewer <ExternalLink size={15} />
              </a>
            ) : <Button variant="secondary" disabled>Viewer starting…</Button>}
            {viewerActive ? <Button variant="danger" onClick={() => stop.mutate(viewer.id)} disabled={stop.isPending}>Stop viewer</Button> : null}
          </div>
          {viewer.state === 'failed' ? (
            <p className="mt-3 text-sm text-danger">Viewer exited with code {viewer.exit_code ?? 'unknown'}. Worker logs remain available through the viewer API.</p>
          ) : null}
        </div>
      ) : null}

      {start.error ? <p className="mt-4 text-sm text-danger">{start.error.message}</p> : null}
    </section>
  )
}
