import { Suspense, lazy, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowUpRight, Film, RefreshCw } from 'lucide-react'
import { api } from '../api'

const RerunWebViewer = lazy(() => import('./RerunWebViewer'))

/** Historical MuJoCo pose playback. No training or simulation runs in the browser. */
export function RerunReplayPanel() {
  const queryClient = useQueryClient()
  const [selectedName, setSelectedName] = useState<string>('')
  const [viewerOpen, setViewerOpen] = useState(false)
  const [renderer, setRenderer] = useState<'webgl' | 'webgpu'>('webgl')
  const [newlyExported, setNewlyExported] = useState<{ name: string; url: string } | null>(null)
  const { data, isLoading, error, refetch, isFetching } = useQuery({
    queryKey: ['rerun-replays'],
    queryFn: api.replays,
    staleTime: 15_000,
  })
  const choices = data?.replays ?? []
  const selected = choices.find((item) => item.name === selectedName) ?? choices[0]
  const loadedUrl = newlyExported && selected && newlyExported.name === selected.name ? newlyExported.url : selected?.rrd_url
  const exportMutation = useMutation({
    mutationFn: (source: string) => api.exportReplay(source),
    onSuccess: async (result) => {
      setNewlyExported({ name: result.name.replace(/\.rrd$/, ''), url: result.rrd_url })
      setViewerOpen(true)
      await queryClient.invalidateQueries({ queryKey: ['rerun-replays'] })
    },
  })

  return (
    <section aria-label="Spatial episode replay" className="min-w-0 overflow-hidden rounded-xl border border-border bg-panel">
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-border p-4 sm:p-5">
        <div className="min-w-0">
          <div className="text-xs font-bold uppercase tracking-[0.1em] text-muted">Spatial debugging</div>
          <h2 className="mt-1 text-xl font-semibold">3D episode replay</h2>
          <p className="mt-1 text-sm text-muted">
            Recorded MuJoCo motion, world targets, velocity vectors and synchronized reward terms.
          </p>
        </div>
        <button
          type="button"
          onClick={() => void refetch()}
          className="inline-flex items-center gap-2 rounded-lg border border-border px-3 py-2 text-sm hover:bg-raised"
          disabled={isFetching}
          aria-label="Refresh recordings"
        >
          <RefreshCw size={15} className={isFetching ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>
      <div className="space-y-3 p-4 sm:p-5">
        {error ? <p role="alert" className="text-sm text-danger">Cannot list recordings: {error.message}</p> : null}
        <div className="flex flex-wrap items-end gap-3">
          <label className="min-w-0 flex-1 text-xs font-semibold text-muted">
            Captured episode
            <select
              className="mt-2 block w-full min-w-0 rounded-lg border border-border bg-background p-2 text-sm text-foreground"
              value={selected?.name ?? ''}
              onChange={(event) => { setSelectedName(event.target.value); setNewlyExported(null); setViewerOpen(false); exportMutation.reset() }}
              disabled={!choices.length}
            >
              {!choices.length ? <option value="">No recordings</option> : null}
              {choices.map((item) => (
                <option key={item.name} value={item.name}>
                  {item.name} {item.rrd_url ? '· Rerun ready' : '· NPZ only'}
                </option>
              ))}
            </select>
          </label>
          {selected?.source && !loadedUrl ? (
            <button
              type="button"
              disabled={exportMutation.isPending}
              onClick={() => exportMutation.mutate(selected.source!)}
              className="rounded-lg border border-accent/40 bg-accent/10 px-4 py-2 text-sm font-semibold text-foreground hover:bg-accent/20 disabled:opacity-50"
            >
              {exportMutation.isPending ? 'Converting…' : 'Generate Rerun replay'}
            </button>
          ) : null}
          {loadedUrl ? (
            <label className="text-xs font-medium text-muted">
              Graphics
              <select value={renderer} onChange={(e) => setRenderer(e.target.value as 'webgl' | 'webgpu')}
                className="ml-2 rounded-lg border border-border bg-background p-2 text-sm text-foreground">
                <option value="webgl">WebGL (compatible)</option>
                <option value="webgpu">WebGPU (faster)</option>
              </select>
            </label>
          ) : null}
          {loadedUrl ? (
            <button type="button" onClick={() => setViewerOpen((opened) => !opened)}
              className="rounded-lg border border-accent/40 bg-accent/10 px-4 py-2 text-sm font-semibold hover:bg-accent/20">
              {viewerOpen ? 'Close 3D viewer' : 'Open 3D viewer'}
            </button>
          ) : null}
          {loadedUrl ? (
            <a className="inline-flex items-center gap-1 rounded-lg border border-border px-3 py-2 text-sm hover:bg-raised"
              href={loadedUrl} download>
              <ArrowUpRight size={15} /> Save .rrd
            </a>
          ) : null}
        </div>
        {exportMutation.error ? <p role="alert" className="text-sm text-danger">{exportMutation.error.message}</p> : null}
        {loadedUrl && viewerOpen ? (
          <div className="relative h-[420px] min-w-0 overflow-hidden rounded-lg border border-border bg-black lg:h-[620px]">
            <Suspense fallback={<div className="flex h-full items-center justify-center text-sm text-muted">Loading 3D viewer…</div>}>
              <RerunWebViewer key={loadedUrl + '-' + renderer} url={loadedUrl} backend={renderer} />
            </Suspense>
          </div>
        ) : (
          <div className="flex min-h-48 flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-border bg-raised/30 p-6 text-center">
            <Film className="text-muted" size={24} />
            <p className="text-sm font-medium">{isLoading ? 'Finding recordings…' : loadedUrl ? 'Rerun recording is ready.' : selected ? 'This recording needs conversion to Rerun.' : 'No motion captures found.'}</p>
            <p className="max-w-lg text-xs text-muted">
              Existing NPZ captures live under captures/. Generate a .rrd above, or select Open 3D viewer for an existing recording. The large WebAssembly renderer only downloads when opened.
              Conversion requires the optional Python rerun dependencies on the dashboard host.
            </p>
          </div>
        )}
        <p className="text-xs text-muted">
          Read-only kinematic replay, not a live MuJoCo simulation. Reward breakdowns appear for captures recorded with the updated MotionRecorder.
        </p>
      </div>
    </section>
  )
}
