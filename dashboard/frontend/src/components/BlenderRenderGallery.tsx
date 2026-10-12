import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Download, Film, RefreshCw } from 'lucide-react'
import { api } from '../api'
import { Button } from './ui/button'

function dateLabel(value?: string) {
  if (!value) return 'Time unavailable'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString()
}

export function BlenderRenderGallery() {
  const [search, setSearch] = useState('')
  const [showAll, setShowAll] = useState(false)
  const renders = useQuery({
    queryKey: ['blender-renders'],
    queryFn: api.blenderRenders,
    refetchInterval: 60_000,
    staleTime: 60_000,
  })
  const items = renders.data?.renders || []
  const matches = items.filter((render) => {
    const text = [render.render_id, render.manifest_path, render.inputs?.source_npz?.task,
      render.inputs?.policy_checkpoint?.path].filter(Boolean).join(' ').toLowerCase()
    return text.includes(search.trim().toLowerCase())
  })
  const visible = showAll ? matches : matches.slice(0, 4)

  return (
    <section className="rounded-xl border border-border bg-panel p-4 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.1em] text-muted"><Film size={14} /> Video library</div>
          <h2 className="mt-1 text-xl font-semibold tracking-[-0.02em]">Rendered animations</h2>
          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted">
            Watch recorded behavior here, then switch to replay or learning curves without leaving the workspace.
          </p>
        </div>
        <Button variant="secondary" size="sm" onClick={() => void renders.refetch()} disabled={renders.isFetching}>
          <RefreshCw size={14} className={renders.isFetching ? 'animate-spin' : ''} /> Refresh
        </Button>
      </div>

      {renders.error ? (
        <p className="mt-5 rounded-lg border border-danger/35 bg-danger/10 p-3 text-sm text-danger" role="alert">
          Could not load Blender renders: {renders.error.message}
        </p>
      ) : null}
      {renders.isLoading ? <p className="mt-5 text-sm text-muted">Scanning mounted render manifests…</p> : null}
      {!renders.isLoading && !renders.error && items.length === 0 ? (
        <p className="mt-5 rounded-lg border border-border bg-background/35 p-4 text-sm text-muted" role="status">
          No render manifests are available yet. Use scripts/render_blender.ps1 to publish a render into the WSL captures mount.
        </p>
      ) : null}

      {items.length > 0 ? <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
        <label className="min-w-0 flex-1 text-xs font-semibold text-muted">
          Filter recordings
          <input type="search" value={search} onChange={(event) => { setSearch(event.target.value); setShowAll(false) }}
            placeholder="Search task, render or checkpoint"
            className="control-focus mt-1.5 block min-h-10 w-full rounded-lg border border-border bg-background px-3 text-sm text-foreground sm:max-w-md" />
        </label>
        <span className="text-xs text-muted">{matches.length} of {items.length} recordings</span>
      </div> : null}
      {items.length > 0 && matches.length === 0 ? <p className="mt-4 text-sm text-muted">No matching recordings. Try a different search.</p> : null}
      {matches.length > 0 ? (
        <div className="mt-4 grid gap-4 xl:grid-cols-2">
          {visible.map((render) => {
            const capture = render.inputs?.source_npz
            const policy = render.inputs?.policy_checkpoint
            const video = render.outputs?.video
            const preview = render.outputs?.preview
            const title = render.render_id || render.manifest_path || 'Blender render'
            return (
              <article key={render.manifest_path || title} className="min-w-0 overflow-hidden rounded-lg border border-border bg-background/35">
                {video?.url ? (
                  <video className="aspect-video w-full bg-black object-contain" controls preload="none">
                    <source src={video.url} type="video/mp4" />
                  </video>
                ) : preview?.url ? (
                  <img className="aspect-video w-full bg-black object-contain" src={preview.url} alt={`${title} preview frame`} loading="lazy" />
                ) : null}
                <div className="p-4">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="min-w-0">
                      <h3 className="break-all text-sm font-semibold">{title}</h3>
                      <p className="mt-1 text-xs text-muted">{dateLabel(render.created_at_utc)} · {render.status || 'unknown status'}</p>
                      <p className="mt-1 text-xs text-secondary">
                        {capture?.task || 'Task unknown'}{capture?.seed ? ` · seed ${capture.seed}` : ''}
                      </p>
                    </div>
                    <div className="flex gap-2">
                      {render.outputs?.blend?.url ? (
                        <a className="control-focus inline-flex h-8 items-center gap-1.5 rounded-md border border-border-strong px-2.5 text-xs font-semibold hover:bg-hover" href={render.outputs.blend.url} download>
                          <Download size={13} /> Scene
                        </a>
                      ) : null}
                      {video?.url ? (
                        <a className="control-focus inline-flex h-8 items-center gap-1.5 rounded-md border border-border-strong px-2.5 text-xs font-semibold hover:bg-hover" href={video.url} download>
                          <Download size={13} /> MP4
                        </a>
                      ) : null}
                      {render.manifest_url ? (
                        <a className="control-focus inline-flex h-8 items-center gap-1.5 rounded-md border border-border-strong px-2.5 text-xs font-semibold hover:bg-hover" href={render.manifest_url} download>
                          <Download size={13} /> Manifest
                        </a>
                      ) : null}
                    </div>
                  </div>
                  <details className="mt-4 border-t border-border pt-3">
                    <summary className="control-focus cursor-pointer rounded text-xs font-semibold text-secondary hover:text-foreground">
                      Capture metadata and provenance
                    </summary>
                  <div className="mt-3 grid gap-3 sm:grid-cols-2">
                    <div className="min-w-0">
                      <span className="block text-[10px] font-bold uppercase tracking-[0.08em] text-muted">Source capture SHA-256</span>
                      <code className="mt-1 block break-all text-[11px] text-secondary">{capture?.sha256 || 'Missing'}</code>
                    </div>
                    <div className="min-w-0">
                      <span className="block text-[10px] font-bold uppercase tracking-[0.08em] text-muted">Policy checkpoint</span>
                      <code className="mt-1 block break-all text-[11px] text-secondary">
                        {policy?.kind === 'zero_policy' ? 'Zero policy (no checkpoint)' : policy?.path || policy?.kind || 'Missing'}
                      </code>
                    </div>
                    <div className="min-w-0 sm:col-span-2">
                      <span className="block text-[10px] font-bold uppercase tracking-[0.08em] text-muted">Checkpoint SHA-256</span>
                      <code className="mt-1 block break-all text-[11px] text-secondary">{policy?.sha256 || 'Not applicable / unavailable'}</code>
                    </div>
                  </div>
                  </details>
                  {render.error ? <p className="mt-3 text-xs text-danger">{render.error}</p> : null}
                </div>
              </article>
            )
          })}
        </div>
      ) : null}
      {!showAll && matches.length > visible.length ? (
        <button type="button" onClick={() => setShowAll(true)}
          className="control-focus mt-4 w-full rounded-lg border border-border-strong px-4 py-3 text-sm font-semibold hover:bg-hover">
          Show all {matches.length} recordings
        </button>
      ) : null}
      {showAll && matches.length > 4 ? (
        <button type="button" onClick={() => setShowAll(false)}
          className="control-focus mt-4 rounded-lg border border-border px-4 py-2 text-sm font-semibold hover:bg-hover">
          Show fewer
        </button>
      ) : null}
    </section>
  )
}
