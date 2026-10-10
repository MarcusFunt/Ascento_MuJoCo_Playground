import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { AlertTriangle, ChevronDown, Clapperboard, Download, Film, RefreshCw, RotateCcw } from 'lucide-react'
import { api } from '../api'
import { Button } from './ui/button'
import { Input, SelectInput } from './ui/input'

function dateLabel(value?: string) {
  if (!value) return 'Time unavailable'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString()
}

export function BlenderRenderGallery({ detailed = false }: { detailed?: boolean }) {
  const [task, setTask] = useState('')
  const [status, setStatus] = useState('')
  const [checkpointHash, setCheckpointHash] = useState('')
  const [captureHash, setCaptureHash] = useState('')
  const [since, setSince] = useState('')
  const renders = useQuery({
    queryKey: ['blender-renders'],
    queryFn: () => api.blenderRenders(500),
    refetchInterval: 60_000,
  })
  const sourceItems = renders.data?.renders || []
  const taskOptions = useMemo(
    () => [...new Set(sourceItems.map((item) => item.inputs?.source_npz?.task).filter((value): value is string => Boolean(value)))],
    [sourceItems],
  )
  const filtered = useMemo(() => sourceItems.filter((render) => {
    const capture = render.inputs?.source_npz
    const policy = render.inputs?.policy_checkpoint
    const hasMedia = Boolean(render.outputs?.video?.url || render.outputs?.preview?.url)
    const effectiveStatus = render.status === 'complete' && !hasMedia ? 'incomplete' : String(render.status || 'unknown')
    const createdAt = render.created_at_utc ? new Date(render.created_at_utc).getTime() : 0
    return (!task || capture?.task === task)
      && (!status || effectiveStatus === status)
      && (!checkpointHash || String(policy?.sha256 || '').toLowerCase().includes(checkpointHash.toLowerCase()))
      && (!captureHash || String(capture?.sha256 || '').toLowerCase().includes(captureHash.toLowerCase()))
      && (!since || createdAt >= new Date(`${since}T00:00:00`).getTime())
  }), [sourceItems, task, status, checkpointHash, captureHash, since])

  const mediaItems = filtered.filter((render) => Boolean(render.outputs?.video?.url || render.outputs?.preview?.url))
  const otherItems = filtered.filter((render) => !render.outputs?.video?.url && !render.outputs?.preview?.url)
  const filtersActive = Boolean(task || status || checkpointHash || captureHash || since)
  const clearFilters = () => {
    setTask('')
    setStatus('')
    setCheckpointHash('')
    setCaptureHash('')
    setSince('')
  }

  return (
    <section className="min-w-0 rounded-xl border border-border bg-panel p-4 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.1em] text-muted"><Film size={14} /> Blender pipeline</div>
          <h2 className="mt-1 text-xl font-semibold tracking-[-0.02em]">Motion renders</h2>
          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-secondary">Browse reproducible motion media with its source and checkpoint identity.</p>
        </div>
        <Button variant="secondary" size="sm" onClick={() => void renders.refetch()} disabled={renders.isFetching}>
          <RefreshCw size={14} className={renders.isFetching ? 'animate-spin' : ''} /> Refresh
        </Button>
      </div>

      {detailed ? (
        <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-5" aria-label="Filter captures and renders">
          <label className="min-w-0 text-xs font-bold uppercase tracking-[0.06em] text-secondary">Task<SelectInput className="mt-2" value={task} onChange={(event) => setTask(event.target.value)}><option value="">All tasks</option>{taskOptions.map((item) => <option key={item}>{item}</option>)}</SelectInput></label>
          <label className="min-w-0 text-xs font-bold uppercase tracking-[0.06em] text-secondary">Status<SelectInput className="mt-2" value={status} onChange={(event) => setStatus(event.target.value)}><option value="">All statuses</option>{['complete', 'incomplete', 'rendering', 'failed'].map((item) => <option key={item}>{item}</option>)}</SelectInput></label>
          <label className="min-w-0 text-xs font-bold uppercase tracking-[0.06em] text-secondary">Checkpoint SHA<Input className="mt-2" value={checkpointHash} onChange={(event) => setCheckpointHash(event.target.value)} placeholder="Search checkpoint hash" /></label>
          <label className="min-w-0 text-xs font-bold uppercase tracking-[0.06em] text-secondary">Source SHA<Input className="mt-2" value={captureHash} onChange={(event) => setCaptureHash(event.target.value)} placeholder="Search source hash" /></label>
          <label className="min-w-0 text-xs font-bold uppercase tracking-[0.06em] text-secondary">Created since<Input className="mt-2" type="date" value={since} onChange={(event) => setSince(event.target.value)} /></label>
        </div>
      ) : null}

      {!renders.isLoading && !renders.error ? (
        <div className="mt-4 flex flex-wrap items-center justify-between gap-2 text-sm text-secondary">
          <span>{filtered.length} results · {mediaItems.length} with media · {otherItems.length} without preview</span>
          {filtersActive ? <Button variant="ghost" size="sm" onClick={clearFilters}><RotateCcw size={14} /> Clear filters</Button> : null}
        </div>
      ) : null}

      {renders.error ? <p className="mt-5 rounded-lg border border-danger/35 bg-danger/10 p-3 text-sm text-danger" role="alert">Could not load Blender renders: {renders.error.message} <button className="ml-2 underline" onClick={() => void renders.refetch()}>Retry</button></p> : null}
      {renders.isLoading ? <p className="mt-5 text-sm text-secondary">Loading saved render manifests…</p> : null}
      {!renders.isLoading && !renders.error && filtered.length === 0 ? (
        <p className="mt-5 rounded-lg border border-border bg-background/35 p-5 text-sm text-secondary" role="status">{sourceItems.length ? 'No renders match these filters.' : 'No render manifests are available yet.'}</p>
      ) : null}

      {mediaItems.length ? (
        <div className="mt-5 grid gap-4 lg:grid-cols-2" aria-label="Available motion renders">
          {mediaItems.map((render) => {
            const capture = render.inputs?.source_npz
            const policy = render.inputs?.policy_checkpoint
            const video = render.outputs?.video
            const preview = render.outputs?.preview
            const title = render.render_id || render.manifest_path || 'Blender render'
            return (
              <article key={render.manifest_path || title} className="min-w-0 overflow-hidden rounded-lg border border-border bg-background/35">
                {video?.url ? <video className="aspect-video w-full bg-black object-contain" controls preload="metadata" poster={preview?.url || undefined}><source src={video.url} type="video/mp4" /></video> :
                  preview?.url ? <img className="aspect-video w-full bg-black object-contain" src={preview.url} alt={`${title} preview frame`} loading="lazy" /> : null}
                <div className="p-4">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="min-w-0 flex-1">
                      <h3 className="line-clamp-2 break-words text-sm font-semibold">{title}</h3>
                      <p className="mt-1 text-xs text-secondary">{dateLabel(render.created_at_utc)} · {render.status || 'unknown'}</p>
                      <p className="mt-1 break-words text-xs text-muted">{capture?.task || 'Task unknown'}{capture?.seed != null ? ` · seed ${capture.seed}` : ''}</p>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {video?.url ? <DownloadLink href={video.url} label="MP4" /> : null}
                      {render.outputs?.blend?.url ? <DownloadLink href={render.outputs.blend.url} label="Scene" /> : null}
                      {render.manifest_url ? <DownloadLink href={render.manifest_url} label="Manifest" /> : null}
                    </div>
                  </div>
                  <RenderProvenance capture={capture} policy={policy} />
                </div>
              </article>
            )
          })}
        </div>
      ) : null}

      {otherItems.length ? (
        <section className="mt-6" aria-label="Other render records">
          <div className="mb-3 flex items-center gap-2"><AlertTriangle size={16} className="text-warning" /><h3 className="text-sm font-semibold">Incomplete or unavailable media</h3><span className="text-sm text-muted">({otherItems.length})</span></div>
          <div className="space-y-2">
            {otherItems.map((render) => {
              const capture = render.inputs?.source_npz
              const policy = render.inputs?.policy_checkpoint
              const title = render.render_id || render.manifest_path || 'Blender render'
              const effectiveStatus = render.status === 'complete' ? 'Incomplete output' : render.status || 'Unknown'
              return (
                <article key={render.manifest_path || title} className="min-w-0 rounded-lg border border-border bg-background/35 px-4 py-3">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2"><Clapperboard size={16} className="shrink-0 text-muted" /><h4 className="min-w-0 break-words text-sm font-semibold">{title}</h4></div>
                      <p className="mt-1 text-xs text-secondary">{dateLabel(render.created_at_utc)} · <span className="text-warning">{effectiveStatus}</span></p>
                      <p className="mt-1 break-words text-xs text-muted">{render.error || 'No playable video or preview was included in this render manifest.'}</p>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {render.outputs?.blend?.url ? <DownloadLink href={render.outputs.blend.url} label="Scene" /> : null}
                      {render.manifest_url ? <DownloadLink href={render.manifest_url} label="Manifest" /> : null}
                    </div>
                  </div>
                  <RenderProvenance capture={capture} policy={policy} />
                </article>
              )
            })}
          </div>
        </section>
      ) : null}
    </section>
  )
}

type Capture = { sha256?: string | null; task?: string | null; seed?: number | string | null }
type Policy = { kind?: string | null; path?: string | null; sha256?: string | null }

function RenderProvenance({ capture, policy }: { capture?: Capture; policy?: Policy }) {
  return (
    <details className="mt-3 border-t border-border pt-3 text-xs">
      <summary className="control-focus flex cursor-pointer items-center gap-2 rounded text-secondary hover:text-foreground"><ChevronDown size={14} aria-hidden /> Provenance and source files</summary>
      <dl className="mt-3 grid gap-3 sm:grid-cols-2">
        <div className="min-w-0"><dt className="text-muted">Source NPZ SHA-256</dt><dd className="mt-1 break-all font-mono text-secondary">{capture?.sha256 || 'Unavailable'}</dd></div>
        <div className="min-w-0"><dt className="text-muted">Checkpoint</dt><dd className="mt-1 break-all font-mono text-secondary">{policy?.kind === 'zero_policy' ? 'Zero policy' : policy?.path || policy?.kind || 'Unavailable'}</dd></div>
        <div className="min-w-0 sm:col-span-2"><dt className="text-muted">Checkpoint SHA-256</dt><dd className="mt-1 break-all font-mono text-secondary">{policy?.sha256 || 'Not applicable / unavailable'}</dd></div>
      </dl>
    </details>
  )
}

function DownloadLink({ href, label }: { href: string; label: string }) {
  return <a className="control-focus inline-flex min-h-8 items-center gap-1.5 rounded-md border border-border-strong px-2.5 text-xs font-semibold text-secondary hover:bg-hover hover:text-foreground" href={href} download><Download size={13} /> {label}</a>
}
