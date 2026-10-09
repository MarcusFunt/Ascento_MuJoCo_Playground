import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Download, Film, RefreshCw } from 'lucide-react'
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
    refetchInterval: 20_000,
  })
  const sourceItems = renders.data?.renders || []
  const taskOptions = useMemo(() => [...new Set(sourceItems.map((item) => item.inputs?.source_npz?.task).filter((value): value is string => Boolean(value)))], [sourceItems])
  const items = useMemo(() => sourceItems.filter((render) => {
    const capture = render.inputs?.source_npz
    const policy = render.inputs?.policy_checkpoint
    const effectiveStatus = render.status === 'complete' && !render.outputs?.video?.url && !render.outputs?.preview?.url ? 'incomplete' : String(render.status || 'unknown')
    const createdAt = render.created_at_utc ? new Date(render.created_at_utc).getTime() : 0
    return (!task || capture?.task === task)
      && (!status || effectiveStatus === status)
      && (!checkpointHash || String(policy?.sha256 || '').toLowerCase().includes(checkpointHash.toLowerCase()))
      && (!captureHash || String(capture?.sha256 || '').toLowerCase().includes(captureHash.toLowerCase()))
      && (!since || createdAt >= new Date(`${since}T00:00:00`).getTime())
  }), [sourceItems, task, status, checkpointHash, captureHash, since])

  return (
    <section className="rounded-xl border border-border bg-panel p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.1em] text-muted"><Film size={14} /> Blender pipeline</div>
          <h2 className="mt-1 text-xl font-semibold tracking-[-0.02em]">Motion renders</h2>
          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted">
            Render manifests and media served read-only from the WSL-mounted captures/blender directory.
          </p>
        </div>
        <Button variant="secondary" size="sm" onClick={() => void renders.refetch()} disabled={renders.isFetching}>
          <RefreshCw size={14} className={renders.isFetching ? 'animate-spin' : ''} /> Refresh
        </Button>
      </div>

      {detailed ? <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-5" aria-label="Filter captures and renders">
        <label className="text-xs font-bold uppercase tracking-[0.06em] text-muted">Task<SelectInput className="mt-2 w-full" value={task} onChange={(event) => setTask(event.target.value)}><option value="">All tasks</option>{taskOptions.map((item) => <option key={item}>{item}</option>)}</SelectInput></label>
        <label className="text-xs font-bold uppercase tracking-[0.06em] text-muted">Render status<SelectInput className="mt-2 w-full" value={status} onChange={(event) => setStatus(event.target.value)}><option value="">All statuses</option>{['complete', 'incomplete', 'rendering', 'failed'].map((item) => <option key={item}>{item}</option>)}</SelectInput></label>
        <label className="text-xs font-bold uppercase tracking-[0.06em] text-muted">Checkpoint SHA-256<Input className="mt-2" value={checkpointHash} onChange={(event) => setCheckpointHash(event.target.value)} placeholder="Search hash" /></label>
        <label className="text-xs font-bold uppercase tracking-[0.06em] text-muted">Source NPZ SHA-256<Input className="mt-2" value={captureHash} onChange={(event) => setCaptureHash(event.target.value)} placeholder="Search hash" /></label>
        <label className="text-xs font-bold uppercase tracking-[0.06em] text-muted">Created since<Input className="mt-2" type="date" value={since} onChange={(event) => setSince(event.target.value)} /></label>
      </div> : null}

      {renders.error ? (
        <p className="mt-5 rounded-lg border border-danger/35 bg-danger/10 p-3 text-sm text-danger" role="alert">
          Could not load Blender renders: {renders.error.message}
        </p>
      ) : null}
      {renders.isLoading ? <p className="mt-5 text-sm text-muted">Scanning mounted render manifests…</p> : null}
      {!renders.isLoading && !renders.error && items.length === 0 ? (
        <p className="mt-5 rounded-lg border border-border bg-background/35 p-4 text-sm text-muted" role="status">
          {sourceItems.length ? 'No render manifests match these filters.' : 'No render manifests are available yet. Use scripts/render_blender.ps1 to publish a render into the WSL captures mount.'}
        </p>
      ) : null}

      {items.length ? (
        <div className="mt-5 grid gap-4 xl:grid-cols-2">
          {items.map((render) => {
            const capture = render.inputs?.source_npz
            const policy = render.inputs?.policy_checkpoint
            const video = render.outputs?.video
            const preview = render.outputs?.preview
            const title = render.render_id || render.manifest_path || 'Blender render'
            const effectiveStatus = render.status === 'complete' && !video?.url && !preview?.url ? 'incomplete outputs' : render.status || 'unknown status'
            return (
              <article key={render.manifest_path || title} className="min-w-0 overflow-hidden rounded-lg border border-border bg-background/35">
                {video?.url ? (
                  <video className="aspect-video w-full bg-black object-contain" controls preload="metadata">
                    <source src={video.url} type="video/mp4" />
                  </video>
                ) : preview?.url ? (
                  <img className="aspect-video w-full bg-black object-contain" src={preview.url} alt={`${title} preview frame`} loading="lazy" />
                ) : <div className="flex aspect-video items-center justify-center bg-black px-6 text-center text-sm text-muted">No video or preview output is present for this render.</div>}
                <div className="p-4">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="min-w-0">
                      <h3 className="break-all text-sm font-semibold">{title}</h3>
                      <p className="mt-1 text-xs text-muted">{dateLabel(render.created_at_utc)} · {effectiveStatus}</p>
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
                  <div className="mt-4 grid gap-3 sm:grid-cols-2">
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
                  {render.error ? <p className="mt-3 text-xs text-danger">{render.error}</p> : null}
                </div>
              </article>
            )
          })}
        </div>
      ) : null}
    </section>
  )
}
