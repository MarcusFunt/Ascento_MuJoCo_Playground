import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate, useParams } from '@tanstack/react-router'
import { ArrowRight, ExternalLink } from 'lucide-react'
import { api } from '../api'
import { PolicyDebugger } from '../components/PolicyDebugger'
import { PageHeader } from '../components/PageHeader'
import { ViewerCard } from '../components/ViewerCard'
import { WaypointConsole } from '../components/WaypointConsole'
import { SelectInput } from '../components/ui/input'
import { usePolicyIntrospection } from '../hooks/usePolicyIntrospection'

export function VisualizerPage() {
  const params = useParams({ strict: false }) as { runId?: string }
  const navigate = useNavigate()
  const runs = useQuery({ queryKey: ['runs'], queryFn: api.runs, refetchInterval: 15_000 })
  const selectedId = params.runId || runs.data?.runs.find((run) => ['running', 'starting', 'stopping'].includes(run.state))?.id || runs.data?.runs[0]?.id || ''
  const run = runs.data?.runs.find((item) => item.id === selectedId)
  const detail = useQuery({ queryKey: ['run', selectedId], queryFn: () => api.run(selectedId), enabled: Boolean(selectedId) })
  const viewers = useQuery({ queryKey: ['viewers'], queryFn: api.viewers, refetchInterval: 2_000 })
  const viewer = viewers.data?.viewers.find((item) => item.run_id === selectedId)
  const introspection = usePolicyIntrospection(selectedId)
  const [browserLoaded, setBrowserLoaded] = useState(false)

  const chooseRun = (runId: string) => {
    setBrowserLoaded(false)
    if (runId) void navigate({ to: '/visualizer/$runId', params: { runId } })
  }

  const viewerUrl = (() => {
    if (!viewer?.port || viewer.state !== 'running') return ''
    const url = new URL(window.location.href)
    url.port = String(viewer.port)
    url.pathname = '/'
    url.search = ''
    url.hash = ''
    return url.toString()
  })()

  return (
    <>
      <PageHeader
        eyebrow="Interactive policy review"
        title="Visualizer"
        description="Inspect a stable checkpoint in the managed Viser viewer, command a bounded world-frame route and read policy telemetry from the same viewer session."
        actions={<div className="min-w-[260px]"><SelectInput aria-label="Select run for visualization" value={selectedId} onChange={(event) => chooseRun(event.target.value)}><option value="">Choose a run</option>{(runs.data?.runs || []).map((item) => <option key={item.id} value={item.id}>{item.display_name} · {item.task || item.stage || 'unknown task'}</option>)}</SelectInput></div>}
      />

      {runs.error ? <div className="mb-5 rounded-xl border border-danger/40 bg-danger/10 p-4 text-sm text-danger">Could not load runs: {runs.error.message}</div> : null}
      {!selectedId ? <div className="rounded-xl border border-border bg-panel p-8 text-center text-sm text-muted">No runs are available yet. Start or import a run before opening a policy viewer.</div> : (
        <div className="space-y-6">
          <div className="grid gap-5 xl:grid-cols-[minmax(0,1.5fr)_minmax(22rem,0.9fr)]">
            <section className="overflow-hidden rounded-xl border border-border bg-panel">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-5 py-4">
                <div><div className="text-xs font-bold uppercase tracking-[0.1em] text-muted">Managed Viser session</div><h2 className="mt-1 font-semibold">3D policy view</h2></div>
                <span className={`rounded border px-2 py-1 text-xs font-bold uppercase ${viewer?.state === 'running' ? 'border-success/40 bg-success/10 text-success' : 'border-border-strong bg-raised text-secondary'}`}>{viewer?.state || 'not started'}</span>
              </div>
              {viewerUrl ? (
                <div className="relative bg-black">
                  <iframe title="Interactive Ascento MuJoCo Viser viewer" src={viewerUrl} onLoad={() => setBrowserLoaded(true)} className="aspect-video w-full border-0" allow="fullscreen" />
                  <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border bg-panel px-4 py-3 text-xs">
                    <span className="text-secondary">Viewer process running · policy telemetry {introspection.connected ? 'connected' : 'connecting'}{browserLoaded ? ' · 3D panel loaded' : ''}</span>
                    <a href={viewerUrl} target="_blank" rel="noreferrer" className="control-focus inline-flex min-h-8 items-center gap-1.5 rounded border border-border-strong px-2.5 font-semibold hover:bg-hover">Open separately <ExternalLink size={13} /></a>
                  </div>
                </div>
              ) : (
                <div className="flex aspect-video items-center justify-center bg-[#080d12] p-8 text-center text-sm text-muted">{viewer?.state === 'starting' ? 'Viewer is starting. The 3D panel will appear when its port is ready.' : viewer?.state === 'failed' ? `Viewer failed${viewer.exit_code === null || viewer.exit_code === undefined ? '' : ` with exit code ${viewer.exit_code}`}. Check its run detail and logs.` : 'Start the selected checkpoint viewer to open the live 3D panel.'}</div>
              )}
            </section>
            <ViewerCard runId={selectedId} checkpointPath={typeof detail.data?.run_info?.checkpoint_path === 'string' ? String(detail.data.run_info.checkpoint_path) : undefined} />
          </div>

          {viewer ? <WaypointConsole viewerId={viewer.id} viewerState={viewer.state} task={viewer.task || run?.task || undefined} /> : <section className="rounded-xl border border-border bg-panel p-5 text-sm text-muted">Select a stable checkpoint above to start the managed viewer. Viewer controls remain isolated from the trainer.</section>}

          <section className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-border bg-panel px-5 py-4">
            <div><div className="text-xs font-bold uppercase tracking-[0.1em] text-muted">Selected run</div><h2 className="mt-1 font-semibold">{run?.display_name || detail.data?.display_name || 'Loading run…'}</h2><p className="mt-1 text-sm text-muted">{String(run?.task || detail.data?.run_info?.task || 'Task unavailable')} · checkpoint {viewer?.checkpoint || 'not loaded'}</p></div>
            <div className="flex gap-2"><a href={`/runs/${encodeURIComponent(selectedId)}`} className="control-focus inline-flex h-9 items-center gap-1.5 rounded-lg border border-border-strong px-3 text-xs font-semibold">Run detail <ArrowRight size={13} /></a><a href={`/analyze/${encodeURIComponent(selectedId)}`} className="control-focus inline-flex h-9 items-center gap-1.5 rounded-lg border border-border-strong px-3 text-xs font-semibold">Training analysis <ArrowRight size={13} /></a></div>
          </section>

          <PolicyDebugger
            viewer={introspection.viewer}
            schema={introspection.displaySchema}
            frame={introspection.displayFrame}
            liveFrame={introspection.frame}
            connected={introspection.connected}
            schemaError={introspection.schemaError as Error | null}
            captures={introspection.captures}
            capture={introspection.capture}
            selectedCaptureId={introspection.selectedCaptureId}
            selectedFrameIndex={introspection.selectedFrameIndex}
            selectCapture={introspection.selectCapture}
            selectFrame={introspection.selectFrame}
            onManualCapture={introspection.manualCapture}
            manualCapturePending={introspection.manualCapturePending}
            manualCaptureError={introspection.manualCaptureError as Error | null}
            captureLoading={introspection.captureLoading}
          />
        </div>
      )}
    </>
  )
}
