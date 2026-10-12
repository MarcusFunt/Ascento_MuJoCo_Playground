import { lazy, Suspense, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link, useNavigate, useParams, useRouterState } from '@tanstack/react-router'
import { Activity, ArrowUpRight, Bug, ChartNoAxesCombined, Film, Play, SlidersHorizontal } from 'lucide-react'
import { api } from '../api'
import { CurriculumRail } from '../components/CurriculumRail'
import { LiveLog } from '../components/LiveLog'
import { LiveTrainingPanel } from '../components/LiveTrainingPanel'
import { PageHeader } from '../components/PageHeader'
import { StateBadge } from '../components/StateBadge'
import { SelectInput } from '../components/ui/input'
import { fmtNumber } from '../lib/utils'

// Load heavy graphics and specialty diagnostics only when their workspace is selected.
const TelemetryChart = lazy(() => import('../components/TelemetryChart')
  .then((module) => ({ default: module.TelemetryChart })))
const RerunReplayPanel = lazy(() => import('../components/RerunReplayPanel')
  .then((module) => ({ default: module.RerunReplayPanel })))
const BlenderRenderGallery = lazy(() => import('../components/BlenderRenderGallery')
  .then((module) => ({ default: module.BlenderRenderGallery })))
const ViewerCard = lazy(() => import('../components/ViewerCard')
  .then((module) => ({ default: module.ViewerCard })))

const WorkspaceInspector = lazy(() => import('../components/WorkspaceInspector')
  .then((module) => ({ default: module.WorkspaceInspector })))

type WorkspaceTab = 'monitor' | 'visualize' | 'metrics' | 'debug'
type MediaTab = 'replay' | 'renders' | 'checkpoint'

const TABS = [
  { id: 'monitor', label: 'Monitor', summary: 'Behavior and reward', icon: Activity },
  { id: 'visualize', label: 'Visualize', summary: 'Replays, videos and viewer', icon: Film },
  { id: 'metrics', label: 'Metrics', summary: 'Learning curves', icon: ChartNoAxesCombined },
  { id: 'debug', label: 'Debug', summary: 'Policy and logs', icon: Bug },
] as const

const CORE = [
  ['reward', 'Reward', 'Training reward is not a measure of physical task success.'],
  ['episode_length', 'Episode length', 'A decline can mean the robot is terminating earlier.'],
  ['ppo_loss', 'PPO surrogate', 'An optimization diagnostic, not a direct success metric.'],
  ['entropy', 'Entropy', 'Exploration and action distribution spread.'],
  ['kl', 'KL divergence', 'How far the policy moved in the last update.'],
  ['clip_fraction', 'Clip fraction', 'Fraction of PPO samples at the clipping boundary.'],
] as const

const ADVANCED = [
  ['controller_request_saturation_fraction', 'Controller request saturation'],
  ['physical_saturation_fraction', 'Actuator saturation'],
  ['effort_rms', 'Controller effort RMS'],
  ['recovery_success', 'Recovery success'],
  ['recovery_time_s', 'Recovery time'],
  ['max_recovery_hold_s', 'Recovery hold duration'],
  ['recovery_dwell', 'Recovery dwell'],
  ['post_landing_stability', 'Post-landing stability'],
] as const

function tabFromHash(value: string | undefined): WorkspaceTab {
  const id = (value || '').replace(/^#/, '')
  return TABS.some((tab) => tab.id === id) ? id as WorkspaceTab : 'monitor'
}

/** One selected run and four focused workspaces, rather than an unbounded analysis page. */
export function AnalyzePage() {
  const params = useParams({ strict: false }) as { runId?: string }
  const navigate = useNavigate()
  const hash = useRouterState({ select: (state) => state.location.hash })
  const tab = tabFromHash(hash)
  const [mediaTab, setMediaTab] = useState<MediaTab>('replay')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [showInspector, setShowInspector] = useState(false)
  const runs = useQuery({ queryKey: ['runs'], queryFn: api.runs, refetchInterval: 15_000, staleTime: 10_000 })
  const selectedId = params.runId
    || runs.data?.runs.find((run) => ['starting', 'running', 'stopping'].includes(run.state))?.id
    || runs.data?.runs[0]?.id || ''
  const selectedRun = runs.data?.runs.find((run) => run.id === selectedId)
  const needsCharts = tab === 'monitor' || tab === 'metrics'
  const telemetry = useQuery({
    queryKey: ['telemetry', selectedId],
    queryFn: () => api.telemetry(selectedId, 450),
    enabled: Boolean(selectedId && needsCharts),
    refetchInterval: needsCharts ? 30_000 : false,
  })
  const curriculum = useQuery({
    queryKey: ['curriculum', selectedId],
    queryFn: () => api.curriculum(selectedId),
    enabled: Boolean(selectedId && needsCharts),
    refetchInterval: needsCharts ? 10_000 : false,
  })
  const records = telemetry.data?.records || []
  const advanced = useMemo(() => ADVANCED.filter(([metric]) =>
    records.some((record) => Number.isFinite(Number(record.canonical_metrics?.[metric] ?? record.metrics?.[metric])))), [records])

  const selectRun = (id: string) => {
    if (id) void navigate({
      to: '/analyze/$runId',
      params: { runId: id },
      hash: tab,
      resetScroll: false,
    })
  }
  const chooseTab = (next: WorkspaceTab) => {
    void navigate({ hash: next, replace: true, resetScroll: false })
  }

  return <>
    <PageHeader
      eyebrow="Training workspace"
      title="Understand the run"
      description="Movement, recordings, learning curves and policy diagnostics, organized around one run."
      actions={<div className="min-w-0 w-full sm:w-[min(26rem,60vw)]">
        <label htmlFor="workspace-run-select" className="mb-1.5 block text-xs font-semibold text-muted">Selected training run</label>
        <SelectInput id="workspace-run-select" value={selectedId} onChange={(event) => selectRun(event.target.value)}>
          {!selectedId ? <option value="">No runs found</option> : null}
          {(runs.data?.runs || []).map((run) =>
            <option key={run.id} value={run.id}>{run.display_name}</option>)}
        </SelectInput>
      </div>}
    />

    {selectedId ? <>
      <section aria-label="Selected run summary" className="mb-5 flex flex-wrap items-center justify-between gap-x-5 gap-y-3 rounded-xl border border-border bg-panel px-4 py-3 sm:px-5">
        <div className="flex min-w-0 flex-1 flex-wrap items-center gap-x-4 gap-y-2">
          {selectedRun ? <StateBadge state={selectedRun.state} stale={selectedRun.stale} /> : null}
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold">{selectedRun?.display_name || 'Loading run…'}</p>
            <p className="truncate text-xs text-muted">{selectedRun?.task || selectedRun?.stage || 'Training run'}</p>
          </div>
          <span className="text-xs text-secondary">Iteration {fmtNumber(selectedRun?.iteration, 0)}</span>
          <span className="text-xs text-secondary">Reward {fmtNumber(selectedRun?.reward, 3)}</span>
        </div>
        <Link to="/runs/$runId" params={{ runId: selectedId }}
          className="control-focus inline-flex min-h-9 items-center gap-1.5 rounded-lg border border-border px-3 text-xs font-semibold hover:bg-hover">
          Run settings <ArrowUpRight size={14} />
        </Link>
      </section>

      <nav aria-label="Workspace sections" role="tablist"
        className="sticky top-[calc(4rem_+_env(safe-area-inset-top))] z-20 mb-5 flex gap-1 overflow-x-auto rounded-xl border border-border bg-background/95 p-1.5 shadow-sm backdrop-blur-xl sm:gap-2">
        {TABS.map((item) => {
          const Icon = item.icon
          const active = tab === item.id
          return <button key={item.id} type="button" role="tab" aria-selected={active}
            aria-controls="workspace-content" onClick={() => chooseTab(item.id)}
            className={'control-focus flex min-h-14 min-w-0 flex-1 flex-col items-center justify-center gap-1 rounded-lg px-1 text-[11px] font-semibold transition-colors sm:min-h-12 sm:flex-row sm:gap-2 sm:px-4 sm:text-sm ' +
              (active ? 'bg-foreground text-background shadow-sm' : 'text-muted hover:bg-raised hover:text-foreground')}>
            <Icon size={16} aria-hidden="true" className="shrink-0" />
            <span className="flex min-w-0 flex-col items-center sm:items-start">
              <span>{item.label}</span>
              <span className={'hidden text-[10px] font-normal lg:block ' + (active ? 'opacity-70' : 'text-muted')}>{item.summary}</span>
            </span>
          </button>
        })}
      </nav>

      <div id="workspace-content" role="tabpanel" className="space-y-5" aria-label={TABS.find((item) => item.id === tab)?.label}>
        {tab === 'monitor' ? <>
          <LiveTrainingPanel key={selectedId} runId={selectedId} />
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <h2 className="text-lg font-semibold">Does the reward match behavior?</h2>
              <p className="mt-1 text-sm text-muted">Keep physical motion visible above and inspect the reward trend beside episode survival.</p>
            </div>
            <button type="button" onClick={() => chooseTab('metrics')}
              className="control-focus rounded-lg border border-border-strong px-3 py-2 text-sm font-semibold hover:bg-hover">All learning metrics</button>
          </div>
          {telemetry.error ? <p role="alert" className="text-sm text-danger">{telemetry.error.message}</p> : null}
          <Suspense fallback={<div className="h-48 animate-pulse rounded-xl border border-border bg-panel" />}>
          <div className="grid gap-4 lg:grid-cols-2">
            <TelemetryChart records={records} metric="reward" title="Training reward" description="Compare with actual movement and collapse indicators; a rising reward is not proof of successful navigation." />
            <TelemetryChart records={records} metric="episode_length" title="Episode survival" description="Unexpected drops can reveal frequent falls and early resets." />
          </div>
          </Suspense>
          <CurriculumRail curriculum={curriculum.data?.curriculum} compact />
        </> : null}

        {tab === 'visualize' ? <>
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <h2 className="text-lg font-semibold">Visual evidence</h2>
              <p className="mt-1 text-sm text-muted">Choose a recorded episode, rendered animation or interactive checkpoint viewer in one place.</p>
            </div>
            <button type="button" onClick={() => chooseTab('metrics')}
              className="control-focus rounded-lg border border-border-strong px-3 py-2 text-sm font-semibold hover:bg-hover">
              Compare learning curves
            </button>
          </div>
          <div role="tablist" aria-label="Visualization source"
            className="flex flex-wrap gap-2 rounded-xl border border-border bg-panel p-2">
            {([
              { id: 'replay', name: 'Episode replay', icon: Play },
              { id: 'renders', name: 'Rendered animations', icon: Film },
              { id: 'checkpoint', name: 'Interactive checkpoint', icon: SlidersHorizontal },
            ] as const).map((item) => {
              const Icon = item.icon
              return <button key={item.id} type="button" role="tab"
                aria-selected={mediaTab === item.id} onClick={() => setMediaTab(item.id)}
                className={'control-focus flex min-h-11 items-center gap-2 rounded-lg px-3 text-xs font-semibold sm:px-4 sm:text-sm ' +
                  (mediaTab === item.id ? 'bg-raised text-foreground ring-1 ring-border-strong' : 'text-muted hover:bg-hover hover:text-foreground')}>
                <Icon size={16} /> {item.name}
              </button>
            })}
          </div>
          <Suspense fallback={<div className="h-52 animate-pulse rounded-xl border border-border bg-panel p-5 text-sm text-muted">Loading visualizations…</div>}>
            {mediaTab === 'replay' ? <RerunReplayPanel /> : null}
            {mediaTab === 'renders' ? <BlenderRenderGallery /> : null}
            {mediaTab === 'checkpoint' ? <ViewerCard runId={selectedId} /> : null}
          </Suspense>
          {mediaTab !== 'checkpoint' ? <p className="text-xs text-muted">The saved media library currently contains captures from multiple runs; its own source and checkpoint metadata identifies each recording. The run selection above controls live data and learning charts.</p> : null}
        </> : null}

        {tab === 'metrics' ? <>
          <div>
            <h2 className="text-lg font-semibold">Learning curves</h2>
            <p className="mt-1 text-sm text-muted">Reward, survival and PPO optimization signals for the selected run.</p>
          </div>
          <CurriculumRail curriculum={curriculum.data?.curriculum} compact />
          {telemetry.error ? <p role="alert" className="text-sm text-danger">{telemetry.error.message}</p> : null}
          <Suspense fallback={<div className="h-48 animate-pulse rounded-xl border border-border bg-panel" />}>
          <div className="grid gap-4 xl:grid-cols-2">
            {CORE.map(([metric, title, description]) =>
              <TelemetryChart key={metric} records={records} metric={metric} title={title} description={description} />)}
          </div>
          </Suspense>
          {advanced.length ? <section className="space-y-4">
            <button type="button" aria-expanded={showAdvanced} onClick={() => setShowAdvanced((value) => !value)}
              className="control-focus flex min-h-12 w-full items-center justify-between rounded-xl border border-border bg-panel px-4 text-left text-sm font-semibold hover:bg-raised">
              Task-specific diagnostics <span className="text-xs text-muted">{showAdvanced ? 'Hide' : 'Show'} {advanced.length} charts</span>
            </button>
            {showAdvanced ? <Suspense fallback={<div className="h-48 animate-pulse rounded-xl border border-border bg-panel" />}>
              <div className="grid gap-4 xl:grid-cols-2">
              {advanced.map(([metric, title]) =>
                <TelemetryChart key={metric} records={records} metric={metric} title={title} />)}
              </div>
            </Suspense> : null}
          </section> : null}
        </> : null}

        {tab === 'debug' ? <>
          <div>
            <h2 className="text-lg font-semibold">Policy diagnostics</h2>
            <p className="mt-1 text-sm text-muted">Detailed observations and action pipelines are advanced tools. Open them only when investigating a specific behavior.</p>
          </div>
          <button type="button" aria-expanded={showInspector} onClick={() => setShowInspector((value) => !value)}
            className="control-focus flex min-h-12 w-full items-center justify-between rounded-xl border border-border bg-panel px-4 text-left text-sm font-semibold hover:bg-raised">
            Observation and policy inspector <span className="text-xs text-muted">{showInspector ? 'Close tools' : 'Open tools'}</span>
          </button>
          {showInspector ? <Suspense fallback={<div className="h-48 animate-pulse rounded-xl border border-border bg-panel" />}>
            <WorkspaceInspector key={selectedId} runId={selectedId} />
          </Suspense> : null}
          <LiveLog runId={selectedId} />
        </> : null}
      </div>
    </> : <section className="rounded-xl border border-border bg-panel p-10 text-center text-muted">
      No training runs are available. Start a run from Overview, or browse the Runs page.
    </section>}
  </>
}
