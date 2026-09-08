import { useEffect, useMemo, useRef, useState } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

const POLL_MS = 15000
// Six SVG charts redraw together.  This cap keeps interactions responsive
// while even sampling retains the full oldest-to-newest history span.
const CHART_RENDER_POINTS = 400
const LOG_TAIL_LIMIT = 400
const LOG_DISPLAY_LIMIT = 600
const LOG_FLUSH_MS = 500
const CANONICAL_CHARTS = [
  {
    key: 'reward',
    label: 'Reward',
    description: 'The task score the policy is optimizing. Read the trend over many iterations, not individual spikes.',
    interpretation: 'Up is usually good. A sustained fall means the policy is performing worse under the current reward definition.',
  },
  {
    key: 'episode_length',
    label: 'Episode length (env steps)',
    description: 'Mean episode duration reported in simulator steps, not seconds. Horizon curriculum stages create visible ceilings.',
    interpretation: 'Plateaus near 2,000/6,000/12,000/30,000 steps correspond to 20/60/120/300-second horizons at 100 Hz. A drop below the active ceiling indicates earlier failures or a transition window.',
  },
  {
    key: 'ppo_loss',
    label: 'PPO / surrogate loss',
    description: 'The policy optimization objective used by PPO. Its absolute value is not a score and should not be minimized visually.',
    interpretation: 'Neither direction is automatically better. Smooth, bounded movement is normal; large persistent jumps can indicate unstable updates.',
  },
  {
    key: 'entropy',
    label: 'Entropy',
    description: 'How random or exploratory the policy is. Higher entropy means actions remain less deterministic.',
    interpretation: 'A gradual decline is common as the policy settles. A sudden collapse can mean exploration disappeared too early; sustained growth can mean the policy is not settling.',
  },
  {
    key: 'kl',
    label: 'KL divergence',
    description: 'How far the updated policy moved from the previous policy during PPO optimization.',
    interpretation: 'Up means larger policy changes. Repeated spikes can signal overly aggressive updates; values near zero mean updates are very small.',
  },
  {
    key: 'clip_fraction',
    label: 'Clip fraction',
    description: 'The fraction of PPO samples whose policy-ratio update hit the clipping boundary.',
    interpretation: 'Up means more updates are being clipped. Persistently high values can mean policy steps are too aggressive; very low values mean updates are more conservative.',
  },
]

const ADVANCED_CHARTS = [
  { key: 'effort_rms', label: 'Command effort RMS (Nm)', description: 'RMS of normalized policy commands mapped to the 65 Nm simulation authority.', interpretation: 'A command-demand diagnostic; it is not a measured actuator-force metric.' },
  { key: 'effort_mean_abs', label: 'Mean command effort (Nm)', description: 'Mean absolute normalized policy command mapped to Nm.', interpretation: 'Useful for distinguishing sustained command demand from brief spikes.' },
  { key: 'physical_saturation_fraction', label: 'Command saturation', description: 'Fraction of policy commands at the configured effort limit.', interpretation: 'Diagnostic only for this simulation; high values indicate the policy is using the available authority.' },
  { key: 'recovery_success', label: 'Recovery success', description: 'Strict binary recovery-success metric.', interpretation: 'This must improve independently of dense shaping rewards.' },
  { key: 'recovery_time_s', label: 'Recovery time', description: 'Time to satisfy the strict recovery condition.', interpretation: 'Lower is better; inspect together with hold duration.' },
  { key: 'max_recovery_hold_s', label: 'Recovery hold duration', description: 'Longest continuous interval satisfying recovery conditions.', interpretation: 'Shows whether recovery is stable or only a transient pose.' },
  { key: 'recovery_dwell', label: 'Recovery dwell shaping', description: 'Dense recovery proximity/dwell shaping term.', interpretation: 'A shaping trend must not be mistaken for binary task success.' },
  { key: 'post_landing_stability', label: 'Post-landing stability', description: 'Dense post-landing stabilization signal.', interpretation: 'Use with jump/recovery success to detect reward hacking.' },
]

function fmtNumber(value, digits = 1) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—'
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits })
}

function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined || !Number.isFinite(Number(seconds))) return '—'
  const total = Math.max(0, Math.round(Number(seconds)))
  const days = Math.floor(total / 86400)
  const hours = Math.floor((total % 86400) / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  const secs = total % 60
  if (days) return `${days}d ${hours}h`
  if (hours) return `${hours}h ${minutes}m`
  if (minutes) return `${minutes}m ${secs}s`
  return `${secs}s`
}

function fmtDate(value) {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return String(value)
  return date.toLocaleString()
}

function shortCommit(value) {
  if (!value) return 'unknown'
  return String(value).slice(0, 10)
}

function sampleRecords(records, maxPoints) {
  if (records.length <= maxPoints) return records
  const lastIndex = records.length - 1
  return Array.from({ length: maxPoints }, (_, index) => (
    records[Math.round(index * lastIndex / (maxPoints - 1))]
  ))
}

function StateBadge({ state }) {
  return <span className={`state-badge state-${state || 'unknown'}`}>{state || 'unknown'}</span>
}

function ApiBadge({ health }) {
  if (!health) return <span className="api-badge api-pending">API checking</span>
  const label = health.ok ? 'API healthy' : 'API error'
  return <span className={`api-badge ${health.ok ? 'api-ok' : 'api-bad'}`}>{label}</span>
}

function MetricChart({ records, metric, label, description, interpretation }) {
  const latestRecord = [...records].reverse().find((record) => Number.isFinite(Number(record[metric])))
  const latestValue = latestRecord?.[metric]

  return (
    <section className="panel metric-panel">
      <div className="metric-header">
        <div>
          <div className="panel-title metric-title">{label}</div>
          <div className="metric-copy">{description}</div>
        </div>
        <div className="metric-latest">
          <span>Latest</span>
          <strong>{fmtNumber(latestValue, 5)}</strong>
        </div>
      </div>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={records} margin={{ top: 8, right: 12, left: -4, bottom: 2 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="iteration"
              type="number"
              scale="linear"
              domain={['dataMin', 'dataMax']}
              allowDataOverflow={false}
              tickFormatter={(value) => fmtNumber(value, 0)}
              minTickGap={36}
              tickCount={6}
            />
            <YAxis width={62} tickFormatter={(value) => fmtNumber(value, 3)} domain={['auto', 'auto']} />
            <Tooltip
              formatter={(value) => fmtNumber(value, 6)}
              labelFormatter={(iteration) => `Iteration ${fmtNumber(iteration, 0)}`}
            />
            <Line
              type="linear"
              dataKey={metric}
              dot={false}
              isAnimationActive={false}
              stroke="var(--chart-line)"
              strokeWidth={2.25}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="metric-guidance">
        <span>How to read movement</span>
        <p>{interpretation}</p>
      </div>
    </section>
  )
}

function HealthMetric({ label, value, suffix = '', warning = false }) {
  return (
    <div className={`health-metric ${warning ? 'health-warning' : ''}`}>
      <span>{label}</span>
      <strong>{value}{value !== '—' ? suffix : ''}</strong>
    </div>
  )
}

function MetadataRow({ label, children, mono = false }) {
  return (
    <div className="metadata-row">
      <dt>{label}</dt>
      <dd className={mono ? 'mono' : ''}>{children ?? '—'}</dd>
    </div>
  )
}

function runOptionLabel(run) {
  const version = run.repository_version || {}
  if (version.is_outdated) return `OUTDATED · ${run.name}`
  if (version.status === 'different') return `DIFFERENT VERSION · ${run.name}`
  if (version.status === 'newer') return `NEWER VERSION · ${run.name}`
  return run.name
}

function App() {
  const [runs, setRuns] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [detail, setDetail] = useState(null)
  const [telemetry, setTelemetry] = useState([])
  const [telemetrySourceCount, setTelemetrySourceCount] = useState(0)
  const [telemetryLoading, setTelemetryLoading] = useState(false)
  const [logs, setLogs] = useState([])
  const [health, setHealth] = useState(null)
  const [apiError, setApiError] = useState('')
  const [autoScroll, setAutoScroll] = useState(true)
  const [copyState, setCopyState] = useState('')
  const terminalRef = useRef(null)
  const healthRefreshInFlight = useRef(false)
  const runsRefreshInFlight = useRef(false)
  const selectedRefreshInFlight = useRef(false)
  const pendingLogLines = useRef([])
  const logFlushTimer = useRef(null)

  async function fetchJson(url, options) {
    const response = await fetch(url, options)
    if (!response.ok) {
      const body = await response.json().catch(() => ({}))
      throw new Error(body.detail || `${response.status} ${response.statusText}`)
    }
    return response.json()
  }

  async function refreshHealth() {
    if (healthRefreshInFlight.current) return
    healthRefreshInFlight.current = true
    try {
      const data = await fetchJson('/api/health')
      setHealth(data)
    } catch (error) {
      setHealth({ ok: false, problems: [error.message] })
    } finally {
      healthRefreshInFlight.current = false
    }
  }

  async function refreshRuns() {
    if (runsRefreshInFlight.current) return
    runsRefreshInFlight.current = true
    try {
      const data = await fetchJson('/api/runs')
      const nextRuns = data.runs || []
      setRuns(nextRuns)
      setSelectedId((current) => {
        if (current && nextRuns.some((run) => run.id === current)) return current
        return nextRuns[0]?.id || null
      })
      setApiError('')
    } catch (error) {
      setApiError(error.message)
    } finally {
      runsRefreshInFlight.current = false
    }
  }

  async function refreshSelected(id, { includeHistory = true } = {}) {
    if (!id) {
      setDetail(null)
      setTelemetry([])
      setTelemetrySourceCount(0)
      setTelemetryLoading(false)
      return
    }
    if (selectedRefreshInFlight.current) return
    selectedRefreshInFlight.current = true
    setTelemetryLoading(true)
    try {
      const runRequest = includeHistory
        ? fetchJson(`/api/runs/${id}`)
        : fetchJson(`/api/runs/${id}/progress`)
      const pointsRequest = includeHistory
        ? fetchJson(`/api/runs/${id}/telemetry?max_points=${CHART_RENDER_POINTS}`)
        : Promise.resolve(null)
      const [run, points] = await Promise.all([runRequest, pointsRequest])
      setDetail((current) => {
        if (includeHistory || !current) return run
        // Progress snapshots intentionally omit expensive detail fields.
        // Preserve the last full snapshot while replacing live state/telemetry.
        return { ...current, ...run }
      })
      if (points) {
        const records = points.records || []
        setTelemetry(records)
        setTelemetrySourceCount(points.source_records ?? records.length)
      } else if (run.telemetry) {
        setTelemetry((current) => {
          const next = [...current]
          const iteration = run.telemetry.iteration
          const index = next.findIndex((record) => record.iteration === iteration)
          if (index >= 0) next[index] = run.telemetry
          else next.push(run.telemetry)
          return next.slice(-CHART_RENDER_POINTS)
        })
      }
      setApiError('')
    } catch (error) {
      setApiError(error.message)
    } finally {
      selectedRefreshInFlight.current = false
      setTelemetryLoading(false)
    }
  }

  const selectedRunState = runs.find((run) => run.id === selectedId)?.state
  const selectedRunIsLive = ['starting', 'running', 'stopping'].includes(selectedRunState)

  useEffect(() => {
    refreshHealth()
    refreshRuns()
    const timer = setInterval(() => {
      refreshHealth()
      refreshRuns()
    }, POLL_MS)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    if (!selectedId) {
      setDetail(null)
      return undefined
    }
    setCopyState('')
    refreshSelected(selectedId, { includeHistory: true })
    // Archived runs have immutable telemetry. Re-reading their logs every
    // poll is expensive on a mounted workspace without yielding newer data.
    if (!selectedRunIsLive) return undefined
    // Re-sample the complete history on each live refresh. Appending one
    // snapshot per poll creates a sparse tail beside dense historical points,
    // which makes chart shape depend on refresh cadence rather than training.
    const timer = setInterval(() => refreshSelected(selectedId), POLL_MS)
    return () => clearInterval(timer)
  }, [selectedId, selectedRunIsLive])

  useEffect(() => {
    if (!selectedId) return undefined
    let cancelled = false
    setLogs([])
    pendingLogLines.current = []
    fetchJson(`/api/runs/${selectedId}/logs?tail=${LOG_TAIL_LIMIT}`)
      .then((data) => !cancelled && setLogs((data.lines || []).slice(-LOG_DISPLAY_LIMIT)))
      .catch(() => {})
    const source = new EventSource(`/api/runs/${selectedId}/logs/stream`)
    const flushLogLines = () => {
      logFlushTimer.current = null
      const lines = pendingLogLines.current.splice(0)
      if (lines.length) setLogs((current) => [...current, ...lines].slice(-LOG_DISPLAY_LIMIT))
    }
    source.onmessage = (event) => {
      try {
        const value = JSON.parse(event.data)
        if (typeof value.line === 'string') {
          pendingLogLines.current.push(value.line)
          if (logFlushTimer.current === null) {
            logFlushTimer.current = setTimeout(flushLogLines, LOG_FLUSH_MS)
          }
        }
      } catch (_) {
        // Ignore malformed log events; the stream will continue.
      }
    }
    return () => {
      cancelled = true
      source.close()
      pendingLogLines.current = []
      if (logFlushTimer.current !== null) clearTimeout(logFlushTimer.current)
      logFlushTimer.current = null
    }
  }, [selectedId])

  useEffect(() => {
    if (autoScroll && terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight
    }
  }, [logs, autoScroll])

  const chartRecords = useMemo(() => {
    const records = telemetry.map((record) => ({
      iteration: record.iteration ?? record.completed_steps,
      ...(record.metrics || {}),
      ...(record.canonical_metrics || {}),
    })).filter((record) => Number.isFinite(Number(record.iteration)))
    return sampleRecords(records, CHART_RENDER_POINTS)
  }, [telemetry])

  const availableCharts = useMemo(
    () => [...CANONICAL_CHARTS, ...ADVANCED_CHARTS].filter(({ key }) => (
      chartRecords.some((record) => Number.isFinite(Number(record[key])))
    )),
    [chartRecords],
  )

  const progress = detail?.telemetry || {}
  const canonical = detail?.training_health?.latest || progress.canonical_metrics || {}
  const percent = Number(progress.percent_complete || 0)
  const currentIteration = progress.iteration ?? progress.completed_steps
  const totalIterations = progress.total_iterations ?? progress.total_steps
  const displayState = detail?.stale ? 'stale' : detail?.state
  const runInfo = detail?.run_info || {}
  const repoVersion = detail?.repository_version || {}
  const processAlive = detail?.process?.alive
  const gpus = detail?.system?.gpus || []

  async function copyRunInformation() {
    if (!detail) return
    const payload = {
      run: detail.name,
      state: detail.state,
      stale: detail.stale,
      freshness_seconds: detail.freshness_seconds,
      repository_version: repoVersion,
      run_info: runInfo,
      training_health: detail.training_health,
    }
    try {
      await navigator.clipboard.writeText(JSON.stringify(payload, null, 2))
      setCopyState('Copied')
    } catch (_) {
      setCopyState('Copy failed')
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">ASCENTO / MJLAB</div>
          <h1>Training Monitor</h1>
        </div>
        <div className="topbar-actions">
          <ApiBadge health={health} />
          {detail && <StateBadge state={displayState} />}
          <select value={selectedId || ''} onChange={(event) => setSelectedId(event.target.value)}>
            {runs.length === 0 && <option value="">No runs found</option>}
            {runs.map((run) => (
              <option key={run.id} value={run.id}>{runOptionLabel(run)}</option>
            ))}
          </select>
        </div>
      </header>

      {apiError && <div className="api-error">{apiError}</div>}
      {health?.problems?.length > 0 && (
        <div className="api-error">
          <strong>Dashboard health check failed.</strong>
          <ul>{health.problems.map((problem) => <li key={problem}>{problem}</li>)}</ul>
        </div>
      )}
      {repoVersion.is_outdated && (
        <div className="stale-warning">
          <strong>Outdated run:</strong> this run was produced by repository {shortCommit(repoVersion.run_commit)}
          {repoVersion.run_branch ? ` (${repoVersion.run_branch})` : ''}, while the dashboard is running
          {' '}{shortCommit(repoVersion.current_commit)}{repoVersion.current_branch ? ` (${repoVersion.current_branch})` : ''}.
          {repoVersion.run_inferred ? ' The old commit was inferred from the checkout present immediately before a maintenance update.' : ''}
        </div>
      )}
      {repoVersion.status === 'different' && (
        <div className="stale-warning">
          <strong>Different repository version:</strong> this run came from {shortCommit(repoVersion.run_commit)}
          {repoVersion.run_branch ? ` (${repoVersion.run_branch})` : ''}, not the current checkout.
        </div>
      )}
      {detail?.stale && (
        <div className="stale-warning">
          Telemetry is stale: no update for {fmtDuration(detail.freshness_seconds)}.
          The run is still marked {detail.state}.
        </div>
      )}
      {detail?.training_health?.non_finite_updates > 0 && (
        <div className="api-error">
          NaN/Inf detected in {detail.training_health.non_finite_updates} telemetry update(s):
          {' '}{detail.training_health.non_finite_metrics.join(', ') || 'unknown metric'}.
        </div>
      )}
      {!detail && !apiError && <div className="empty-state">No training artifacts found yet.</div>}

      {detail && (
        <>
          <section className="stats-grid">
            <div className="stat-card"><span>Stage</span><strong>{detail.stage || '—'}</strong></div>
            <div className="stat-card"><span>Progress</span><strong>{fmtNumber(percent, 2)}%</strong></div>
            <div className="stat-card">
              <span>Iteration</span>
              <strong>{fmtNumber(currentIteration, 0)}</strong>
              <small>/ {fmtNumber(totalIterations, 0)}</small>
            </div>
            <div className="stat-card">
              <span>Freshness</span>
              <strong>{fmtDuration(detail.freshness_seconds)}</strong>
              <small>since last telemetry</small>
            </div>
            <div className="stat-card">
              <span>Throughput</span>
              <strong>{fmtNumber(progress.steps_per_second ?? canonical.throughput, 0)}</strong>
              <small>env steps/s</small>
            </div>
            <div className="stat-card"><span>Elapsed</span><strong>{fmtDuration(progress.elapsed_seconds)}</strong></div>
            <div className="stat-card"><span>ETA</span><strong>{fmtDuration(progress.eta_seconds)}</strong></div>
            <div className="stat-card">
              <span>Process</span>
              <strong>{processAlive === true ? 'alive' : processAlive === false ? 'stopped' : 'unknown'}</strong>
              <small>{detail.process?.pid ? `PID ${detail.process.pid}` : 'no PID recorded'}</small>
            </div>
          </section>

          <section className="panel progress-panel">
            <div className="progress-heading">
              <span>Training progress</span>
              <span>{fmtNumber(percent, 2)}%</span>
            </div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${Math.min(100, Math.max(0, percent))}%` }} />
            </div>
          </section>

          <section className="content-grid">
            <div className="charts-grid">
              <section className="panel charts-intro">
                <div>
                  <div className="section-kicker">Learning curves</div>
                  <div className="charts-intro-title">Read the whole run, not just the latest window</div>
                  <p>
                    The x-axis spans the oldest to newest available iteration and compresses as training grows.
                    Long runs are sampled evenly for display, so early training does not roll off the left side.
                  </p>
                </div>
                <div className="history-summary">
                  <span>{fmtNumber(telemetrySourceCount, 0)} source points</span>
                  <strong>{fmtNumber(chartRecords.length, 0)} drawn</strong>
                  {!selectedRunIsLive && (
                    <button
                      type="button"
                      disabled={telemetryLoading}
                      onClick={() => refreshSelected(selectedId)}
                    >
                      {telemetryLoading ? 'Loading history…' : 'Refresh history'}
                    </button>
                  )}
                </div>
              </section>
              {telemetryLoading && chartRecords.length === 0
                ? <section className="panel empty-panel" role="status">Loading complete telemetry history…</section>
                : availableCharts.length ? availableCharts.map((chart) => (
                <MetricChart key={chart.key} records={chartRecords} metric={chart.key} {...chart} />
                )) : <section className="panel empty-panel">Waiting for reward/PPO telemetry…</section>}
            </div>

            <aside className="side-column">
              <section className="panel">
                <div className="panel-title">Training health</div>
                <div className="health-grid">
                  <HealthMetric label="Reward" value={fmtNumber(canonical.reward, 4)} />
                  <HealthMetric label="Episode length" value={fmtNumber(canonical.episode_length, 2)} />
                  <HealthMetric label="PPO loss" value={fmtNumber(canonical.ppo_loss, 6)} />
                  <HealthMetric label="Entropy" value={fmtNumber(canonical.entropy, 6)} />
                  <HealthMetric label="KL" value={fmtNumber(canonical.kl, 6)} />
                  <HealthMetric label="Clip fraction" value={fmtNumber(canonical.clip_fraction, 5)} />
                  <HealthMetric label="Effort RMS" value={fmtNumber(canonical.effort_rms, 3)} />
                  <HealthMetric label="Saturation" value={fmtNumber(canonical.physical_saturation_fraction, 3)} />
                  <HealthMetric label="Recovery success" value={fmtNumber(canonical.recovery_success, 3)} />
                  <HealthMetric label="Recovery hold" value={fmtNumber(canonical.max_recovery_hold_s, 2)} />
                  <HealthMetric
                    label="Invalid updates"
                    value={fmtNumber(detail.training_health?.invalid_updates, 0)}
                    warning={(detail.training_health?.invalid_updates || 0) > 0}
                  />
                  <HealthMetric
                    label="NaN / Inf updates"
                    value={fmtNumber(detail.training_health?.non_finite_updates, 0)}
                    warning={(detail.training_health?.non_finite_updates || 0) > 0}
                  />
                </div>
              </section>

              <section className="panel">
                <div className="panel-title">GPU / process</div>
                <div className="gpu-summary">
                  <div><span>Process</span><strong>{processAlive === true ? 'alive' : processAlive === false ? 'stopped' : 'unknown'}</strong></div>
                  <div><span>PID</span><strong>{detail.process?.pid || '—'}</strong></div>
                  <div><span>Process GPU memory</span><strong>{fmtNumber(detail.system?.process_gpu_memory_mb, 0)} MB</strong></div>
                </div>
                {gpus.length ? gpus.map((gpu) => (
                  <div className="gpu-card" key={gpu.uuid || gpu.index}>
                    <div className="gpu-name">GPU {gpu.index}: {gpu.name}</div>
                    <div className="gpu-values">
                      <span>{fmtNumber(gpu.utilization_percent, 0)}% util</span>
                      <span>{fmtNumber(gpu.memory_used_mb, 0)} / {fmtNumber(gpu.memory_total_mb, 0)} MB</span>
                      <span>{fmtNumber(gpu.temperature_c, 0)} °C</span>
                    </div>
                  </div>
                )) : (
                  <div className="empty-panel">
                    {detail.system?.available === false ? 'nvidia-smi unavailable.' : 'No GPU telemetry available.'}
                  </div>
                )}
              </section>

              <section className={`panel error-panel ${detail.errors?.length ? 'has-errors' : ''}`}>
                <div className="panel-title">Errors</div>
                {detail.errors?.length
                  ? <pre>{detail.errors.join('\n')}</pre>
                  : <div className="ok-message">No recent errors detected.</div>}
              </section>
            </aside>
          </section>

          <section className="panel metadata-panel">
            <div className="panel-title-row">
              <div>
                <div className="panel-title">Run information</div>
                <div className="panel-subtitle">Reproducibility metadata captured by dashboard.launch and the run artifacts.</div>
              </div>
              <div className="button-row">
                <button onClick={copyRunInformation}>{copyState || 'Copy run information'}</button>
                <a
                  className="button-link"
                  href={`/api/runs/${selectedId}/summary.json`}
                  download="run-summary.json"
                >
                  Download run-summary.json
                </a>
              </div>
            </div>
            <dl className="metadata-grid">
              <MetadataRow label="Repository status">{repoVersion.status}</MetadataRow>
              <MetadataRow label="Run repository commit" mono>{repoVersion.run_commit || runInfo.git_commit}</MetadataRow>
              <MetadataRow label="Current repository commit" mono>{repoVersion.current_commit}</MetadataRow>
              <MetadataRow label="Git branch">{runInfo.git_branch || repoVersion.run_branch}</MetadataRow>
              <MetadataRow label="Task">{runInfo.task}</MetadataRow>
              <MetadataRow label="Stage">{runInfo.stage || detail.stage}</MetadataRow>
              <MetadataRow label="Seed">{runInfo.seed}</MetadataRow>
              <MetadataRow label="Device">{runInfo.device}</MetadataRow>
              <MetadataRow label="Simulation timestep">{runInfo.simulation_timestep}</MetadataRow>
              <MetadataRow label="Checkpoint / model" mono>{runInfo.checkpoint_path || runInfo.model_path}</MetadataRow>
              <MetadataRow label="Started">{fmtDate(runInfo.started_at)}</MetadataRow>
              <MetadataRow label="Finished">{fmtDate(runInfo.finished_at)}</MetadataRow>
              <MetadataRow label="Exit code">{runInfo.exit_code}</MetadataRow>
              <MetadataRow label="Command" mono>
                {Array.isArray(runInfo.command) ? runInfo.command.join(' ') : runInfo.command}
              </MetadataRow>
            </dl>
            <div className="config-files">
              <div className="panel-title">Configuration files</div>
              {runInfo.configuration_files?.length ? (
                <ul>
                  {runInfo.configuration_files.map((file) => (
                    <li key={file.path}><code>{file.path}</code> <span>{fmtNumber(file.size_bytes, 0)} bytes</span></li>
                  ))}
                </ul>
              ) : <div className="empty-panel">No run configuration files found.</div>}
            </div>
          </section>

          <section className="panel config-panel">
            <div className="panel-title-row">
              <div>
                <div className="panel-title">Dashboard configuration</div>
                <div className="panel-subtitle">Active server configuration; useful for diagnosing artifact discovery.</div>
              </div>
              <ApiBadge health={health} />
            </div>
            <dl className="metadata-grid compact">
              <MetadataRow label="Artifact root" mono>{health?.config?.artifact_root}</MetadataRow>
              <MetadataRow label="Frontend dist" mono>{health?.config?.frontend_dist}</MetadataRow>
              <MetadataRow label="Frontend built">{health?.config?.frontend_built ? 'yes' : 'no'}</MetadataRow>
              <MetadataRow label="Stale threshold">{health?.config?.stale_after_seconds ? `${health.config.stale_after_seconds}s` : '—'}</MetadataRow>
              <MetadataRow label="Python" mono>{health?.config?.python_executable}</MetadataRow>
              <MetadataRow label="Detected runs">{health?.run_count}</MetadataRow>
              <MetadataRow label="Current repo commit" mono>{health?.repository_version?.commit}</MetadataRow>
              <MetadataRow label="Current repo branch">{health?.repository_version?.branch}</MetadataRow>
            </dl>
            {health?.warnings?.length > 0 && (
              <ul className="warning-list">{health.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
            )}
          </section>

          <section className="panel terminal-panel">
            <div className="panel-title-row">
              <div className="panel-title">Console output</div>
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={autoScroll}
                  onChange={(event) => setAutoScroll(event.target.checked)}
                />
                auto-scroll
              </label>
            </div>
            <pre ref={terminalRef} className="terminal">
              {logs.length ? logs.join('\n') : 'No captured training.log for this run.'}
            </pre>
          </section>
        </>
      )}
    </main>
  )
}

export default App
