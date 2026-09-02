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

const POLL_MS = 5000
const METRIC_PRIORITY = [
  'eval/episode_reward',
  'training/total_loss',
  'training/invalid_update',
  'eval/avg_episode_length',
  'training/policy_loss',
  'training/v_loss',
  'training/entropy_loss',
]

function fmtNumber(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits })
}

function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined || !Number.isFinite(Number(seconds))) return '—'
  const total = Math.max(0, Math.round(Number(seconds)))
  const hours = Math.floor(total / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  const secs = total % 60
  if (hours) return `${hours}h ${minutes}m`
  if (minutes) return `${minutes}m ${secs}s`
  return `${secs}s`
}

function StateBadge({ state }) {
  return <span className={`state-badge state-${state || 'unknown'}`}>{state || 'unknown'}</span>
}

function MetricChart({ records, metric }) {
  return (
    <section className="panel metric-panel">
      <div className="panel-title">{metric}</div>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={records} margin={{ top: 8, right: 14, left: -8, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="step" tickFormatter={(value) => `${fmtNumber(value / 1e6, 1)}M`} minTickGap={24} />
            <YAxis width={58} tickFormatter={(value) => fmtNumber(value, 2)} domain={['auto', 'auto']} />
            <Tooltip
              formatter={(value) => fmtNumber(value, 5)}
              labelFormatter={(step) => `${fmtNumber(step, 0)} steps`}
            />
            <Line type="monotone" dataKey={metric} dot={false} isAnimationActive={false} strokeWidth={2} connectNulls />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  )
}

function App() {
  const [runs, setRuns] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [detail, setDetail] = useState(null)
  const [telemetry, setTelemetry] = useState([])
  const [logs, setLogs] = useState([])
  const [apiError, setApiError] = useState('')
  const [renderState, setRenderState] = useState('idle')
  const [autoScroll, setAutoScroll] = useState(true)
  const terminalRef = useRef(null)

  async function fetchJson(url, options) {
    const response = await fetch(url, options)
    if (!response.ok) {
      const body = await response.json().catch(() => ({}))
      throw new Error(body.detail || `${response.status} ${response.statusText}`)
    }
    return response.json()
  }

  async function refreshRuns() {
    try {
      const data = await fetchJson('/api/runs')
      setRuns(data.runs || [])
      setSelectedId((current) => current || data.runs?.[0]?.id || null)
      setApiError('')
    } catch (error) {
      setApiError(error.message)
    }
  }

  async function refreshSelected(id) {
    if (!id) return
    try {
      const [run, points, render] = await Promise.all([
        fetchJson(`/api/runs/${id}`),
        fetchJson(`/api/runs/${id}/telemetry?limit=3000`),
        fetchJson(`/api/runs/${id}/render-status`),
      ])
      setDetail(run)
      setTelemetry(points.records || [])
      setRenderState(render.state || 'idle')
      setApiError('')
    } catch (error) {
      setApiError(error.message)
    }
  }

  useEffect(() => {
    refreshRuns()
    const timer = setInterval(refreshRuns, POLL_MS)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    if (!selectedId) return undefined
    refreshSelected(selectedId)
    const timer = setInterval(() => refreshSelected(selectedId), POLL_MS)
    return () => clearInterval(timer)
  }, [selectedId])

  useEffect(() => {
    if (!selectedId) return undefined
    let cancelled = false
    setLogs([])
    fetchJson(`/api/runs/${selectedId}/logs?tail=800`)
      .then((data) => !cancelled && setLogs(data.lines || []))
      .catch(() => {})
    const source = new EventSource(`/api/runs/${selectedId}/logs/stream`)
    source.onmessage = (event) => {
      try {
        const value = JSON.parse(event.data)
        if (typeof value.line === 'string') {
          setLogs((current) => [...current, value.line].slice(-1500))
        }
      } catch (_) {
        // Ignore malformed log events; the stream will continue.
      }
    }
    return () => {
      cancelled = true
      source.close()
    }
  }, [selectedId])

  useEffect(() => {
    if (autoScroll && terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight
    }
  }, [logs, autoScroll])

  const chartRecords = useMemo(
    () => telemetry.map((record) => ({ step: record.completed_steps, ...(record.metrics || {}) })),
    [telemetry],
  )

  const metricKeys = useMemo(() => {
    const keys = [...new Set(telemetry.flatMap((record) => Object.keys(record.metrics || {})))]
    const preferred = METRIC_PRIORITY.filter((key) => keys.includes(key))
    const fallback = keys.filter((key) => !preferred.includes(key) && /reward|episode|loss|entropy|eval|surviv|fall/i.test(key))
    return [...preferred, ...fallback, ...keys.filter((key) => !preferred.includes(key) && !fallback.includes(key))].slice(0, 4)
  }, [telemetry])

  const progress = detail?.telemetry || {}
  const percent = Number(progress.percent_complete || 0)
  const stage = progress.stage || detail?.status?.stage || detail?.stage
  const trainingActive = detail?.state === 'running' || detail?.state === 'starting'
  const renderDisabled = !detail?.has_checkpoint || trainingActive || renderState === 'running' || renderState === 'starting'

  async function requestRender() {
    if (!selectedId) return
    setRenderState('starting')
    try {
      await fetchJson(`/api/runs/${selectedId}/render-latest`, { method: 'POST' })
      setRenderState('running')
      window.setTimeout(() => refreshSelected(selectedId), 1500)
    } catch (error) {
      setApiError(error.message)
      setRenderState('error')
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">ASCENTO / MJX</div>
          <h1>Training Monitor</h1>
        </div>
        <div className="topbar-actions">
          {detail && <StateBadge state={detail.state} />}
          <select value={selectedId || ''} onChange={(event) => setSelectedId(event.target.value)}>
            {runs.map((run) => (
              <option key={run.id} value={run.id}>{run.name}</option>
            ))}
          </select>
        </div>
      </header>

      {apiError && <div className="api-error">{apiError}</div>}
      {!detail && !apiError && <div className="empty-state">No training artifacts found yet.</div>}

      {detail && (
        <>
          <section className="stats-grid">
            <div className="stat-card"><span>Stage</span><strong>{stage}</strong></div>
            <div className="stat-card"><span>Progress</span><strong>{fmtNumber(percent, 2)}%</strong></div>
            <div className="stat-card"><span>Steps</span><strong>{fmtNumber(progress.completed_steps, 0)}</strong><small>/ {fmtNumber(progress.total_steps, 0)}</small></div>
            <div className="stat-card"><span>Throughput</span><strong>{fmtNumber(progress.steps_per_second, 0)}</strong><small>steps/s</small></div>
            <div className="stat-card"><span>Elapsed</span><strong>{fmtDuration(progress.elapsed_seconds)}</strong></div>
            <div className="stat-card"><span>ETA</span><strong>{fmtDuration(progress.eta_seconds)}</strong></div>
          </section>

          <section className="panel progress-panel">
            <div className="progress-heading"><span>Training progress</span><span>{fmtNumber(percent, 2)}%</span></div>
            <div className="progress-track"><div className="progress-fill" style={{ width: `${Math.min(100, Math.max(0, percent))}%` }} /></div>
          </section>

          <section className="content-grid">
            <div className="charts-grid">
              {metricKeys.length ? metricKeys.map((metric) => (
                <MetricChart key={metric} records={chartRecords} metric={metric} />
              )) : <section className="panel empty-panel">Waiting for metric telemetry…</section>}
            </div>

            <aside className="side-column">
              <section className="panel render-panel">
                <div className="panel-title-row">
                  <div className="panel-title">Latest rollout</div>
                  <button
                    onClick={requestRender}
                    disabled={renderDisabled}
                    title={trainingActive ? 'Rendering is disabled while training is running to protect GPU throughput.' : undefined}
                  >
                    {renderState === 'running' || renderState === 'starting' ? 'Rendering…' : 'Render latest'}
                  </button>
                </div>
                {detail.latest_render ? (
                  <>
                    <img src={`/api/runs/${detail.id}/render/latest?t=${detail.modified_at}`} alt="Latest deterministic MuJoCo rollout preview" />
                    <div className="render-meta">
                      {['step', 'return', 'survival_steps', 'min_height', 'max_abs_qvel', 'action_saturation_fraction'].map((key) => (
                        detail.latest_render[key] !== undefined && <div key={key}><span>{key}</span><strong>{fmtNumber(detail.latest_render[key], 3)}</strong></div>
                      ))}
                    </div>
                  </>
                ) : <div className="empty-panel">No preview rendered yet.</div>}
              </section>

              <section className={`panel error-panel ${detail.errors?.length ? 'has-errors' : ''}`}>
                <div className="panel-title">Errors</div>
                {detail.errors?.length ? <pre>{detail.errors.join('\n')}</pre> : <div className="ok-message">No recent errors detected.</div>}
              </section>
            </aside>
          </section>

          <section className="panel terminal-panel">
            <div className="panel-title-row">
              <div className="panel-title">Console output</div>
              <label className="checkbox"><input type="checkbox" checked={autoScroll} onChange={(event) => setAutoScroll(event.target.checked)} /> auto-scroll</label>
            </div>
            <pre ref={terminalRef} className="terminal">{logs.length ? logs.join('\n') : 'No captured training.log for this run.'}</pre>
          </section>
        </>
      )}
    </main>
  )
}

export default App
