import { useEffect, useMemo, useState } from 'react'

const POLL_MS = 5000
const TASKS = [
  'Ascento-Balance-Flat',
  'Ascento-Velocity-Flat',
  'Ascento-Recovery-Flat',
  'Ascento-Jump-Flat',
]

async function fetchJson(url, options) {
  const response = await fetch(url, options)
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(body.detail || `${response.status} ${response.statusText}`)
  return body
}

function fmtDate(value) {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString()
}

function fmtNumber(value, digits = 2) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—'
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits })
}

function shortCommit(value) {
  return value ? String(value).slice(0, 10) : 'unknown'
}

function StateBadge({ state }) {
  return <span className={`runs-state runs-state-${state || 'unknown'}`}>{state || 'unknown'}</span>
}

function tagList(tags = []) {
  return tags.length ? tags.join(', ') : '—'
}

function defaultCreateForm() {
  return {
    display_name: '',
    task: 'Ascento-Balance-Flat',
    purpose: 'exploratory',
    tags: '',
    notes: '',
    parent_run_id: '',
    parent_checkpoint: '',
    training_args: '--env.scene.num-envs\n512\n--agent.max-iterations\n10000',
  }
}

function parseArgs(value) {
  return value.split('\n').map((line) => line.trim()).filter(Boolean)
}

function RunsPage() {
  const [runs, setRuns] = useState([])
  const [selectedId, setSelectedId] = useState('')
  const [detail, setDetail] = useState(null)
  const [createForm, setCreateForm] = useState(defaultCreateForm)
  const [editForm, setEditForm] = useState(null)
  const [compareIds, setCompareIds] = useState([])
  const [comparison, setComparison] = useState(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState(false)

  async function refreshRuns() {
    try {
      const data = await fetchJson('/api/runs')
      const next = data.runs || []
      setRuns(next)
      setSelectedId((current) => current && next.some((run) => run.id === current) ? current : (next[0]?.id || ''))
      setError('')
    } catch (caught) {
      setError(caught.message)
    }
  }

  async function refreshDetail(id) {
    if (!id) {
      setDetail(null)
      return
    }
    try {
      const value = await fetchJson(`/api/runs/${id}`)
      setDetail(value)
      setEditForm({
        display_name: value.display_name || value.name || '',
        purpose: value.metadata?.purpose || '',
        tags: (value.tags || []).join(', '),
        notes: value.notes || '',
        parent_run_id: value.lineage?.parent_run_id || '',
        parent_checkpoint: value.lineage?.parent_checkpoint || '',
      })
      setError('')
    } catch (caught) {
      setError(caught.message)
    }
  }

  useEffect(() => {
    refreshRuns()
    const timer = setInterval(refreshRuns, POLL_MS)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    refreshDetail(selectedId)
    if (!selectedId) return undefined
    const timer = setInterval(() => refreshDetail(selectedId), POLL_MS)
    return () => clearInterval(timer)
  }, [selectedId])

  const parentOptions = useMemo(
    () => runs.filter((run) => run.id !== selectedId),
    [runs, selectedId],
  )

  async function createRun(event) {
    event.preventDefault()
    setBusy(true)
    setNotice('')
    try {
      const payload = {
        display_name: createForm.display_name,
        task: createForm.task,
        purpose: createForm.purpose,
        tags: createForm.tags.split(',').map((tag) => tag.trim()).filter(Boolean),
        notes: createForm.notes,
        parent_run_id: createForm.parent_run_id || null,
        parent_checkpoint: createForm.parent_checkpoint || null,
        training_args: parseArgs(createForm.training_args),
      }
      const created = await fetchJson('/api/runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      setCreateForm(defaultCreateForm())
      setNotice(`Started ${created.display_name}.`)
      setSelectedId(created.id)
      setTimeout(refreshRuns, 500)
    } catch (caught) {
      setError(caught.message)
    } finally {
      setBusy(false)
    }
  }

  async function saveMetadata(event) {
    event.preventDefault()
    if (!selectedId || !editForm) return
    setBusy(true)
    setNotice('')
    try {
      const updated = await fetchJson(`/api/runs/${selectedId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...editForm,
          tags: editForm.tags.split(',').map((tag) => tag.trim()).filter(Boolean),
          parent_run_id: editForm.parent_run_id || null,
          parent_checkpoint: editForm.parent_checkpoint || null,
        }),
      })
      setDetail(updated)
      setNotice('Run metadata saved.')
      refreshRuns()
    } catch (caught) {
      setError(caught.message)
    } finally {
      setBusy(false)
    }
  }

  async function stopRun() {
    if (!selectedId) return
    setBusy(true)
    setNotice('')
    try {
      const updated = await fetchJson(`/api/runs/${selectedId}/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason: 'user_requested' }),
      })
      setDetail(updated)
      setNotice('Graceful stop requested.')
      refreshRuns()
    } catch (caught) {
      setError(caught.message)
    } finally {
      setBusy(false)
    }
  }

  function toggleCompare(id) {
    setComparison(null)
    setCompareIds((current) => {
      if (current.includes(id)) return current.filter((value) => value !== id)
      if (current.length >= 8) return current
      return [...current, id]
    })
  }

  async function compareRuns() {
    if (compareIds.length < 2) return
    setBusy(true)
    try {
      const data = await fetchJson(`/api/runs/compare?run_ids=${encodeURIComponent(compareIds.join(','))}`)
      setComparison(data)
      setError('')
    } catch (caught) {
      setError(caught.message)
    } finally {
      setBusy(false)
    }
  }

  const active = ['starting', 'running', 'stopping'].includes(detail?.state)

  return (
    <main className="runs-page">
      <section className="runs-hero">
        <div>
          <div className="eyebrow">RUN CONTROL / PROVENANCE</div>
          <h1>Runs</h1>
          <p>Start training, label experiments, preserve lineage, stop safely, and compare results from one place.</p>
        </div>
        <div className="runs-counts">
          <div><strong>{runs.length}</strong><span>discovered</span></div>
          <div><strong>{runs.filter((run) => ['starting', 'running', 'stopping'].includes(run.state)).length}</strong><span>active</span></div>
          <div><strong>{runs.filter((run) => run.repository_version?.is_outdated).length}</strong><span>outdated</span></div>
        </div>
      </section>

      {error && <div className="runs-alert runs-alert-error">{error}</div>}
      {notice && <div className="runs-alert runs-alert-ok">{notice}</div>}

      <section className="runs-layout">
        <div className="runs-main-column">
          <section className="runs-panel">
            <div className="runs-panel-heading">
              <div>
                <h2>Run library</h2>
                <p>Human names are separate from machine artifact directories. Existing runs can be annotated in place.</p>
              </div>
              <button className="runs-button subtle" onClick={refreshRuns}>Refresh</button>
            </div>
            <div className="runs-table-wrap">
              <table className="runs-table">
                <thead>
                  <tr>
                    <th>Compare</th>
                    <th>Run</th>
                    <th>State</th>
                    <th>Stage</th>
                    <th>Tags</th>
                    <th>Version</th>
                    <th>Progress</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((run) => (
                    <tr key={run.id} className={run.id === selectedId ? 'selected' : ''} onClick={() => setSelectedId(run.id)}>
                      <td onClick={(event) => event.stopPropagation()}>
                        <input type="checkbox" checked={compareIds.includes(run.id)} onChange={() => toggleCompare(run.id)} />
                      </td>
                      <td>
                        <strong>{run.display_name || run.name}</strong>
                        <small>{run.name}</small>
                      </td>
                      <td><StateBadge state={run.state} /></td>
                      <td>{run.stage || '—'}</td>
                      <td>{tagList(run.tags)}</td>
                      <td>
                        <span className={run.repository_version?.is_outdated ? 'runs-version-old' : ''}>
                          {run.repository_version?.status || 'unknown'}
                        </span>
                        <small>{shortCommit(run.repository_version?.run_commit)}</small>
                      </td>
                      <td>{fmtNumber(run.telemetry?.percent_complete, 1)}%</td>
                    </tr>
                  ))}
                  {runs.length === 0 && <tr><td colSpan="7" className="runs-empty">No runs found yet.</td></tr>}
                </tbody>
              </table>
            </div>
            <div className="runs-compare-bar">
              <span>{compareIds.length} selected</span>
              <button className="runs-button" disabled={compareIds.length < 2 || busy} onClick={compareRuns}>Compare selected</button>
              {compareIds.length > 0 && <button className="runs-button subtle" onClick={() => { setCompareIds([]); setComparison(null) }}>Clear</button>}
            </div>
          </section>

          {comparison && (
            <section className="runs-panel">
              <div className="runs-panel-heading"><div><h2>Comparison</h2><p>First selected run is the baseline; deltas are latest normalized telemetry.</p></div></div>
              <div className="runs-table-wrap">
                <table className="runs-table comparison-table">
                  <thead><tr><th>Run</th><th>Reward</th><th>Δ reward</th><th>Episode length</th><th>PPO loss</th><th>KL</th><th>Iteration</th></tr></thead>
                  <tbody>
                    {comparison.runs.map((run) => (
                      <tr key={run.id}>
                        <td><strong>{run.display_name}</strong>{run.id === comparison.baseline_id && <small>baseline</small>}</td>
                        <td>{fmtNumber(run.latest_metrics.reward, 4)}</td>
                        <td>{fmtNumber(run.delta_from_baseline.reward, 4)}</td>
                        <td>{fmtNumber(run.latest_metrics.episode_length, 2)}</td>
                        <td>{fmtNumber(run.latest_metrics.ppo_loss, 6)}</td>
                        <td>{fmtNumber(run.latest_metrics.kl, 6)}</td>
                        <td>{fmtNumber(run.iteration, 0)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </div>

        <aside className="runs-side-column">
          <section className="runs-panel">
            <div className="runs-panel-heading"><div><h2>New run</h2><p>Create a managed training process.</p></div></div>
            <form className="runs-form" onSubmit={createRun}>
              <label>Human name<input required value={createForm.display_name} onChange={(event) => setCreateForm({ ...createForm, display_name: event.target.value })} placeholder="Recovery baseline after PR83" /></label>
              <label>Task<select value={createForm.task} onChange={(event) => setCreateForm({ ...createForm, task: event.target.value })}>{TASKS.map((task) => <option key={task}>{task}</option>)}</select></label>
              <label>Purpose<select value={createForm.purpose} onChange={(event) => setCreateForm({ ...createForm, purpose: event.target.value })}><option>exploratory</option><option>baseline</option><option>tuning</option><option>validation</option><option>regression</option></select></label>
              <label>Tags<input value={createForm.tags} onChange={(event) => setCreateForm({ ...createForm, tags: event.target.value })} placeholder="recovery, pr83, baseline" /></label>
              <label>Parent run<select value={createForm.parent_run_id} onChange={(event) => setCreateForm({ ...createForm, parent_run_id: event.target.value })}><option value="">None</option>{runs.map((run) => <option key={run.id} value={run.id}>{run.display_name || run.name}</option>)}</select></label>
              <label>Parent checkpoint<input value={createForm.parent_checkpoint} onChange={(event) => setCreateForm({ ...createForm, parent_checkpoint: event.target.value })} placeholder="model_7500.pt" /></label>
              <label>Notes<textarea rows="3" value={createForm.notes} onChange={(event) => setCreateForm({ ...createForm, notes: event.target.value })} placeholder="Why this run exists and what should be learned from it." /></label>
              <label>Training arguments <small>one argument/value per line</small><textarea className="mono" rows="7" value={createForm.training_args} onChange={(event) => setCreateForm({ ...createForm, training_args: event.target.value })} /></label>
              <button className="runs-button primary" disabled={busy}>Start training</button>
            </form>
          </section>

          {detail && editForm && (
            <section className="runs-panel">
              <div className="runs-panel-heading">
                <div><h2>Selected run</h2><p>{detail.name}</p></div>
                <StateBadge state={detail.state} />
              </div>
              <div className="runs-summary-grid">
                <div><span>Task</span><strong>{detail.run_info?.task || '—'}</strong></div>
                <div><span>Iteration</span><strong>{fmtNumber(detail.telemetry?.iteration, 0)}</strong></div>
                <div><span>Started</span><strong>{fmtDate(detail.run_info?.started_at)}</strong></div>
                <div><span>Commit</span><strong className="mono">{shortCommit(detail.repository_version?.run_commit)}</strong></div>
              </div>
              {active && <button className="runs-button danger full" disabled={busy || detail.state === 'stopping'} onClick={stopRun}>{detail.state === 'stopping' ? 'Stopping…' : 'Graceful stop'}</button>}
              <form className="runs-form edit-form" onSubmit={saveMetadata}>
                <label>Human name<input value={editForm.display_name} onChange={(event) => setEditForm({ ...editForm, display_name: event.target.value })} /></label>
                <label>Purpose<input value={editForm.purpose} onChange={(event) => setEditForm({ ...editForm, purpose: event.target.value })} /></label>
                <label>Tags<input value={editForm.tags} onChange={(event) => setEditForm({ ...editForm, tags: event.target.value })} /></label>
                <label>Parent run<select value={editForm.parent_run_id} onChange={(event) => setEditForm({ ...editForm, parent_run_id: event.target.value })}><option value="">None</option>{parentOptions.map((run) => <option key={run.id} value={run.id}>{run.display_name || run.name}</option>)}</select></label>
                <label>Parent checkpoint<input value={editForm.parent_checkpoint} onChange={(event) => setEditForm({ ...editForm, parent_checkpoint: event.target.value })} /></label>
                <label>Notes<textarea rows="4" value={editForm.notes} onChange={(event) => setEditForm({ ...editForm, notes: event.target.value })} /></label>
                <button className="runs-button" disabled={busy}>Save metadata</button>
              </form>
            </section>
          )}
        </aside>
      </section>
    </main>
  )
}

export default RunsPage
