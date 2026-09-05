import { useEffect, useMemo, useState } from 'react'
import './system.css'

const POLL_MS = 10000

function shortCommit(value) {
  return value ? String(value).slice(0, 10) : 'unknown'
}

function fmtDate(value) {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString()
}

function StatusPill({ good, children }) {
  return <span className={`system-pill ${good ? 'good' : 'warn'}`}>{children}</span>
}

function SystemPage() {
  const [status, setStatus] = useState(null)
  const [error, setError] = useState('')
  const [action, setAction] = useState('')

  async function fetchJson(url, options) {
    const response = await fetch(url, options)
    const body = await response.json().catch(() => ({}))
    if (!response.ok) throw new Error(body.detail || `${response.status} ${response.statusText}`)
    return body
  }

  async function refresh(force = false) {
    try {
      const data = await fetchJson(`/api/system${force ? '?refresh=true' : ''}`)
      setStatus(data)
      setError('')
    } catch (err) {
      setError(err.message)
    }
  }

  useEffect(() => {
    refresh(true)
    const timer = setInterval(() => refresh(false), POLL_MS)
    return () => clearInterval(timer)
  }, [])

  async function updateSystem() {
    setAction('Starting update…')
    try {
      await fetchJson('/api/system/update', {
        method: 'POST',
        headers: { 'X-Ascento-Control': '1' },
      })
      setAction('Update started. The dashboard may disconnect briefly while containers rebuild.')
      await refresh(false)
    } catch (err) {
      setAction(`Update not started: ${err.message}`)
      await refresh(true)
    }
  }

  const repository = status?.repository || {}
  const update = status?.update || {}
  const tailscale = status?.tailscale || {}
  const activeRuns = status?.active_runs || []
  const blockers = status?.update_blockers || []
  const incoming = repository.incoming_commits || []
  const updateRunning = update.status === 'running'
  const updateAvailable = Boolean(repository.update_available)

  const headline = useMemo(() => {
    if (!status?.connected) return 'Host supervisor unavailable'
    if (updateRunning) return 'Updating workstation'
    if (updateAvailable) return `${repository.behind_by || 0} commit${repository.behind_by === 1 ? '' : 's'} available`
    return 'Repository is current'
  }, [status, repository.behind_by, updateAvailable, updateRunning])

  return (
    <main className="system-page">
      <header className="system-header">
        <div>
          <div className="system-eyebrow">WORKSTATION / CONTROL</div>
          <h1>System</h1>
          <p>Repository updates, host-control health, and remote Tailnet access.</p>
        </div>
        <button className="secondary-button" onClick={() => refresh(true)}>Refresh remote status</button>
      </header>

      {error && <div className="system-banner danger">{error}</div>}
      {action && <div className="system-banner">{action}</div>}

      <section className="system-hero panel-like">
        <div>
          <div className="system-section-label">Update state</div>
          <h2>{headline}</h2>
          <p>
            Local <code>{shortCommit(repository.local_commit)}</code>
            {' · '}origin/main <code>{shortCommit(repository.remote_commit)}</code>
          </p>
        </div>
        <div className="system-hero-actions">
          <StatusPill good={Boolean(status?.connected)}>
            {status?.connected ? 'Supervisor connected' : 'Supervisor offline'}
          </StatusPill>
          <button
            className="primary-button"
            disabled={!status?.can_update || updateRunning}
            onClick={updateSystem}
          >
            {updateRunning ? 'Updating…' : updateAvailable ? 'Update to latest main' : 'Up to date'}
          </button>
        </div>
      </section>

      <section className="system-grid">
        <article className="panel-like system-card">
          <div className="system-card-title">Repository</div>
          <dl className="system-kv">
            <div><dt>Branch</dt><dd>{repository.branch || '—'}</dd></div>
            <div><dt>Local commit</dt><dd><code>{shortCommit(repository.local_commit)}</code></dd></div>
            <div><dt>Remote main</dt><dd><code>{shortCommit(repository.remote_commit)}</code></dd></div>
            <div><dt>Ahead / behind</dt><dd>{repository.ahead_by ?? '—'} / {repository.behind_by ?? '—'}</dd></div>
            <div><dt>Tracked changes</dt><dd>{repository.dirty ? 'dirty' : 'clean'}</dd></div>
            <div><dt>Checked</dt><dd>{fmtDate(status?.checked_at)}</dd></div>
          </dl>
          {repository.remote_error && <div className="inline-warning">Remote check: {repository.remote_error}</div>}
        </article>

        <article className="panel-like system-card">
          <div className="system-card-title">Tailnet access</div>
          <div className="tailnet-state">
            <StatusPill good={Boolean(tailscale.connected)}>
              {tailscale.connected ? 'Connected' : tailscale.enabled ? 'Disconnected' : 'Not enrolled'}
            </StatusPill>
          </div>
          <dl className="system-kv">
            <div><dt>MagicDNS</dt><dd>{tailscale.dns_name || '—'}</dd></div>
            <div><dt>Tailscale IP</dt><dd>{tailscale.ips?.join(', ') || '—'}</dd></div>
          </dl>
          {tailscale.url ? (
            <a className="tailnet-link" href={tailscale.url} target="_blank" rel="noreferrer">
              Open remote dashboard
            </a>
          ) : (
            <div className="system-help">
              Enroll once on the workstation with <code>scripts/setup_tailscale.sh</code> and a Tailscale auth key.
              The key is not committed or retained in the running container after enrollment.
            </div>
          )}
          {tailscale.error && <div className="inline-warning">{tailscale.error}</div>}
        </article>

        <article className="panel-like system-card">
          <div className="system-card-title">Host supervisor</div>
          <p className="system-help">
            The web container has no Docker socket and no arbitrary shell endpoint. It can request only repository status or a guarded maintenance update over a Unix socket.
          </p>
          {!status?.connected && (
            <div className="inline-warning">
              Install the host boundary once with <code>scripts/install_supervisor.sh</code>.
            </div>
          )}
          {status?.error && <div className="inline-warning">{status.error}</div>}
        </article>

        <article className="panel-like system-card">
          <div className="system-card-title">Last update</div>
          <dl className="system-kv">
            <div><dt>Status</dt><dd>{update.status || 'idle'}</dd></div>
            <div><dt>Started</dt><dd>{fmtDate(update.started_at)}</dd></div>
            <div><dt>Finished</dt><dd>{fmtDate(update.finished_at)}</dd></div>
            <div><dt>From</dt><dd><code>{shortCommit(update.from_commit)}</code></dd></div>
            <div><dt>Target</dt><dd><code>{shortCommit(update.target_commit)}</code></dd></div>
            <div><dt>Exit code</dt><dd>{update.return_code ?? '—'}</dd></div>
          </dl>
        </article>
      </section>

      {blockers.length > 0 && (
        <section className="panel-like system-section">
          <div className="system-card-title">Why update is blocked</div>
          <ul>{blockers.map((item) => <li key={item}>{item}</li>)}</ul>
        </section>
      )}

      {activeRuns.length > 0 && (
        <section className="panel-like system-section">
          <div className="system-card-title">Active runs</div>
          <p>Repository updates are deliberately disabled while training is active so rebuilding the Dashboard cannot terminate a run.</p>
          <div className="active-run-list">
            {activeRuns.map((run) => (
              <div key={`${run.path}:${run.state}`}>
                <strong>{run.name}</strong><span>{run.state}</span><code>{run.path}</code>
              </div>
            ))}
          </div>
        </section>
      )}

      {incoming.length > 0 && (
        <section className="panel-like system-section">
          <div className="system-card-title">Incoming commits</div>
          <div className="incoming-list">
            {incoming.map((commit) => (
              <div key={commit.commit}>
                <code>{shortCommit(commit.commit)}</code>
                <span>{commit.subject}</span>
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="system-banner subtle">
        An update can temporarily interrupt the web connection while Docker rebuilds the Dashboard and Tailscale sidecar. The host supervisor remains outside Docker, finishes the update, and the persistent Tailnet node reconnects automatically.
      </section>
    </main>
  )
}

export default SystemPage
