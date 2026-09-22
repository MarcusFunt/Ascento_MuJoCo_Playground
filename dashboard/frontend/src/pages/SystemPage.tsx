import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ExternalLink, RefreshCw, ShieldCheck, Unplug, UploadCloud } from 'lucide-react'
import { api } from '../api'
import { PageHeader } from '../components/PageHeader'
import { Badge } from '../components/ui/badge'
import { Button } from '../components/ui/button'
import { fmtDate, shortCommit } from '../lib/utils'

export function SystemPage() {
  const queryClient = useQueryClient()
  const status = useQuery({ queryKey: ['system'], queryFn: () => api.system(true), refetchInterval: 10_000 })
  const health = useQuery({ queryKey: ['health'], queryFn: api.health, refetchInterval: 15_000 })
  const update = useMutation({
    mutationFn: api.updateSystem,
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['system'] })
    },
  })

  const value = status.data || {}
  const repository = value.repository || {}
  const tailscale = value.tailscale || {}
  const updateState = value.update || {}
  const activeRuns = value.active_runs || []
  const blockers = value.update_blockers || []
  const incoming = Array.isArray(repository.incoming_commits) ? repository.incoming_commits : []
  const updateRunning = updateState.status === 'running'
  const updateAvailable = Boolean(repository.update_available)

  return (
    <>
      <PageHeader
        eyebrow="Workstation"
        title="System"
        description="Repository maintenance, Tailnet access and dashboard infrastructure."
        actions={<Button onClick={() => status.refetch()} disabled={status.isFetching}><RefreshCw size={16} /> Refresh status</Button>}
      />

      {status.error ? <div className="mb-6 rounded-xl border border-danger/40 bg-danger/10 p-4 text-sm text-danger">{status.error.message}</div> : null}

      <section className="rounded-xl border border-border bg-panel p-6 lg:p-7">
        <div className="flex flex-wrap items-start justify-between gap-5">
          <div>
            <div className="text-xs font-bold uppercase tracking-[0.1em] text-muted">Repository state</div>
            <h2 className="mt-2 text-2xl font-semibold tracking-[-0.03em]">
              {!value.connected ? 'Host supervisor unavailable' : updateRunning ? 'Updating workstation' : updateAvailable ? `${repository.behind_by || 0} commit${repository.behind_by === 1 ? '' : 's'} available` : 'Repository is current'}
            </h2>
            <p className="mt-2 text-sm text-muted">Local <code>{shortCommit(repository.local_commit)}</code> · origin/main <code>{shortCommit(repository.remote_commit)}</code></p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <Badge className={value.connected ? 'border-success/40 bg-success/10 text-success' : 'border-danger/40 bg-danger/10 text-danger'}>{value.connected ? 'Supervisor connected' : 'Supervisor offline'}</Badge>
            <Button variant="primary" disabled={!value.can_update || updateRunning || update.isPending} onClick={() => update.mutate()}>
              <UploadCloud size={16} /> {updateRunning || update.isPending ? 'Updating…' : updateAvailable ? 'Update to latest main' : 'Up to date'}
            </Button>
          </div>
        </div>
      </section>

      {update.error ? <div className="mt-5 rounded-xl border border-danger/40 bg-danger/10 p-4 text-sm text-danger">{update.error.message}</div> : null}

      <div className="mt-6 grid gap-6 xl:grid-cols-2">
        <SystemCard title="Repository" icon={<RefreshCw size={16} />}>
          <KeyValue rows={[
            ['Branch', repository.branch],
            ['Local commit', shortCommit(repository.local_commit)],
            ['Remote main', shortCommit(repository.remote_commit)],
            ['Ahead / behind', `${repository.ahead_by ?? '—'} / ${repository.behind_by ?? '—'}`],
            ['Tracked changes', repository.dirty ? 'dirty' : 'clean'],
            ['Checked', fmtDate(value.checked_at)],
          ]} />
          {repository.remote_error ? <Warning>{String(repository.remote_error)}</Warning> : null}
        </SystemCard>

        <SystemCard title="Tailnet access" icon={tailscale.connected ? <ShieldCheck size={16} /> : <Unplug size={16} />}>
          <div className="mb-4">
            <Badge className={tailscale.connected ? 'border-success/40 bg-success/10 text-success' : 'border-warning/40 bg-warning/10 text-warning'}>
              {tailscale.connected ? 'Connected' : tailscale.enabled ? 'Disconnected' : 'Not enrolled'}
            </Badge>
          </div>
          <KeyValue rows={[
            ['MagicDNS', tailscale.dns_name],
            ['Tailscale IP', Array.isArray(tailscale.ips) ? tailscale.ips.join(', ') : '—'],
          ]} />
          {tailscale.url ? <a href={String(tailscale.url)} target="_blank" rel="noreferrer" className="control-focus mt-4 inline-flex items-center gap-2 rounded-lg text-sm font-semibold text-secondary hover:text-foreground">Open remote dashboard <ExternalLink size={14} /></a> : null}
          {tailscale.error ? <Warning>{String(tailscale.error)}</Warning> : null}
        </SystemCard>

        <SystemCard title="Dashboard index" icon={<ShieldCheck size={16} />}>
          <KeyValue rows={[
            ['API', health.data?.ok ? 'healthy' : health.data ? 'degraded' : 'checking'],
            ['Database', health.data?.database?.available ? 'connected' : health.data?.database?.enabled ? 'filesystem fallback' : 'optional'],
            ['Backend', health.data?.database?.backend || '—'],
            ['Artifact root', health.data?.artifact_root || '—'],
          ]} />
          {health.data?.database?.error ? <Warning>{String(health.data.database.error)}</Warning> : null}
        </SystemCard>

        <SystemCard title="Last update" icon={<UploadCloud size={16} />}>
          <KeyValue rows={[
            ['Status', updateState.status || 'idle'],
            ['Started', fmtDate(updateState.started_at)],
            ['Finished', fmtDate(updateState.finished_at)],
            ['From', shortCommit(updateState.from_commit)],
            ['Target', shortCommit(updateState.target_commit)],
            ['Exit code', updateState.return_code ?? '—'],
          ]} />
        </SystemCard>
      </div>

      {blockers.length ? (
        <section className="mt-6 rounded-xl border border-warning/30 bg-warning/5 p-6">
          <div className="text-xs font-bold uppercase tracking-[0.1em] text-warning">Why update is blocked</div>
          <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-secondary">{blockers.map((item: string) => <li key={item}>{item}</li>)}</ul>
        </section>
      ) : null}

      {activeRuns.length ? (
        <section className="mt-6 rounded-xl border border-border bg-panel p-6">
          <div className="text-xs font-bold uppercase tracking-[0.1em] text-muted">Active runs</div>
          <p className="mt-2 text-sm text-muted">Repository updates remain disabled while training is active so a rebuild cannot terminate a run.</p>
          <div className="mt-4 divide-y divide-border">
            {activeRuns.map((run: any) => (
              <div key={`${run.path}:${run.state}`} className="grid gap-1 py-3 sm:grid-cols-[1fr_auto]">
                <strong className="text-sm">{run.name}</strong><span className="text-sm text-warning">{run.state}</span>
                <code className="col-span-full break-all text-xs text-muted">{run.path}</code>
              </div>
            ))}
          </div>
        </section>
      ) : null}

      {incoming.length ? (
        <section className="mt-6 rounded-xl border border-border bg-panel p-6">
          <div className="text-xs font-bold uppercase tracking-[0.1em] text-muted">Incoming commits</div>
          <div className="mt-4 divide-y divide-border">
            {incoming.map((commit: any) => (
              <div key={commit.commit} className="flex gap-4 py-3 text-sm">
                <code className="shrink-0 text-muted">{shortCommit(commit.commit)}</code>
                <span className="text-secondary">{commit.subject}</span>
              </div>
            ))}
          </div>
        </section>
      ) : null}
    </>
  )
}

function SystemCard({ title, icon, children }: { title: string; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="rounded-xl border border-border bg-panel p-6">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-[17px] font-semibold">{title}</h2>
        <span className="text-muted">{icon}</span>
      </div>
      <div className="mt-5">{children}</div>
    </section>
  )
}

function KeyValue({ rows }: { rows: Array<[string, unknown]> }) {
  return (
    <dl className="divide-y divide-border">
      {rows.map(([label, raw]) => (
        <div key={label} className="grid grid-cols-[140px_1fr] gap-4 py-2.5 text-sm first:pt-0 last:pb-0">
          <dt className="text-muted">{label}</dt>
          <dd className="m-0 min-w-0 break-words text-right text-secondary">{raw === null || raw === undefined || raw === '' ? '—' : String(raw)}</dd>
        </div>
      ))}
    </dl>
  )
}

function Warning({ children }: { children: React.ReactNode }) {
  return <div className="mt-4 rounded-lg border border-warning/30 bg-warning/10 px-3 py-2 text-xs leading-relaxed text-warning">{children}</div>
}
