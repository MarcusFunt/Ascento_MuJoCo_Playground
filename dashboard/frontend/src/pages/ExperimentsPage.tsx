import { useQuery } from '@tanstack/react-query'
import { Link, useParams } from '@tanstack/react-router'
import { ArrowRight, FlaskConical, Link2Off } from 'lucide-react'
import { api } from '../api'
import { PageHeader } from '../components/PageHeader'
import { StateBadge } from '../components/StateBadge'

export function ExperimentsPage() {
  const { experimentId } = useParams({ strict: false }) as { experimentId?: string }
  const query = useQuery({ queryKey: ['experiments'], queryFn: api.experiments, refetchInterval: 30_000 })
  const program = query.data?.programs.find((item: any) => item.id === experimentId)

  return (
    <>
      <PageHeader
        eyebrow="Research program"
        title={program ? String(program.title || program.id) : 'Experiments'}
        description="Declared plans, observed results and explicitly linked run attempts. Similar names are never treated as lineage."
        actions={experimentId ? <Link to="/experiments" className="control-focus inline-flex h-10 items-center gap-2 rounded-lg border border-border-strong bg-raised px-4 text-sm font-semibold">All experiments <ArrowRight size={15} /></Link> : null}
      />

      {query.error ? <div className="mb-5 rounded-xl border border-danger/40 bg-danger/10 p-4 text-sm text-danger">{query.error.message}</div> : null}
      {query.isLoading ? <div className="h-64 animate-pulse rounded-xl border border-border bg-panel" /> : null}
      {!query.isLoading && !query.error && !experimentId ? (
        <div className="space-y-4">
          {(query.data?.programs || []).map((item: any) => (
            <Link key={item.id} to="/experiments/$experimentId" params={{ experimentId: item.id }} className="control-focus block rounded-xl border border-border bg-panel p-5 transition-colors hover:bg-hover">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div className="min-w-0">
                  <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.08em] text-muted"><FlaskConical size={14} /> Program plan</div>
                  <h2 className="mt-2 truncate text-lg font-semibold">{item.title || item.id}</h2>
                  <p className="mt-1 text-xs text-muted">{item.id} · {item.source_path} · schema {item.source_schema_version ?? 'unknown'}</p>
                </div>
                <StateBadge state={String(item.status || 'unknown')} />
              </div>
              <div className="mt-4 grid gap-3 text-sm sm:grid-cols-3">
                <SummaryCell label="Declared arms" value={item.arms?.length ?? 0} />
                <SummaryCell label="Explicitly linked runs" value={item.linked_runs?.length ?? 0} />
                <SummaryCell label="Unlinked attempts" value={item.unlinked_run_count ?? 0} />
              </div>
            </Link>
          ))}
          {!query.data?.programs.length ? <div className="rounded-xl border border-border bg-panel p-8 text-center text-sm text-muted">No experiment plan files are available under the approved experiment registry.</div> : null}
        </div>
      ) : null}

      {experimentId && !query.isLoading && !query.error && !program ? <div className="rounded-xl border border-danger/40 bg-danger/10 p-5 text-sm text-danger">Experiment plan not found.</div> : null}
      {program ? (
        <div className="space-y-6">
          <section className="grid gap-4 rounded-xl border border-border bg-panel p-5 sm:grid-cols-2 xl:grid-cols-4">
            <SummaryCell label="Plan status (declared)" value={program.status || 'not declared'} />
            <SummaryCell label="Target robot" value={program.target_robot || 'not declared'} />
            <SummaryCell label="Baseline checkpoint" value={program.baseline?.checkpoint || 'not declared'} />
            <SummaryCell label="Baseline result (declared)" value={program.baseline?.evaluation_status || 'not declared'} />
          </section>

          <section className="overflow-hidden rounded-xl border border-border bg-panel">
            <div className="border-b border-border px-5 py-4"><h2 className="font-semibold">Plan arms and observed outcomes</h2><p className="mt-1 text-xs text-muted">The source plan is authoritative for declared status. Saved run and evaluation records supply observed status.</p></div>
            <div className="divide-y divide-border">
              {(program.arms || []).map((arm: any) => (
                <div key={arm.id} className="grid gap-3 px-5 py-4 md:grid-cols-[1fr_auto_auto] md:items-start">
                  <div><strong>{arm.id}</strong><div className="mt-1 text-xs text-muted">{arm.run_attempt_count ?? '—'} attempt(s) listed in the plan artifact</div></div>
                  <SummaryCell label="Declared status" value={arm.declared_status || 'not declared'} />
                  <SummaryCell label="Observed outcome" value={arm.observed_outcome || 'not recorded'} />
                  {arm.run_attempts?.length ? <div className="col-span-full rounded-lg bg-background/40 p-3 text-xs text-secondary"><pre className="overflow-x-auto whitespace-pre-wrap">{JSON.stringify(arm.run_attempts, null, 2)}</pre></div> : null}
                </div>
              ))}
              {!program.arms?.length ? <p className="p-5 text-sm text-muted">This plan has no separately declared experiment arms.</p> : null}
            </div>
          </section>

          <section className="overflow-hidden rounded-xl border border-border bg-panel">
            <div className="border-b border-border px-5 py-4"><h2 className="font-semibold">Explicit run lineage</h2></div>
            {program.linked_runs?.length ? (
              <div className="divide-y divide-border">{program.linked_runs.map((item: any) => (
                <Link key={item.run.id} to="/runs/$runId" params={{ runId: item.run.id }} className="control-focus flex flex-wrap items-center justify-between gap-3 px-5 py-4 hover:bg-hover">
                  <div><strong>{item.run.display_name || item.run.name}</strong><div className="mt-1 text-xs text-muted">{item.run.task || item.run.stage || 'Unknown task'} · linked by {item.link_source}</div></div>
                  <StateBadge state={item.run.state} stale={item.run.stale} />
                </Link>
              ))}</div>
            ) : <p className="p-5 text-sm text-muted">No run is explicitly linked to this plan. No parent is inferred from names or checkpoint similarity.</p>}
          </section>

          <section className="overflow-hidden rounded-xl border border-warning/30 bg-warning/5">
            <div className="flex items-center gap-2 border-b border-warning/20 px-5 py-4 text-sm font-semibold text-warning"><Link2Off size={16} /> Unlinked run attempts</div>
            {program.unlinked_runs?.length ? (
              <div className="divide-y divide-warning/15">{program.unlinked_runs.map((run: any) => (
                <div key={run.id} className="flex flex-wrap items-center justify-between gap-3 px-5 py-3 text-sm">
                  <div><strong>{run.display_name || run.name}</strong><div className="mt-1 text-xs text-muted">{run.state} · {run.task || run.stage || 'Unknown task'}</div></div>
                  <Link to="/runs/$runId" params={{ runId: run.id }} className="control-focus rounded px-2 py-1 text-xs font-semibold underline underline-offset-2">Review metadata to link</Link>
                </div>
              ))}</div>
            ) : <p className="p-5 text-sm text-muted">No unlinked runs are present in the current run index.</p>}
          </section>
        </div>
      ) : null}
    </>
  )
}

function SummaryCell({ label, value }: { label: string; value: unknown }) {
  return <div className="min-w-0"><div className="text-[11px] font-bold uppercase tracking-[0.06em] text-muted">{label}</div><div className="mt-1 break-all text-sm text-secondary">{value === null || value === undefined || value === '' ? '—' : String(value)}</div></div>
}
