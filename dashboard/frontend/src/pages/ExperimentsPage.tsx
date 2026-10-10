import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link, useParams } from '@tanstack/react-router'
import { ArrowLeft, ArrowRight, ChevronDown, FlaskConical, GitBranch, Link2Off, Search } from 'lucide-react'
import { api } from '../api'
import { PageHeader } from '../components/PageHeader'
import { StateBadge } from '../components/StateBadge'
import { Input } from '../components/ui/input'

type LinkedRun = { run: { id: string; name?: string; display_name?: string; task?: string; stage?: string; state: string; stale?: boolean }; link_source: string }
type ExperimentArm = {
  id: string
  declared_status?: string | null
  observed_outcome?: string | null
  run_attempt_count?: number | null
  run_attempts?: Array<Record<string, unknown>>
}
type Program = {
  id: string
  title?: string
  status?: string
  target_robot?: string
  source_path?: string
  source_schema_version?: number
  baseline?: { checkpoint?: string; evaluation_status?: string }
  arms: ExperimentArm[]
  linked_runs: LinkedRun[]
  unlinked_run_count: number
  unlinked_runs?: LinkedRun['run'][]
}

function presentStatus(value?: string | null) {
  if (!value) return 'Not recorded'
  return value.replaceAll('_', ' ')
}

function readableTitle(value: string) {
  return value.replace(/[_-]/g, ' ').replace(/\s+/g, ' ').trim()
}

function ArmCard({ arm }: { arm: ExperimentArm }) {
  const [open, setOpen] = useState(false)
  const attempts = arm.run_attempts || []
  return (
    <article className="rounded-lg border border-border bg-background/40 p-4 sm:p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="break-words text-sm font-semibold text-foreground">{readableTitle(arm.id)}</div>
          <div className="mt-1 font-mono text-xs text-muted">{arm.id}</div>
        </div>
        <span className="rounded-full border border-border-strong bg-raised px-2.5 py-1 text-xs text-secondary">{presentStatus(arm.declared_status)}</span>
      </div>
      <p className="mt-3 text-sm leading-relaxed text-secondary">{arm.observed_outcome ? presentStatus(arm.observed_outcome) : 'No observed result has been recorded for this arm.'}</p>
      {attempts.length ? (
        <div className="mt-4 border-t border-border pt-3">
          <button
            type="button"
            className="control-focus flex w-full items-center justify-between rounded-md py-1 text-left text-sm font-medium text-secondary hover:text-foreground"
            aria-expanded={open}
            onClick={() => setOpen(!open)}
          >
            <span>{attempts.length} recorded attempt{attempts.length === 1 ? '' : 's'}</span>
            <ChevronDown size={17} className={open ? 'rotate-180' : ''} />
          </button>
          {open ? (
            <div className="mt-3 space-y-2">
              {attempts.map((attempt, index) => {
                const runId = String(attempt.run_id || '')
                const outcome = String(attempt.outcome || attempt.status || 'Outcome unrecorded')
                return (
                  <div key={runId || index} className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-border bg-panel px-3 py-2 text-xs">
                    <div className="min-w-0">
                      <span className="font-mono text-secondary">{runId || `Attempt ${index + 1}`}</span>
                      <span className="ml-2 text-muted">{presentStatus(outcome)}</span>
                    </div>
                    {runId ? <Link to="/runs/$runId" params={{ runId }} className="control-focus rounded font-semibold text-foreground hover:underline">Open run <ArrowRight size={12} className="inline" /></Link> : null}
                  </div>
                )
              })}
            </div>
          ) : null}
        </div>
      ) : null}
    </article>
  )
}

export function ExperimentsPage() {
  const { experimentId } = useParams({ strict: false }) as { experimentId?: string }
  const query = useQuery({ queryKey: ['experiments'], queryFn: api.experiments, staleTime: 60_000, refetchInterval: 60_000 })
  const programs = (query.data?.programs || []) as Program[]
  const program = programs.find((item) => item.id === experimentId)
  const [search, setSearch] = useState('')
  const [showUnlinked, setShowUnlinked] = useState(false)
  const [unlinkedSearch, setUnlinkedSearch] = useState('')

  const filtered = useMemo(
    () => programs.filter((item) => `${item.title} ${item.id} ${item.target_robot}`.toLowerCase().includes(search.toLowerCase())),
    [programs, search],
  )
  const unlinked = useMemo(
    () => (program?.unlinked_runs || []).filter((run) => `${run.display_name} ${run.name} ${run.id}`.toLowerCase().includes(unlinkedSearch.toLowerCase())),
    [program, unlinkedSearch],
  )

  return (
    <>
      <PageHeader
        eyebrow="Research program"
        title={program ? readableTitle(program.title || program.id) : 'Experiments'}
        description="Track declared experiment arms, observed results and explicitly linked training runs. Only recorded provenance establishes lineage."
        actions={experimentId ? <Link to="/experiments" className="control-focus inline-flex min-h-10 items-center gap-2 rounded-lg border border-border-strong bg-raised px-4 text-sm font-semibold hover:bg-hover"><ArrowLeft size={15} /> All programs</Link> : null}
      />

      {query.error ? <div className="mb-5 rounded-xl border border-danger/40 bg-danger/10 p-4 text-sm text-danger" role="alert">Could not load the experiment registry: {query.error.message} <button className="ml-2 font-semibold underline" onClick={() => void query.refetch()}>Retry</button></div> : null}
      {query.isLoading ? <div className="space-y-3"><div className="h-20 animate-pulse rounded-xl border border-border bg-panel"/><div className="h-44 animate-pulse rounded-xl border border-border bg-panel" /></div> : null}

      {!experimentId && !query.isLoading && !query.error ? (
        <>
          <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
            <span className="text-sm text-secondary">{filtered.length} program{filtered.length === 1 ? '' : 's'} · read-only research registry</span>
            <label className="relative w-full sm:max-w-xs">
              <Search size={16} className="pointer-events-none absolute left-3 top-3.5 text-muted" />
              <span className="sr-only">Search programs</span>
              <Input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Find a program" className="pl-10" />
            </label>
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            {filtered.map((item) => (
              <Link key={item.id} to="/experiments/$experimentId" params={{ experimentId: item.id }} className="control-focus group block min-w-0 rounded-xl border border-border bg-panel p-5 transition-colors hover:border-border-strong hover:bg-raised sm:p-6">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.08em] text-muted"><FlaskConical size={14} /> Program plan</div>
                    <h2 className="mt-2 break-words text-lg font-semibold leading-snug">{readableTitle(item.title || item.id)}</h2>
                    <p className="mt-2 text-sm text-secondary">{item.target_robot || 'Robot not specified'}</p>
                  </div>
                  <ArrowRight size={18} className="mt-1 shrink-0 text-muted transition-transform group-hover:translate-x-1" aria-hidden />
                </div>
                <div className="mt-5 flex flex-wrap gap-x-5 gap-y-2 border-t border-border pt-4 text-sm">
                  <span><b className="text-foreground">{item.arms?.length ?? 0}</b> <span className="text-muted">arms</span></span>
                  <span><b className="text-foreground">{item.linked_runs?.length ?? 0}</b> <span className="text-muted">linked runs</span></span>
                  <span className="text-secondary">{presentStatus(item.status)}</span>
                </div>
                <p className="mt-3 break-all font-mono text-xs text-muted">{item.id}</p>
              </Link>
            ))}
            {!filtered.length ? <div className="col-span-full rounded-xl border border-border bg-panel p-8 text-center text-sm text-secondary">{programs.length ? 'No program matches this search.' : 'No approved experiment plans were found.'}</div> : null}
          </div>
        </>
      ) : null}

      {experimentId && !query.isLoading && !query.error && !program ? <div className="rounded-xl border border-danger/40 bg-danger/10 p-5 text-sm text-danger">Experiment plan not found. <Link to="/experiments" className="underline">View all programs</Link></div> : null}

      {program ? (
        <div className="space-y-6">
          <section className="grid gap-3 rounded-xl border border-border bg-panel p-5 sm:grid-cols-2 xl:grid-cols-4">
            <SummaryCell label="Declared status" value={presentStatus(program.status)} />
            <SummaryCell label="Robot" value={program.target_robot || 'Not declared'} />
            <SummaryCell label="Baseline result" value={program.baseline?.evaluation_status || 'Not recorded'} />
            <SummaryCell label="Linked runs" value={program.linked_runs.length} />
            <div className="col-span-full border-t border-border pt-3 text-xs text-muted">Source: {program.source_path} · schema v{program.source_schema_version ?? 'unknown'} · declared plan state is separate from observed evidence</div>
          </section>

          <section className="rounded-xl border border-border bg-panel p-4 sm:p-6">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div><div className="text-xs font-bold uppercase tracking-[0.08em] text-muted">Research progression</div><h2 className="mt-1 text-xl font-semibold">Experiment arms</h2><p className="mt-1 text-sm text-secondary">Results are reported from the declared plan; select an attempt to inspect the underlying run.</p></div>
              <span className="text-sm text-muted">{program.arms.length} arms</span>
            </div>
            <div className="mt-5 grid gap-3 xl:grid-cols-2">
              {program.arms.map((arm) => <ArmCard key={arm.id} arm={arm} />)}
              {!program.arms.length ? <p className="text-sm text-secondary">No separately declared experiment arms in this plan.</p> : null}
            </div>
          </section>

          <section className="overflow-hidden rounded-xl border border-border bg-panel">
            <div className="border-b border-border p-5"><div className="flex items-center gap-2"><GitBranch size={17} className="text-secondary" /><h2 className="font-semibold">Verified run links</h2></div><p className="mt-1 text-sm text-muted">Only explicit experiment IDs or tags are considered. Names are not used to infer related work.</p></div>
            {program.linked_runs.length ? <div className="divide-y divide-border">{program.linked_runs.map((item) => (
              <Link key={item.run.id} to="/runs/$runId" params={{ runId: item.run.id }} className="control-focus flex flex-wrap items-center justify-between gap-3 px-5 py-4 hover:bg-hover">
                <div className="min-w-0"><strong className="break-words text-sm">{item.run.display_name || item.run.name || item.run.id}</strong><p className="mt-1 text-xs text-muted">{item.run.task || item.run.stage || 'Unknown task'} · {item.link_source.replaceAll('_', ' ')}</p></div>
                <div className="flex shrink-0 items-center gap-2"><StateBadge state={item.run.state} stale={item.run.stale} /><ArrowRight size={15} /></div>
              </Link>
            ))}</div> : <p className="p-5 text-sm text-secondary">No run is explicitly linked to this plan yet.</p>}
          </section>

          <section className="rounded-xl border border-border bg-panel">
            <button className="control-focus flex w-full flex-wrap items-center justify-between gap-3 rounded-xl p-5 text-left hover:bg-hover" type="button" aria-expanded={showUnlinked} onClick={() => setShowUnlinked(!showUnlinked)}>
              <span className="flex items-center gap-2 font-semibold"><Link2Off size={17} /> Unassigned runs across the registry <span className="text-sm font-normal text-muted">({program.unlinked_run_count})</span></span>
              <ChevronDown size={18} className={showUnlinked ? 'rotate-180' : ''} />
            </button>
            {showUnlinked ? <div className="border-t border-border p-5">
              <p className="mb-3 text-sm text-secondary">These are globally unassigned runs, not confirmed attempts for this program. Review metadata before linking them.</p>
              <label className="block max-w-sm"><span className="sr-only">Filter unassigned runs</span><Input value={unlinkedSearch} onChange={(event) => setUnlinkedSearch(event.target.value)} placeholder="Filter by run name or ID" /></label>
              <div className="mt-3 divide-y divide-border">
                {unlinked.map((run) => <Link key={run.id} to="/runs/$runId" params={{ runId: run.id }} className="control-focus flex items-center justify-between gap-3 py-3 text-sm hover:text-foreground"><span className="min-w-0 truncate text-secondary">{run.display_name || run.name || run.id}</span><ArrowRight size={15} className="shrink-0" /></Link>)}
                {!unlinked.length ? <p className="py-3 text-sm text-muted">No matching unassigned runs.</p> : null}
              </div>
              {program.unlinked_run_count > (program.unlinked_runs?.length || 0) ? <p className="mt-3 text-xs text-muted">Showing the first {program.unlinked_runs?.length || 0} unassigned runs.</p> : null}
            </div> : null}
          </section>
        </div>
      ) : null}
    </>
  )
}

function SummaryCell({ label, value }: { label: string; value: unknown }) {
  return <div className="min-w-0"><div className="text-xs font-bold uppercase tracking-[0.06em] text-muted">{label}</div><div className="mt-1 break-words text-sm font-medium text-foreground">{value === null || value === undefined || value === '' ? '—' : String(value)}</div></div>
}
