import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link, useParams } from '@tanstack/react-router'
import { ArrowLeft, ArrowRight, ShieldAlert } from 'lucide-react'
import { api } from '../api'
import { PageHeader } from '../components/PageHeader'
import { Badge } from '../components/ui/badge'
import { Input, SelectInput } from '../components/ui/input'
import { shortCommit } from '../lib/utils'
import type { EvaluationDetail, EvaluationSummary } from '../types'

const STATUS_TONE: Record<string, string> = {
  PASS: 'border-success/40 bg-success/10 text-success',
  FAIL: 'border-danger/40 bg-danger/10 text-danger',
  INVALID: 'border-danger/40 bg-danger/10 text-danger',
  INCOMPLETE: 'border-warning/40 bg-warning/10 text-warning',
  DIAGNOSTIC_ONLY: 'border-border-strong bg-raised text-secondary',
}

const TASK_METRICS: Record<string, { label: string; unit: string }> = {
  dual_wheel_contact_fraction: { label: 'Both wheels supported', unit: '%' },
  left_only_contact_fraction: { label: 'Left-only wheel contact', unit: '%' },
  right_only_contact_fraction: { label: 'Right-only wheel contact', unit: '%' },
  max_continuous_single_wheel_support_s: { label: 'Longest single-wheel support', unit: 's' },
  wheel_contact_transition_count: { label: 'Wheel-contact transitions', unit: 'count' },
  left_right_leg_pose_asymmetry_rms_rad: { label: 'Left/right leg pose asymmetry (RMS)', unit: 'rad' },
  left_right_leg_pose_asymmetry_p95_rad: { label: 'Left/right leg pose asymmetry (p95)', unit: 'rad' },
  leg_target_offset_rms_rad: { label: 'Leg target offset (RMS)', unit: 'rad' },
  leg_target_rate_rms_rad_s: { label: 'Leg target rate (RMS)', unit: 'rad/s' },
  speed_tracking_rmse_fraction: { label: 'Speed tracking RMSE', unit: 'fraction of cap' },
  speed_tracking_mae_mps: { label: 'Speed tracking MAE', unit: 'm/s' },
  physical_saturation_fraction: { label: 'Physical action saturation', unit: '%' },
  action_clip_fraction: { label: 'Policy action clipping', unit: '%' },
}

export function EvaluationsPage() {
  const { evaluationId } = useParams({ strict: false }) as { evaluationId?: string }
  const [task, setTask] = useState('')
  const [suite, setSuite] = useState('')
  const [status, setStatus] = useState('')
  const [evidenceClass, setEvidenceClass] = useState('')
  const [metric, setMetric] = useState('')
  const [baselineId, setBaselineId] = useState('')
  const [candidateId, setCandidateId] = useState('')
  const [offset, setOffset] = useState(0)
  const pageSize = 100
  const suites = useQuery({ queryKey: ['evaluation-suites'], queryFn: api.evaluationSuites, staleTime: 60_000 })
  const list = useQuery({
    queryKey: ['evaluations', task, suite, status, evidenceClass, offset],
    queryFn: () => api.evaluations({ task, suite, status, evidence_class: evidenceClass, limit: pageSize, offset }),
    enabled: !evaluationId,
    refetchInterval: 30_000,
  })
  const detail = useQuery({
    queryKey: ['evaluation', evaluationId],
    queryFn: () => api.evaluation(evaluationId!),
    enabled: Boolean(evaluationId),
    staleTime: 60_000,
  })
  const scenarios = useQuery({
    queryKey: ['evaluation-scenarios', evaluationId, metric],
    queryFn: () => api.evaluationScenarios(evaluationId!, metric ? `metric=${encodeURIComponent(metric)}&direction=high&limit=100` : 'limit=100'),
    enabled: Boolean(evaluationId),
    staleTime: 60_000,
  })
  const taskOptions = useMemo(
    () => [...new Set((suites.data?.suites || []).map((item) => String(item.task || '')).filter(Boolean))],
    [suites.data],
  )
  const comparison = useQuery({
    queryKey: ['evaluation-comparison', baselineId, candidateId],
    queryFn: () => api.compareEvaluations(baselineId, candidateId),
    enabled: Boolean(baselineId && candidateId && baselineId !== candidateId && !evaluationId),
    retry: false,
  })

  if (evaluationId) {
    return (
      <EvaluationDetailPage
        evaluationId={evaluationId}
        detail={detail.data}
        loading={detail.isLoading}
        error={detail.error?.message}
        scenarios={scenarios.data?.scenarios || []}
        scenarioTotal={scenarios.data?.total}
        scenarioError={scenarios.error?.message}
        metric={metric}
        setMetric={setMetric}
      />
    )
  }

  return (
    <>
      <PageHeader
        eyebrow="Evidence registry"
        title="Evaluations"
        description="Inspect immutable suite results, hard gates, scenario failures and provenance. Passing is not the same as being selected or improved."
      />

      <section className="mb-6 flex gap-3 rounded-xl border border-warning/35 bg-warning/5 p-4" role="note">
        <ShieldAlert size={18} className="mt-0.5 shrink-0 text-warning" />
        <p className="text-sm leading-relaxed text-secondary">This page is read-only. Promotion and held-out evaluation are never triggered automatically from routine checkpoint browsing.</p>
      </section>

      <section className="mb-5 grid gap-3 rounded-xl border border-border bg-panel p-4 sm:grid-cols-2 xl:grid-cols-4">
        <label className="text-xs font-bold uppercase tracking-[0.06em] text-muted">Task
          <SelectInput className="mt-2 w-full" value={task} onChange={(event) => { setTask(event.target.value); setOffset(0) }}>
            <option value="">All tasks</option>
            {taskOptions.map((item) => <option key={item}>{item}</option>)}
          </SelectInput>
        </label>
        <label className="text-xs font-bold uppercase tracking-[0.06em] text-muted">Suite
          <SelectInput className="mt-2 w-full" value={suite} onChange={(event) => { setSuite(event.target.value); setOffset(0) }}>
            <option value="">All suites</option>
            {(suites.data?.suites || []).map((item) => <option key={item.suite_id} value={item.suite_id}>{item.suite_id}</option>)}
          </SelectInput>
        </label>
        <label className="text-xs font-bold uppercase tracking-[0.06em] text-muted">Result
          <SelectInput className="mt-2 w-full" value={status} onChange={(event) => { setStatus(event.target.value); setOffset(0) }}>
            <option value="">All results</option>
            {['PASS', 'FAIL', 'INCOMPLETE', 'INVALID', 'DIAGNOSTIC_ONLY'].map((item) => <option key={item}>{item}</option>)}
          </SelectInput>
        </label>
        <label className="text-xs font-bold uppercase tracking-[0.06em] text-muted">Evidence
          <SelectInput className="mt-2 w-full" value={evidenceClass} onChange={(event) => { setEvidenceClass(event.target.value); setOffset(0) }}>
            <option value="">All evidence</option>
            <option value="quantitative">Quantitative</option>
            <option value="diagnostic_only">Diagnostic only</option>
          </SelectInput>
        </label>
      </section>

      <section className="mb-6 rounded-xl border border-border bg-panel p-5" aria-label="Paired evaluation comparison">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="font-semibold">Paired comparison</h2>
            <p className="mt-1 max-w-3xl text-sm text-muted">Select a baseline and candidate. Only complete artifacts with identical suite, resolved scenarios, task ABI and compatible checkpoint contracts can be compared.</p>
          </div>
          <span className="rounded border border-border-strong bg-raised px-2 py-1 text-[10px] font-bold uppercase text-secondary">Candidate − baseline</span>
        </div>
        <div className="mt-4 grid gap-3 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] md:items-end">
          <label className="text-xs font-bold uppercase tracking-[0.06em] text-muted">Baseline
            <SelectInput className="mt-2 w-full" value={baselineId} onChange={(event) => setBaselineId(event.target.value)}>
              <option value="">Choose baseline</option>
              {(list.data?.evaluations || []).map((item) => <option key={item.evaluation_id} value={item.evaluation_id}>{item.suite_id} · {item.status} · {item.evaluation_id}</option>)}
            </SelectInput>
          </label>
          <label className="text-xs font-bold uppercase tracking-[0.06em] text-muted">Candidate
            <SelectInput className="mt-2 w-full" value={candidateId} onChange={(event) => setCandidateId(event.target.value)}>
              <option value="">Choose candidate</option>
              {(list.data?.evaluations || []).map((item) => <option key={item.evaluation_id} value={item.evaluation_id}>{item.suite_id} · {item.status} · {item.evaluation_id}</option>)}
            </SelectInput>
          </label>
          <span className="min-h-10 text-sm text-muted">{baselineId && candidateId ? 'Comparison reads saved evaluation artifacts only.' : 'Choose two saved evaluations.'}</span>
        </div>
        {comparison.error ? <p className="mt-4 rounded-lg border border-warning/35 bg-warning/5 p-3 text-sm text-warning" role="alert">Comparison unavailable: {comparison.error.message}</p> : null}
        {comparison.data ? (
          <div className="mt-4 overflow-x-auto rounded-lg border border-border">
            <div className="grid min-w-[700px] grid-cols-[1fr_auto_auto_auto] gap-4 border-b border-border bg-background/40 px-4 py-3 text-xs font-bold uppercase tracking-[0.06em] text-muted">
              <span>Metric</span><span>Mean Δ</span><span>Median Δ</span><span>IQM Δ · 95% CI</span>
            </div>
            <div className="divide-y divide-border">
              {Object.entries(comparison.data.comparison?.metrics || {}).map(([name, value]: [string, any]) => (
                <div key={name} className="grid min-w-[700px] grid-cols-[1fr_auto_auto_auto] gap-4 px-4 py-3 text-sm">
                  <strong>{name}</strong><span className="numeric">{Number(value.mean_delta).toPrecision(4)}</span><span className="numeric">{Number(value.median_delta).toPrecision(4)}</span>
                  <span className="numeric">{Number(value.iqm_delta).toPrecision(4)} [{Number(value.iqm_ci95_low).toPrecision(4)}, {Number(value.iqm_ci95_high).toPrecision(4)}]</span>
                </div>
              ))}
              {!Object.keys(comparison.data.comparison?.metrics || {}).length ? <p className="p-4 text-sm text-muted">No paired metrics were available.</p> : null}
            </div>
            <p className="border-t border-border px-4 py-3 text-xs text-secondary">{comparison.data.comparison?.paired_scenarios ?? 0} paired scenario(s). Positive values mean the candidate scored higher. This result does not trigger selection or held-out evaluation.</p>
          </div>
        ) : null}
        {comparison.isFetching ? <p className="mt-3 text-sm text-muted">Checking compatibility and calculating paired deltas…</p> : null}
      </section>

      {list.error ? <div className="mb-5 rounded-xl border border-danger/40 bg-danger/10 p-4 text-sm text-danger">{list.error.message}</div> : null}
      <section className="overflow-hidden rounded-xl border border-border bg-panel" aria-label="Evaluation results">
        <div className="flex items-center justify-between gap-3 border-b border-border px-5 py-4">
          <h2 className="font-semibold">Results</h2>
          <span className="text-sm text-muted">{list.data?.total ?? '—'} evaluation(s)</span>
        </div>
        {list.isLoading ? <p className="p-6 text-sm text-muted">Loading evaluation index…</p> : null}
        {!list.isLoading && !list.error && !list.data?.evaluations.length ? (
          <p className="p-8 text-center text-sm text-muted">No evaluation artifacts match these filters. Partial artifacts remain marked incomplete.</p>
        ) : null}
        <div className="divide-y divide-border">
          {(list.data?.evaluations || []).map((item) => <EvaluationRow key={item.evaluation_id} evaluation={item} />)}
        </div>
        <div className="flex items-center justify-between gap-3 border-t border-border px-5 py-3 text-sm">
          <span className="text-muted">Showing {list.data?.total ? offset + 1 : 0}–{Math.min(offset + pageSize, list.data?.total || 0)} of {list.data?.total ?? '—'}</span>
          <div className="flex gap-2"><button className="control-focus rounded-md border border-border-strong px-3 py-1.5 disabled:opacity-40" disabled={offset <= 0 || list.isFetching} onClick={() => setOffset(Math.max(0, offset - pageSize))}>Previous</button><button className="control-focus rounded-md border border-border-strong px-3 py-1.5 disabled:opacity-40" disabled={offset + pageSize >= (list.data?.total || 0) || list.isFetching} onClick={() => setOffset(offset + pageSize)}>Next</button></div>
        </div>
      </section>
    </>
  )
}

function EvaluationRow({ evaluation }: { evaluation: EvaluationSummary }) {
  return (
    <Link
      to="/evaluations/$evaluationId"
      params={{ evaluationId: evaluation.evaluation_id }}
      className="control-focus grid gap-3 px-5 py-4 transition-colors hover:bg-hover md:grid-cols-[minmax(0,1fr)_auto_auto] md:items-center"
    >
      <div className="min-w-0">
        <div className="truncate font-semibold">{evaluation.suite_id || 'Unknown suite'}</div>
        <div className="mt-1 truncate text-xs text-muted">{evaluation.task || 'Unknown task'} · {evaluation.scenario_count ?? '—'} scenarios · checkpoint {shortCommit(evaluation.checkpoint_sha256)}</div>
        <div className="mt-1 truncate font-mono text-[11px] text-muted">{evaluation.evaluation_id}</div>
      </div>
      <span className="text-sm text-secondary">Hard gates {evaluation.hard_gates_passed}/{evaluation.hard_gates_total}</span>
      <span className="flex items-center gap-2">
        <Badge className={STATUS_TONE[evaluation.status] || ''}>{evaluation.status}</Badge>
        <ArrowRight size={15} className="text-muted" />
      </span>
    </Link>
  )
}

function EvaluationDetailPage({
  evaluationId, detail, loading, error, scenarios, scenarioTotal, scenarioError, metric, setMetric,
}: {
  evaluationId: string
  detail?: EvaluationDetail
  loading: boolean
  error?: string
  scenarios: Array<Record<string, any>>
  scenarioTotal?: number
  scenarioError?: string
  metric: string
  setMetric: (value: string) => void
}) {
  if (loading) return <div className="h-96 animate-pulse rounded-xl border border-border bg-panel" />
  if (error || !detail) return <div className="rounded-xl border border-danger/40 bg-danger/10 p-5 text-sm text-danger">{error || 'Evaluation not found.'}</div>
  const evaluation = detail.evaluation
  const gateRows = detail.gate?.gates || []

  return (
    <>
      <PageHeader
        eyebrow="Evaluation evidence"
        title={evaluation.suite_id || 'Evaluation'}
        description={`${evaluation.task || 'Unknown task'} · ${evaluation.scenario_count ?? '—'} scenarios · ${evaluation.evaluation_id}`}
        actions={<Link to="/evaluations" className="control-focus inline-flex h-10 items-center gap-2 rounded-lg border border-border-strong bg-raised px-4 text-sm font-semibold"><ArrowLeft size={15} /> All evaluations</Link>}
      />

      <section className="mb-6 grid gap-4 rounded-xl border border-border bg-panel p-5 sm:grid-cols-2 lg:grid-cols-4">
        <EvidenceValue label="Verdict" value={<Badge className={STATUS_TONE[evaluation.status] || ''}>{evaluation.status}</Badge>} />
        <EvidenceValue label="Evidence class" value={evaluation.evidence_class === 'diagnostic_only' ? 'Diagnostic only · no gates' : 'Quantitative gated evaluation'} />
        <EvidenceValue label="Hard gates passed" value={`${evaluation.hard_gates_passed}/${evaluation.hard_gates_total}`} />
        <EvidenceValue label="Checkpoint SHA-256" value={evaluation.checkpoint_sha256 || 'Unavailable'} mono />
      </section>

      {evaluation.integrity_error ? <div className="mb-5 rounded-xl border border-danger/40 bg-danger/10 p-4 text-sm text-danger">Artifact integrity error: {evaluation.integrity_error}</div> : null}
      {evaluation.status === 'INVALID' ? <div className="mb-5 rounded-xl border border-danger/40 bg-danger/10 p-4 text-sm text-danger">This result is invalid. Numeric policy-quality claims are blocked until the evaluator inconsistency is resolved.</div> : null}
      {evaluation.status === 'INCOMPLETE' ? <div className="mb-5 rounded-xl border border-warning/35 bg-warning/5 p-4 text-sm text-warning">Required result artifacts are missing or unfinished. This is not a policy failure.</div> : null}
      {evaluation.status === 'DIAGNOSTIC_ONLY' ? <div className="mb-5 rounded-xl border border-border-strong bg-raised p-4 text-sm text-secondary">This suite has no hard gates. The result is diagnostic and cannot be treated as a policy pass.</div> : null}
      {detail.manifest?.checkpoint_task_compatibility_detail?.compatible === false ? <div className="mb-5 rounded-xl border border-danger/40 bg-danger/10 p-4 text-sm text-danger">The saved checkpoint is incompatible with this task contract. This evaluation must not be used to select or promote the checkpoint.</div> : null}

      <div className="grid gap-6 xl:grid-cols-[1fr_0.9fr]">
        <section className="overflow-hidden rounded-xl border border-border bg-panel">
          <div className="border-b border-border px-5 py-4"><h2 className="font-semibold">Gate matrix</h2></div>
          {gateRows.length ? (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[700px] text-left text-sm">
                <thead className="bg-background/40 text-xs uppercase tracking-[0.06em] text-muted"><tr><th className="px-4 py-3">Gate</th><th className="px-4 py-3">Observed</th><th className="px-4 py-3">Requirement</th><th className="px-4 py-3">Type</th><th className="px-4 py-3">Result</th></tr></thead>
                <tbody className="divide-y divide-border">{gateRows.map((gate) => (
                  <tr key={String(gate.gate_id)}>
                    <td className="px-4 py-3"><strong>{String(gate.gate_id || 'Unnamed gate')}</strong><span className="mt-1 block text-xs text-muted">{String(gate.family || 'all families')} · {String(gate.statistic || 'statistic')}</span></td>
                    <td className="px-4 py-3 numeric">{gate.observed === null || gate.observed === undefined ? '—' : String(gate.observed)}</td>
                    <td className="px-4 py-3 font-mono">{String(gate.op || '')} {String(gate.threshold ?? '—')}</td>
                    <td className="px-4 py-3">{gate.hard ? 'Hard' : 'Soft'}</td>
                    <td className="px-4 py-3">
                      {(() => {
                        const observationMissing = gate.observed === null || gate.observed === undefined || typeof gate.passed !== 'boolean'
                        const gateStatus = evaluation.status === 'INVALID' ? 'INVALID' : observationMissing ? 'INCOMPLETE' : gate.passed ? 'PASS' : 'FAIL'
                        return <Badge className={STATUS_TONE[gateStatus]}>{gateStatus}</Badge>
                      })()}
                    </td>
                  </tr>
                ) )}</tbody>
              </table>
            </div>
          ) : <p className="p-6 text-sm text-muted">No gate results are available for this artifact.</p>}
        </section>

        <section className="rounded-xl border border-border bg-panel p-5">
          <h2 className="font-semibold">Reproducibility and contracts</h2>
          <dl className="mt-4 divide-y divide-border text-sm">
            <EvidenceRow label="Suite SHA-256" value={evaluation.suite_sha256} />
            <EvidenceRow label="Resolved scenarios SHA-256" value={evaluation.resolved_scenarios_sha256} />
            <EvidenceRow label="Source commit" value={detail.manifest?.repository_commit} />
            <EvidenceRow label="Checkpoint path" value={evaluation.checkpoint} />
            <EvidenceRow label="Plant contract" value={detail.manifest?.plant_contract?.version || detail.manifest?.checkpoint_plant_compatibility} />
            <EvidenceRow label="Action contract" value={detail.manifest?.action_contract?.version || detail.manifest?.checkpoint_action_compatibility} />
            <EvidenceRow label="Task compatibility" value={detail.manifest?.checkpoint_task_compatibility} />
            <EvidenceRow label="Task compatibility detail" value={detail.manifest?.checkpoint_task_compatibility_detail?.reason || detail.manifest?.checkpoint_task_compatibility_detail?.status} />
            <EvidenceRow label="Speed-command cap" value={detail.manifest?.speed_command?.max_speed_mps === undefined ? null : `${detail.manifest.speed_command.max_speed_mps} m/s`} />
            <EvidenceRow label="Speed-command slew limit" value={detail.manifest?.speed_command?.max_speed_slew_rate_mps_per_s === undefined ? null : `${detail.manifest.speed_command.max_speed_slew_rate_mps_per_s} m/s²`} />
            <EvidenceRow label="Finished" value={evaluation.finished_at_utc} />
          </dl>
          {detail.consistency?.passed === false ? <p className="mt-4 text-sm text-danger">Consistency check failed: {(detail.consistency.checks || []).map((item) => item.reason).filter(Boolean).join('; ') || 'see consistency artifact'}</p> : null}
        </section>
      </div>

      <TaskDiagnostics summary={detail.summary} task={evaluation.task} />

      <section className="mt-6 overflow-hidden rounded-xl border border-border bg-panel">
        <div className="flex flex-wrap items-end justify-between gap-3 border-b border-border px-5 py-4">
          <div><h2 className="font-semibold">Scenario breakdown</h2><p className="mt-1 text-xs text-muted">Failed scenarios appear first. Add a metric to sort by the largest value.</p></div>
          <label className="w-full max-w-xs text-xs font-bold uppercase tracking-[0.06em] text-muted">Metric for worst cases<Input className="mt-2" value={metric} onChange={(event) => setMetric(event.target.value)} placeholder="e.g. target_error_m" /></label>
        </div>
        {scenarioError ? <p className="p-5 text-sm text-warning">Scenario results unavailable: {scenarioError}</p> : null}
        {scenarios.length ? (
          <div className="divide-y divide-border">
            {scenarios.map((scenario) => (
              <div key={scenario.scenario_id} className="grid gap-2 px-5 py-3 text-sm sm:grid-cols-[1fr_auto_auto_auto] sm:items-center">
                <div><strong>{String(scenario.scenario_id)}</strong><span className="ml-2 text-xs text-muted">{String(scenario.family)}</span></div>
                <span className={scenario.success ? 'text-success' : 'text-danger'}>{scenario.success ? 'Succeeded' : String(scenario.termination_reason || 'Failed')}</span>
                <span className="numeric text-secondary">{scenario.metric ? `${scenario.metric}: ${scenario.metric_value ?? '—'}` : `${scenario.episode_steps ?? '—'} steps`}</span>
                <span className="text-xs text-muted">{scenario.spec?.tags?.join?.(', ') || 'scenario artifact'}</span>
              </div>
            ))}
          </div>
        ) : !scenarioError ? <p className="p-5 text-sm text-muted">No scenario results are available. {scenarioTotal === 0 ? 'The evaluation may be incomplete.' : ''}</p> : null}
      </section>
    </>
  )
}

function TaskDiagnostics({ summary, task }: { summary?: Record<string, any> | null; task?: string }) {
  const families = Object.entries(summary || {}).filter((entry): entry is [string, Record<string, any>] => Boolean(entry[1]) && typeof entry[1] === 'object' && !Array.isArray(entry[1]))
  const rows = families.flatMap(([family, metrics]) => Object.entries(TASK_METRICS).flatMap(([key, descriptor]) => {
    const value = metrics[key]
    if (!value || typeof value !== 'object') return []
    const mean = value.mean
    const p95 = value.p95
    const count = value.count
    if (mean === null || mean === undefined || !Number.isFinite(Number(mean))) return []
    return [{ family, key, label: descriptor.label, unit: descriptor.unit, mean: Number(mean), p95: p95 === null || p95 === undefined || !Number.isFinite(Number(p95)) ? null : Number(p95), count: count === null || count === undefined ? null : Number(count) }]
  }))
  return <section className="mt-6 overflow-hidden rounded-xl border border-border bg-panel">
    <div className="border-b border-border px-5 py-4"><h2 className="font-semibold">Task diagnostics</h2><p className="mt-1 text-xs text-muted">Measured per evaluation family. Morphology and speed signals provide context; they do not replace hard gates.</p></div>
    {rows.length ? <div className="overflow-x-auto"><table className="w-full min-w-[680px] text-left text-sm"><thead className="bg-background/40 text-xs uppercase tracking-[0.06em] text-muted"><tr><th className="px-4 py-3">Family</th><th className="px-4 py-3">Measure</th><th className="px-4 py-3">Mean</th><th className="px-4 py-3">P95</th><th className="px-4 py-3">Samples</th></tr></thead><tbody className="divide-y divide-border">{rows.map((row) => <tr key={`${row.family}-${row.key}`}><td className="px-4 py-3">{row.family}</td><td className="px-4 py-3">{row.label}</td><td className="numeric px-4 py-3">{row.unit === '%' ? `${(row.mean * 100).toFixed(1)}%` : `${row.mean.toFixed(3)} ${row.unit}`}</td><td className="numeric px-4 py-3">{row.p95 === null ? '—' : row.unit === '%' ? `${(row.p95 * 100).toFixed(1)}%` : `${row.p95.toFixed(3)} ${row.unit}`}</td><td className="numeric px-4 py-3">{row.count ?? '—'}</td></tr>)}</tbody></table></div> : <p className="p-5 text-sm text-muted">{task?.includes('Speed') || task?.includes('Locomotion') ? 'This artifact does not include the morphology or speed summary metrics needed for these diagnostics.' : 'No task-specific morphology or speed diagnostics are present in this summary.'}</p>}
  </section>
}

function EvidenceValue({ label, value, mono = false }: { label: string; value: React.ReactNode; mono?: boolean }) {
  return <div className="min-w-0"><div className="text-xs font-bold uppercase tracking-[0.06em] text-muted">{label}</div><div className={`mt-2 break-all text-sm text-secondary ${mono ? 'font-mono text-xs' : ''}`}>{value}</div></div>
}

function EvidenceRow({ label, value }: { label: string; value: unknown }) {
  return <div className="grid grid-cols-[minmax(130px,0.8fr)_minmax(0,1fr)] gap-4 py-2.5"><dt className="text-muted">{label}</dt><dd className="m-0 break-all text-right text-secondary">{value === null || value === undefined || value === '' ? '—' : String(value)}</dd></div>
}
