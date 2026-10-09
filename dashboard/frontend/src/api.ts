import type {
  Checkpoint,
  CheckpointCompatibility,
  ActivitySnapshot,
  DashboardAssessment,
  EvaluationDetail,
  EvaluationSummary,
  CheckpointEvidence,
  BlenderRender,
  OverviewResponse,
  HealthSnapshot,
  RuntimePreflight,
  PolicyArchitecture,
  IntrospectionSchema,
  PolicyIntrospectionFrame,
  IntrospectionCapture,
  IntrospectionCaptureSummary,
  IntrospectionExplanation,
  RunDetail,
  RunIndexRow,
  SystemStatus,
  TaskOption,
  TelemetryRecord,
  ViewerState,
  WaypointCommand,
  WaypointState,
} from './types'

export async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = typeof body?.detail === 'string' ? body.detail : `${response.status} ${response.statusText}`
    throw new Error(detail)
  }
  return body as T
}

export const api = {
  controlSession: () => fetchJson<{ configured: boolean; authenticated: boolean; expires_in_seconds?: number | null }>('/api/control/session'),
  openControlSession: (token: string) => fetchJson<{ authenticated: boolean; expires_in_seconds: number }>('/api/control/session', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ token }),
  }),
  closeControlSession: () => fetchJson<{ authenticated: boolean }>('/api/control/session', { method: 'DELETE' }),
  blenderRenders: (limit = 100) => fetchJson<{ renders: BlenderRender[]; root: string }>(`/api/blender/renders?limit=${encodeURIComponent(String(limit))}`),
  overview: () => fetchJson<OverviewResponse>('/api/overview'),
  activity: () => fetchJson<ActivitySnapshot>('/api/activity'),
  assessments: () => fetchJson<{ assessed_at: number; read_only: boolean; assessments: DashboardAssessment[] }>('/api/assessments'),
  evaluationSuites: () => fetchJson<{ suites: Array<Record<string, any>>; read_only: boolean; evaluation_launch_available: boolean }>('/api/evaluation-suites'),
  evaluations: (filters: { task?: string; suite?: string; status?: string; evidence_class?: string; experiment_id?: string; checkpoint_id?: string; limit?: number; offset?: number } = {}) => {
    const query = new URLSearchParams()
    Object.entries(filters).forEach(([key, value]) => { if (value !== undefined && value !== '') query.set(key, String(value)) })
    return fetchJson<{ evaluations: EvaluationSummary[]; total: number; limit: number; offset: number }>(`/api/evaluations${query.size ? `?${query}` : ''}`)
  },
  evaluation: (evaluationId: string) => fetchJson<EvaluationDetail>(`/api/evaluations/${encodeURIComponent(evaluationId)}`),
  compareEvaluations: (baseline: string, candidate: string) => {
    const query = new URLSearchParams({ baseline, candidate })
    return fetchJson<Record<string, any>>(`/api/evaluations/compare?${query}`)
  },
  evaluationGates: (evaluationId: string) => fetchJson<{ evaluation_id: string; status: string; gates: Array<Record<string, any>> }>(`/api/evaluations/${encodeURIComponent(evaluationId)}/gates`),
  evaluationScenarios: (evaluationId: string, query = '') => fetchJson<{ total: number; scenarios: Array<Record<string, any>> }>(`/api/evaluations/${encodeURIComponent(evaluationId)}/scenarios${query ? `?${query}` : ''}`),
  experiments: () => fetchJson<{ programs: Array<Record<string, any>>; read_only: boolean }>('/api/experiments'),
  runs: () => fetchJson<{ runs: RunIndexRow[] }>('/api/runs/index'),
  run: (id: string) => fetchJson<RunDetail>(`/api/runs/${id}`),
  curriculum: (id: string) => fetchJson<{ curriculum: import('./types').Curriculum | null }>(`/api/runs/${id}/curriculum`),
  tasks: () => fetchJson<{ tasks: TaskOption[] }>('/api/tasks'),
  telemetry: (id: string, maxPoints = 800) =>
    fetchJson<{ records: TelemetryRecord[]; source_records?: number; coverage?: Record<string, unknown> }>(
      `/api/runs/${id}/telemetry?max_points=${maxPoints}`,
    ),
  logs: (id: string, tail = 400) => fetchJson<{ lines: string[] }>(`/api/runs/${id}/logs?tail=${tail}`),
  checkpoints: (id: string) =>
    fetchJson<{ checkpoints: Checkpoint[]; latest?: string | null }>(`/api/runs/${id}/checkpoints`),
  checkpointCompatibility: (runId: string, checkpoint: string, task: string) => {
    const query = new URLSearchParams({ checkpoint, task })
    return fetchJson<CheckpointCompatibility>(`/api/runs/${encodeURIComponent(runId)}/checkpoint-compatibility?${query}`)
  },
  checkpointEvidence: (runId: string, checkpoint = 'latest') =>
    fetchJson<CheckpointEvidence>(`/api/runs/${encodeURIComponent(runId)}/checkpoint-evidence?checkpoint=${encodeURIComponent(checkpoint)}`),
  architecture: (id: string) => fetchJson<PolicyArchitecture>(`/api/runs/${id}/architecture`),
  introspectionSchema: (viewerId: string) =>
    fetchJson<IntrospectionSchema | { available: false; message: string }>(
      `/api/viewers/${viewerId}/introspection/schema`,
    ),
  introspectionLatest: (viewerId: string) =>
    fetchJson<PolicyIntrospectionFrame | { available: false; message: string }>(
      `/api/viewers/${viewerId}/introspection/latest`,
    ),
  introspectionStreamUrl: (viewerId: string) => `/api/viewers/${viewerId}/introspection/stream`,
  introspectionCaptures: (viewerId: string) =>
    fetchJson<{ viewer_id: string; captures: IntrospectionCaptureSummary[] }>(
      `/api/viewers/${viewerId}/captures`,
    ),
  introspectionCapture: (viewerId: string, eventId: string) =>
    fetchJson<IntrospectionCapture>(`/api/viewers/${viewerId}/captures/${eventId}`),
  waypointState: (viewerId: string) =>
    fetchJson<WaypointState | { available: false; message: string }>(
      `/api/viewers/${viewerId}/waypoints`,
    ),
  commandWaypoint: (viewerId: string, command: WaypointCommand) =>
    fetchJson<{
      viewer_id: string
      request_id: string
      state: string
      command: WaypointCommand
    }>(`/api/viewers/${viewerId}/waypoints`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(command),
    }),
  requestIntrospectionCapture: (viewerId: string) =>
    fetchJson<{ viewer_id: string; request_id: string; state: string }>(
      `/api/viewers/${viewerId}/captures`,
      { method: 'POST' },
    ),
  requestIntrospectionExplanation: (viewerId: string, payload: Record<string, unknown>) =>
    fetchJson<{ viewer_id: string; explanation_id: string; state: string }>(
      `/api/viewers/${viewerId}/explanations`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      },
    ),
  introspectionExplanation: (viewerId: string, explanationId: string) =>
    fetchJson<IntrospectionExplanation>(
      `/api/viewers/${viewerId}/explanations/${explanationId}`,
    ),
  viewers: () => fetchJson<{ viewers: ViewerState[] }>('/api/viewers'),
  system: (refresh = false) => fetchJson<SystemStatus>(`/api/system${refresh ? '?refresh=true' : ''}`),
  health: () => fetchJson<HealthSnapshot>('/api/health'),
  runtimePreflight: (task: string, device = 'cuda:0') =>
    fetchJson<RuntimePreflight>(`/api/runtime/preflight?task=${encodeURIComponent(task)}&device=${encodeURIComponent(device)}`),
  runtimeIdentity: () => fetchJson<Record<string, any>>('/api/runtime/identity'),
  createRun: (payload: Record<string, unknown>) =>
    fetchJson<RunDetail>('/api/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  updateRun: (id: string, payload: Record<string, unknown>) =>
    fetchJson<RunDetail>(`/api/runs/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  stopRun: (id: string) =>
    fetchJson<RunDetail>(`/api/runs/${id}/stop`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason: 'user_requested' }),
    }),
  compareRuns: (ids: string[]) =>
    fetchJson<Record<string, any>>(`/api/runs/compare?run_ids=${encodeURIComponent(ids.join(','))}`),
  startViewer: (payload: { run_id: string; checkpoint: string; follow: boolean; jacobian_hz: number }) =>
    fetchJson<ViewerState>('/api/viewers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  stopViewer: (id: string) =>
    fetchJson<ViewerState>(`/api/viewers/${id}`, {
      method: 'DELETE',
      headers: {},
    }),
  updateSystem: () =>
    fetchJson<Record<string, unknown>>('/api/system/update', {
      method: 'POST',
      headers: {},
    }),
}
