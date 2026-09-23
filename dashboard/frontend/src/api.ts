import type {
  Checkpoint,
  OverviewResponse,
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
  overview: () => fetchJson<OverviewResponse>('/api/overview'),
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
  requestIntrospectionCapture: (viewerId: string) =>
    fetchJson<{ viewer_id: string; request_id: string; state: string }>(
      `/api/viewers/${viewerId}/captures`,
      { method: 'POST', headers: { 'X-Ascento-Control': '1' } },
    ),
  requestIntrospectionExplanation: (viewerId: string, payload: Record<string, unknown>) =>
    fetchJson<{ viewer_id: string; explanation_id: string; state: string }>(
      `/api/viewers/${viewerId}/explanations`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Ascento-Control': '1' },
        body: JSON.stringify(payload),
      },
    ),
  introspectionExplanation: (viewerId: string, explanationId: string) =>
    fetchJson<IntrospectionExplanation>(
      `/api/viewers/${viewerId}/explanations/${explanationId}`,
    ),
  viewers: () => fetchJson<{ viewers: ViewerState[] }>('/api/viewers'),
  system: (refresh = false) => fetchJson<SystemStatus>(`/api/system${refresh ? '?refresh=true' : ''}`),
  health: () => fetchJson<Record<string, any>>('/api/health'),
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
      headers: { 'Content-Type': 'application/json', 'X-Ascento-Control': '1' },
      body: JSON.stringify(payload),
    }),
  stopViewer: (id: string) =>
    fetchJson<ViewerState>(`/api/viewers/${id}`, {
      method: 'DELETE',
      headers: { 'X-Ascento-Control': '1' },
    }),
  updateSystem: () =>
    fetchJson<Record<string, unknown>>('/api/system/update', {
      method: 'POST',
      headers: { 'X-Ascento-Control': '1' },
    }),
}
