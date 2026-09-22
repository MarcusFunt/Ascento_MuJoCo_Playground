import type {
  Checkpoint,
  OverviewResponse,
  PolicyArchitecture,
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
  startViewer: (payload: { run_id: string; checkpoint: string; follow: boolean }) =>
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
