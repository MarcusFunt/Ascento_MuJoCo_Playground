import type {
  Checkpoint,
  BlenderRender,
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

export type LivePpoEnvironment = {
  environment_id: number
  episode_step: number
  reset: boolean
  position_xyz: number[]
  quaternion_wxyz: number[]
  velocity_xy: number[]
  speed_mps: number
  target_xy: number[] | null
  target_distance_m: number | null
  closing_speed_mps: number | null
  speed_command_mps: number | null
  tilt_deg: number
  root_height_m: number
  joint_positions_rad: number[]
  wheel_contacts: { left: boolean | null; right: boolean | null }
  step_reward: number
  reward_terms_rate: Record<string, number>
}
export type LivePpoFrame = {
  available: boolean
  source?: string
  schema_version?: number
  sampled_at: number
  sequence: number
  global_step: number
  capture_overhead_ms: number
  dropped_display_frames?: number
  environments: LivePpoEnvironment[]
}

export type ReplayEntry = {
  name: string
  source: string | null
  rrd_url: string | null
  updated_at: number
}

export type DeepCheckpoint = {
  id: string; run: string; run_label: string; lineage: string; iteration: number
  path: string; modified_at: number; size_bytes: number; recording_count: number
}
export type DeepRecording = {
  id: string; scenario: string | null; task: string | null
  frames: number; updated_at: number; rrd_url: string | null
}
export type DeepLiveSource = {
  id: string; root: string; label: string; lineage: string
  iteration: number | null; sequence: number | null
  sampled_at: number | null; age_seconds: number | null
  state: 'streaming' | 'stale' | 'unavailable'
  available: boolean; reason: string | null
}
export type DeepLiveStatus = {
  available: false; state: 'stale' | 'unavailable'
  reason?: string; age_seconds?: number
}

export type LiveTrainingRun = {
  id: string; root: string; label: string; state: string; sampled_at: number | null
  iteration: number | null; mean_reward: number | null
  reward_trend: Array<{ iteration: number; reward: number }>
  curriculum_stage: number | null; curriculum_stage_count: number | null
  training_remaining_seconds: number | null; deadline_remaining_seconds: number | null
  training_cutoff: string | null; deadline: string | null; metadata_available: boolean
}
export type LiveTrainingSummary = {
  runs: LiveTrainingRun[]
  gpu: { utilization_percent: number; memory_used_mib: number; memory_total_mib: number; temperature_c: number } | null
}

export type DeepCatalog = {
  checkpoints: DeepCheckpoint[]
  recordings: Record<string, DeepRecording[]>
  live_runs: DeepLiveSource[]
  recent_live_sources: DeepLiveSource[]
  note: string
}
export type DeepStats = {
  count: number; mean?: number | null; p50?: number | null; p95?: number | null
  min?: number | null; max?: number | null
}
export type DeepEpisode = {
  checkpoint_id: string; recording_id: string
  scenario: string | null; task: string | null
  checkpoint_sha256: string | null; suite_sha256: string | null
  frames: number; sampled_frames: number; duration_s: number
  time: (number | null)[]
  channels: Record<string, (number | null)[] | (number | null)[][]>
  reward_terms: Record<string, (number | null)[]>
  stats: Record<string, DeepStats>
  has_evaluator_signals: boolean; coverage_note: string
}

export const api = {
  deepCatalog: () => fetchJson<DeepCatalog>('/api/deep-analysis/catalog'),
  activeTrainingSummary: () => fetchJson<LiveTrainingSummary>('/api/deep-analysis/active-summary'),
  deepLiveLatest: (sourceId: string) =>
    fetchJson<LivePpoFrame | DeepLiveStatus>('/api/deep-analysis/live/' + encodeURIComponent(sourceId)),
  deepLiveStreamUrl: (sourceId: string) =>
    '/api/deep-analysis/live/' + encodeURIComponent(sourceId) + '/stream',
  deepEpisode: (checkpointId: string, recordingId: string) =>
    fetchJson<DeepEpisode>('/api/deep-analysis/episode?' + new URLSearchParams({
      checkpoint_id: checkpointId, recording_id: recordingId,
    }).toString()),
  livePpoLatest: (runId: string) =>
    fetchJson<LivePpoFrame>('/api/runs/' + encodeURIComponent(runId) + '/live-monitor'),
  livePpoStreamUrl: (runId: string) =>
    '/api/runs/' + encodeURIComponent(runId) + '/live-monitor/stream',
  replays: () => fetchJson<{ replays: ReplayEntry[] }>('/api/replays'),
  exportReplay: (source: string) =>
    fetchJson<{ name: string; rrd_url: string; bytes: number }>(
      '/api/replays/export/' + source.split('/').map(encodeURIComponent).join('/'),
      { method: 'POST', headers: { 'X-Ascento-Control': '1' } },
    ),
  blenderRenders: () => fetchJson<{ renders: BlenderRender[]; root: string }>(`/api/blender/renders`),
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
      headers: { 'Content-Type': 'application/json', 'X-Ascento-Control': '1' },
      body: JSON.stringify(command),
    }),
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
