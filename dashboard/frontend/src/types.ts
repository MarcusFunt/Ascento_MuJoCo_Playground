export type RepositoryVersion = {
  status?: string | null
  is_outdated?: boolean
  run_commit?: string | null
  current_commit?: string | null
}

export type RunIndexRow = {
  id: string
  display_name: string
  name: string
  task?: string | null
  stage?: string | null
  state: string
  stale?: boolean
  freshness_seconds?: number | null
  modified_at?: number | null
  tags?: string[]
  purpose?: string
  lineage?: { parent_run_id?: string | null; parent_checkpoint?: string | null }
  repository_version?: RepositoryVersion
  iteration?: number | null
  total_iterations?: number | null
  percent_complete?: number | null
  eta_seconds?: number | null
  throughput?: number | null
  reward?: number | null
  episode_length?: number | null
  kl?: number | null
  entropy?: number | null
  ppo_loss?: number | null
  clip_fraction?: number | null
  invalid_update?: number | null
}

export type HorizonCurriculum = {
  kind: 'horizon'
  label: string
  task?: string
  stage: number
  stage_count: number
  current_horizon_s: number
  stages: Array<{ index: number; label: string; value: number; state: 'complete' | 'current' | 'upcoming' }>
  promotion: {
    qualified_windows: number
    required_windows: number
    timeout_fraction?: number | null
    timeout_threshold: number
    quality_fraction?: number | null
    quality_threshold: number
  }
  demotion: {
    failed_windows: number
    required_windows: number
    severe_timeout_threshold: number
  }
  stage_windows?: number | null
  top_horizon_windows?: number | null
  transition?: string
  protected?: boolean
  candidate_checkpoint?: string | null
  secondary?: RecoveryDifficulty
}

export type RecoveryDifficulty = {
  kind: 'recovery_difficulty'
  label: string
  progress: number
  control_steps: number
  ramp_control_steps: number
  hard_fraction: number
  hard_fraction_start: number
  hard_fraction_end: number
  pitch_max_rad: number
  linear_x_max_m_s: number
  angular_max_rad_s: number
}

export type SequenceCurriculum = {
  kind: 'sequence'
  label: string
  task?: string
  note?: string
  stages: Array<{ label: string; detail: string }>
}

export type StaticCurriculum = {
  kind: 'static'
  label: string
  task?: string
  stage?: string
  note?: string
}

export type GeneralistCurriculum = {
  kind: 'generalist'
  label: string
  task?: string
  stage?: number | null
  stage_count: number
  progress?: number | null
  target_distance_m: { minimum?: number | null; maximum?: number | null }
  goal_fractions: { short?: number | null; medium?: number | null; long?: number | null }
  gate_like_fraction?: number | null
  demotion_streak?: number | null
  gates: Array<{
    id: string
    label: string
    estimate?: number | null
    threshold: number
    successes?: number | null
    samples?: number | null
    minimum_samples: number
    grain: string
    state: 'waiting' | 'pass' | 'below_threshold'
  }>
  cohorts: Array<{
    id: string
    label: string
    episode_count?: number | null
    arrival_count?: number | null
    arrival_rate?: number | null
    recovery_count?: number | null
    settled_stop_count?: number | null
    fall_count?: number | null
    timeout_count?: number | null
    grain: string
  }>
  attempt_bands: Array<{
    id: string
    label: string
    attempts?: number | null
    arrivals?: number | null
    arrival_rate?: number | null
    settled_stops?: number | null
    settled_stop_rate?: number | null
    heading_valid_at_settle?: number | null
    fall_interruptions?: number | null
    timeout_interruptions?: number | null
    mean_final_target_error_m?: number | null
    p95_final_target_error_m?: number | null
    mean_time_to_arrival_s?: number | null
    p95_time_to_arrival_s?: number | null
    mean_time_to_settle_s?: number | null
    p95_time_to_settle_s?: number | null
    mean_heading_error_at_settle_rad?: number | null
    grain: string
  }>
  telemetry: { iteration?: number | null; wall_time?: number | null; source: string; window_label: string }
  has_metrics: boolean
  note: string
}

export type Curriculum = HorizonCurriculum | SequenceCurriculum | GeneralistCurriculum | StaticCurriculum

export type OverviewSeriesPoint = {
  iteration?: number | null
  reward?: number | null
  episode_length?: number | null
  kl?: number | null
  entropy?: number | null
  ppo_loss?: number | null
  clip_fraction?: number | null
}

export type DashboardEvent = {
  id: number
  run_id?: string | null
  type: string
  message: string
  payload?: Record<string, unknown> | null
  created_at: number
}

export type OverviewResponse = {
  active_run: (RunIndexRow & {
    started_at?: string | null
    device?: string | null
    invalid_updates?: number
    non_finite_updates?: number
    system?: Record<string, unknown>
  }) | null
  recent_run?: RunIndexRow | null
  curriculum?: Curriculum | null
  series: OverviewSeriesPoint[]
  events: DashboardEvent[]
  counts: { total: number; active: number; errors: number; outdated: number }
  database?: { enabled: boolean; available: boolean; backend?: string | null; error?: string | null }
}

export type TaskOption = {
  id: string
  label: string
  description: string
  supports_horizon: boolean
  supports_speed_command?: boolean
}

export type TelemetryRecord = {
  iteration?: number
  total_iterations?: number
  percent_complete?: number
  canonical_metrics?: Record<string, number | null>
  metrics?: Record<string, number | null>
  [key: string]: unknown
}

export type RunDetail = {
  id: string
  name: string
  display_name?: string
  state?: string
  stale?: boolean
  stage?: string
  tags?: string[]
  notes?: string
  metadata?: Record<string, unknown>
  lineage?: { parent_run_id?: string | null; parent_checkpoint?: string | null }
  telemetry?: TelemetryRecord
  training_health?: {
    latest?: Record<string, number | null>
    invalid_updates?: number
    non_finite_updates?: number
    [key: string]: unknown
  }
  run_info?: Record<string, unknown>
  repository_version?: RepositoryVersion
  plant_contract?: { status?: string }
  action_contract?: { status?: string }
  task_contract?: { status?: string; compatibility?: { reason?: string } }
  system?: Record<string, unknown>
  process?: Record<string, unknown>
  [key: string]: unknown
}

export type Checkpoint = {
  relative_path: string
  iteration?: number | null
  stable?: boolean
}

export type CheckpointCompatibility = {
  run_id: string
  task: string
  checkpoint: string
  checkpoint_sha256: string
  status: 'COMPATIBLE' | 'INCOMPATIBLE' | 'UNVERIFIABLE'
  compatible: boolean
  reason: string
}

export type PolicyLayerStats = {
  module_index: number
  input_size: number
  output_size: number
  weight_count: number
  weight_rms: number
  weight_mean_abs: number
  weight_max_abs: number
  bias_rms?: number | null
}

export type PolicyBranchArchitecture = {
  layers: number[]
  activation: string
  distribution?: string
  std_parameters?: number
  parameter_count?: number
  linear_layers?: PolicyLayerStats[]
}

export type PolicyArchitecture = {
  available: boolean
  checkpoint?: string
  iteration?: number | null
  message?: string
  actor?: PolicyBranchArchitecture
  critic?: PolicyBranchArchitecture
}

export type ViewerState = {
  id: string
  run_id: string
  task?: string
  state: string
  checkpoint?: string
  checkpoint_iteration?: number | null
  training_iteration?: number | null
  lag_iterations?: number | null
  follow?: boolean
  jacobian_hz?: number
  port?: number
  exit_code?: number | null
}

export type Waypoint = {
  id: string
  x_m: number
  y_m: number
  yaw_rad: number | null
}

export type WaypointCommand = {
  operation: 'set' | 'queue' | 'hold' | 'resume' | 'cancel' | 'speed'
  x_m?: number
  y_m?: number
  yaw_rad?: number | null
  speed_mps?: number
}

export type WaypointState = {
  available: true
  frame: 'sim_world'
  origin: { x_m: number; y_m: number }
  state: 'holding' | 'driving' | 'paused' | 'arrived'
  robot: {
    x_m: number
    y_m: number
    yaw_rad: number
    speed_m_s: number
    yaw_rate_rad_s: number
    tilt_rad: number
    both_wheels_supported: boolean
  }
  target: { x_m: number; y_m: number; yaw_rad: number }
  active: Waypoint | null
  queue: Waypoint[]
  distance_m: number
  dwell_s: number
  completed: number
  last_command: {
    request_id?: string | null
    operation?: WaypointCommand['operation']
    state?: 'accepted' | 'rejected'
    error?: string
  } | null
  limits: { arena_half_extent_m: number; max_segment_m: number; max_queued: number }
  arrival?: { distance_m: number; heading_rad: number; speed_m_s: number; yaw_rate_rad_s: number; dwell_s: number }
  speed_command?: {
    requested_mps: number
    applied_mps: number
    measured_mps: number
    max_speed_mps: number
    max_slew_rate_mps_per_s: number
    manual_override: boolean
  } | null
}

export type ObservationFeature = {
  index: number
  term: string
  dimension: number
  label: string
  unit: string | null
  source: string
  scale: number | number[] | null
  clip: [number, number] | number[] | null
}

export type ObservationTermSchema = {
  name: string
  start: number
  end: number
  shape: number[]
  source: string
  scale: number | number[] | null
  clip: [number, number] | number[] | null
}

export type ObservationSchema = {
  group: string
  input_dim: number
  terms: ObservationTermSchema[]
  features: ObservationFeature[]
}

export type IntrospectionSchema = {
  schema_version: number
  checkpoint: string
  policy_generation?: number
  actor: ObservationSchema
  critic: ObservationSchema | null
  actor_network?: PolicyBranchArchitecture | null
  critic_network?: PolicyBranchArchitecture | null
}

export type ActionTarget = {
  channel: string
  kind: 'position' | 'velocity'
  value: number
  unit: string
  joint_limit_clipped: boolean
}

export type ActionPipeline = {
  actor_output: number[]
  wrapper_clipped: number[]
  processed_action: number[]
  wrapper_clipped_flags: boolean[]
  action_term_clipped_flags: boolean[]
  targets: ActionTarget[]
}

export type PolicyIntrospectionFrame = {
  sequence_id: number
  policy_generation: number
  checkpoint: string
  captured_at: number
  raw_observation: number[]
  normalized_observation: number[]
  actor_output: number[]
  hidden_activations: Record<string, number[]>
  critic_value: number | null
  policy_std: number[] | null
  policy_std_parameter_count: number | null
  policy_std_state_dependent: boolean | null
  jacobian: number[][] | null
  jacobian_calculated_at_sequence: number | null
  jacobian_age_steps: number | null
  jacobian_age_ms: number | null
  action_pipeline: ActionPipeline | null
  transition: TransitionSnapshot | null
}

export type TransitionSnapshot = {
  episode_id: number
  episode_step: number
  sim_time_s: number
  reward_rate: number
  step_reward: number
  reward_terms: Record<string, number>
  tilt_rad: number
  tilt_rate_rad_s: number
  height_m: number
  contacts: Record<string, boolean>
  balance_margin: number
  fallen: boolean
  reset: boolean
}

export type IntrospectionCaptureSummary = {
  event_id: string
  event_type: 'fall' | 'recovery' | 'manual'
  viewer_id: string
  run_id: string
  checkpoint: string
  checkpoint_iteration: number | null
  policy_generation?: number
  trigger_sequence: number
  trigger_reason: string
  created_at: string
  artifact: string
}

export type IntrospectionCapture = IntrospectionCaptureSummary & {
  version: number
  schema: IntrospectionSchema
  frames: PolicyIntrospectionFrame[]
  markers: Array<{ sequence: number; kind: string; label: string }>
}

export type IntrospectionExplanation = {
  explanation_id: string
  status: 'pending' | 'complete' | 'error'
  event_id?: string | null
  live_sequence?: number | null
  checkpoint?: string
  action_index: number
  action_name: string
  method: 'integrated_gradients'
  baseline_kind: string
  baseline_description: string
  n_steps: number
  input_raw?: number[]
  baseline_raw?: number[]
  attribution?: number[]
  absolute_fraction?: number[]
  convergence_delta?: number
  duration_ms?: number
  error?: string
}

export type SystemStatus = {
  connected?: boolean
  error?: string
  checked_at?: number | string
  repository?: Record<string, any>
  update?: Record<string, any>
  tailscale?: Record<string, any>
  active_runs?: Array<Record<string, any>> | null
  update_blockers?: string[]
  can_update?: boolean
}

export type ServiceHealth = {
  status: 'healthy' | 'degraded' | 'unavailable' | 'optional' | 'unknown'
  message?: string
  checked_at?: number
}

export type HealthSnapshot = {
  ok: boolean
  live: boolean
  ready: boolean
  status: 'healthy' | 'degraded' | 'unavailable'
  checked_at: number
  artifact_root?: string
  components: Record<string, ServiceHealth>
  database?: { enabled: boolean; available: boolean; backend?: string | null; error?: string | null }
  problems?: string[]
}

export type RuntimePreflight = {
  status: 'ready' | 'blocked'
  allowed: boolean
  task: { id: string; label: string | null }
  requested_device: string
  runtime: Record<string, unknown> & { device?: string; source_commit?: string; source_branch?: string }
  blockers: string[]
}

export type ActivitySnapshot = {
  checked_at: number
  fresh_for_seconds: number
  trainer: { status: string; verified: boolean; message?: string; active_runs?: Array<Record<string, any>> | null; indexed_active_runs?: Array<Record<string, any>> }
  evaluator: { status: string; verified: boolean; message?: string }
  viewer: { status: string; verified: boolean; message?: string; items?: Array<Record<string, any>> | null }
  render: { status: string; verified: boolean; message?: string }
  gpu: { available: boolean; observed_at: number; gpus?: Array<Record<string, any>>; error?: string }
  index: { status: string; backend?: string | null; available: boolean; last_successful_sync_at?: number | null; source_conflicts?: number | null; indexed_run_count: number; source: string }
}

export type DashboardAssessment = {
  id: string
  rule_id: string
  severity: 'critical' | 'warning' | 'info'
  headline: string
  explanation: string
  subject_type: string
  subject_id?: string | null
  recommended_action: string
  evidence_refs: Array<{ label: string; href: string }>
  observed_at: number
  freshness_seconds: number
  source: string
}

export type EvaluationSummary = {
  evaluation_id: string
  suite_id?: string
  task?: string
  checkpoint?: string
  checkpoint_sha256?: string
  suite_sha256?: string
  resolved_scenarios_sha256?: string
  repository_commit?: string
  scenario_count?: number
  status: 'PASS' | 'FAIL' | 'INCOMPLETE' | 'INVALID' | 'DIAGNOSTIC_ONLY'
  validity: 'valid' | 'incomplete' | 'invalid'
  evidence_class: 'quantitative' | 'diagnostic_only'
  hard_gates_passed: number
  hard_gates_total: number
  hard_gates_failed: number
  hard_gates_unavailable?: number
  finished_at_utc?: string
  integrity_error?: string | null
}

export type EvaluationDetail = {
  evaluation: EvaluationSummary
  manifest?: Record<string, any> | null
  suite?: Record<string, any> | null
  summary?: Record<string, any> | null
  gate?: { status?: string; reason?: string; gates?: Array<Record<string, any>> } | null
  consistency?: { passed?: boolean; checks?: Array<Record<string, any>> } | null
  artifact_errors?: Record<string, string>
}

export type CheckpointEvidence = {
  checkpoint_id: string
  run_id: string
  relative_path: string
  iteration?: number | null
  stable: boolean
  sha256: string
  size_bytes: number
  modified_at: number
  selection_status: 'selected' | 'not_selected' | 'not_recorded'
  visual_review_status: 'not_recorded'
  evaluations: Array<{ evaluation_id: string; suite_id?: string; status: string; evidence_class: string; finished_at_utc?: string }>
}

export type BlenderRenderOutput = {
  path?: string
  url?: string | null
  sha256?: string
  size_bytes?: number
  count?: number
}

export type BlenderRender = {
  schema_version?: number
  render_id?: string
  status?: 'rendering' | 'complete' | 'failed' | string
  created_at_utc?: string
  manifest_path?: string
  manifest_url?: string | null
  inputs?: {
    source_npz?: { path?: string; sha256?: string; task?: string | null; seed?: string | null }
    policy_checkpoint?: { kind?: string; path?: string | null; sha256?: string | null; provenance?: string }
  }
  render?: {
    fps?: number
    resolution?: { width?: number; height?: number }
  }
  outputs?: {
    blend?: BlenderRenderOutput
    video?: BlenderRenderOutput | null
    preview?: BlenderRenderOutput
  }
  error?: string
}
