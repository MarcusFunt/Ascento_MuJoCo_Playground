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

export type Curriculum = HorizonCurriculum | SequenceCurriculum | StaticCurriculum

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
  active_runs?: Array<Record<string, any>>
  update_blockers?: string[]
  can_update?: boolean
}
