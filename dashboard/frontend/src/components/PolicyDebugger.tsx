import { useEffect, useState } from 'react'
import type { IntrospectionCapture, IntrospectionCaptureSummary, IntrospectionSchema, PolicyIntrospectionFrame, ViewerState } from '../types'
import { ActionPipelinePanel } from './ActionPipelinePanel'
import { FailureReplayPanel } from './FailureReplayPanel'
import { ExplanationDrawer } from './ExplanationDrawer'
import { ObservationInspector } from './ObservationInspector'
import { PolicyJacobianHeatmap } from './PolicyJacobianHeatmap'

export function PolicyDebugger({
  viewer,
  schema,
  frame,
  liveFrame,
  connected,
  schemaError,
  captures,
  capture,
  selectedCaptureId,
  selectedFrameIndex,
  selectCapture,
  selectFrame,
  onManualCapture,
  manualCapturePending,
  manualCaptureError,
  captureLoading,
}: {
  viewer: ViewerState | null
  schema: IntrospectionSchema | null
  frame: PolicyIntrospectionFrame | null
  liveFrame: PolicyIntrospectionFrame | null
  connected: boolean
  schemaError: Error | null
  captures: IntrospectionCaptureSummary[]
  capture: IntrospectionCapture | undefined
  selectedCaptureId: string | null
  selectedFrameIndex: number
  selectCapture: (eventId: string | null) => void
  selectFrame: (index: number) => void
  onManualCapture: () => void
  manualCapturePending: boolean
  manualCaptureError: Error | null
  captureLoading: boolean
}) {
  const [selectedActionIndex, setSelectedActionIndex] = useState<number | null>(null)
  const [selectedFeatureIndex, setSelectedFeatureIndex] = useState<number | null>(null)
  useEffect(() => {
    setSelectedActionIndex(null)
    setSelectedFeatureIndex(null)
  }, [frame?.checkpoint, frame?.policy_generation])
  const live = viewer?.state === 'running' && connected && !selectedCaptureId
  const captureTrigger = capture
    ? capture.frames.find((item) => item.sequence_id === capture.trigger_sequence)
    : undefined
  const replayOffset = captureTrigger?.transition && frame?.transition
    ? frame.transition.sim_time_s - captureTrigger.transition.sim_time_s
    : null

  return (
    <section className="rounded-xl border border-border bg-panel p-5 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="text-xs font-bold uppercase tracking-[0.1em] text-muted">Policy introspection</div>
          <h2 className="mt-1 text-xl font-semibold tracking-[-0.02em]">Live policy debugger</h2>
          <p className="mt-2 max-w-3xl text-sm leading-relaxed text-muted">
            Read-only viewer snapshots show the observation, deterministic action, control clipping, physical target, local sensitivity, critic value, and distribution scale from one sequence.
          </p>
        </div>
        <span className={`rounded-md border px-2.5 py-1.5 font-mono text-xs ${live ? 'border-success/40 bg-success/10 text-success' : 'border-border bg-background/50 text-secondary'}`}>
          {selectedCaptureId
            ? `REPLAY · ${capture?.event_type || 'event'}${replayOffset !== null ? ` · ${replayOffset.toFixed(2)} s` : ''}`
            : live ? '● LIVE' : frame ? viewer?.state === 'running' ? 'CONNECTING' : 'LAST FRAME' : 'WAITING'}
          {frame ? ` · seq ${frame.sequence_id}` : ''}
        </span>
      </div>

      {!viewer ? (
        <p className="mt-5 rounded-lg border border-border bg-background/35 p-4 text-sm text-secondary">
          Start the isolated policy viewer above to populate live observations and actions. The trainer is not instrumented by this debugger.
        </p>
      ) : schemaError ? (
        <p className="mt-5 rounded-lg border border-danger/35 bg-danger/10 p-4 text-sm text-danger">Could not load the live observation schema: {schemaError.message}</p>
      ) : !schema || !frame ? (
        <p className="mt-5 rounded-lg border border-border bg-background/35 p-4 text-sm text-secondary">
          {viewer.state === 'starting' ? 'Viewer process is starting; waiting for its runtime schema and first policy transition.' : 'Waiting for the viewer to publish a complete policy transition.'}
        </p>
      ) : (
        <>
          <FailureReplayPanel
            captures={captures}
            capture={capture}
            captureId={selectedCaptureId}
            frameIndex={selectedFrameIndex}
            liveFrame={liveFrame}
            loading={captureLoading}
            onSelectCapture={selectCapture}
            onSelectFrame={selectFrame}
            onManualCapture={onManualCapture}
            manualCapturePending={manualCapturePending}
          />
          {manualCaptureError ? <p className="mt-2 text-xs text-danger">Capture request failed: {manualCaptureError.message}</p> : null}
          <div className="mt-5 grid gap-px overflow-hidden rounded-lg border border-border bg-border sm:grid-cols-2">
            <div className="bg-background/35 p-4">
              <span className="block text-[10px] font-bold uppercase tracking-[0.08em] text-muted">Critic V(s)</span>
              <strong className="numeric mt-1 block text-xl">{frame.critic_value?.toFixed(5) ?? '—'}</strong>
              <span className="mt-1 block text-xs text-muted">Value estimate for the matching critic observation group.</span>
            </div>
            <div className="bg-background/35 p-4">
              <span className="block text-[10px] font-bold uppercase tracking-[0.08em] text-muted">Policy σ · Gaussian scale, not confidence</span>
              {frame.policy_std?.length ? (
                <div className="mt-2 flex flex-wrap gap-2">
                  {frame.policy_std.map((value, index) => (
                    <span key={index} className="rounded border border-border bg-panel px-2 py-1 font-mono text-xs text-secondary" title={frame.action_pipeline?.targets[index]?.channel || `action ${index}`}>
                      {frame.action_pipeline?.targets[index]?.channel || `a${index}`} σ {value.toFixed(3)}
                    </span>
                  ))}
                </div>
              ) : <strong className="numeric mt-1 block text-xl">—</strong>}
              <span className="mt-1 block text-xs text-muted">
                Actual distribution scale · {frame.policy_std_state_dependent === false ? 'state-independent' : frame.policy_std_state_dependent === true ? 'state-dependent' : 'scope unknown'}
                {frame.policy_std_parameter_count !== null ? ` · ${frame.policy_std_parameter_count} learned std parameter${frame.policy_std_parameter_count === 1 ? '' : 's'}` : ''}
              </span>
            </div>
          </div>

          {frame.transition ? (
            <div className="mt-4 rounded-lg border border-border bg-background/30 p-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="text-sm font-semibold">Matched environment transition</h3>
                <span className={`rounded border px-2 py-1 text-[10px] font-bold uppercase ${frame.transition.fallen ? 'border-danger/40 bg-danger/10 text-danger' : frame.transition.reset ? 'border-warning/40 bg-warning/10 text-warning' : 'border-success/30 bg-success/10 text-success'}`}>
                  {frame.transition.fallen ? 'FALL' : frame.transition.reset ? 'RESET / TIMEOUT' : 'CONTINUE'}
                </span>
              </div>
              <div className="mt-3 grid gap-3 sm:grid-cols-3 xl:grid-cols-6">
                {[
                  ['Reward / step', frame.transition.step_reward.toFixed(5)],
                  ['Reward rate', `${frame.transition.reward_rate.toFixed(3)}/s`],
                  ['Tilt', `${(frame.transition.tilt_rad * 180 / Math.PI).toFixed(2)}°`],
                  ['Tilt rate', `${(frame.transition.tilt_rate_rad_s * 180 / Math.PI).toFixed(1)}°/s`],
                  ['Height', `${frame.transition.height_m.toFixed(3)} m`],
                  ['Balance margin*', `${(frame.transition.balance_margin * 100).toFixed(1)}%`],
                ].map(([label, value]) => (
                  <div key={label}>
                    <span className="block text-[10px] uppercase tracking-wide text-muted">{label}</span>
                    <strong className="numeric mt-1 block text-sm">{value}</strong>
                  </div>
                ))}
              </div>
              <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-muted">
                <span>episode {frame.transition.episode_id} · step {frame.transition.episode_step}</span>
                <span>·</span>
                <span>support: {Object.entries(frame.transition.contacts).filter(([, contact]) => contact).map(([name]) => name).join(' + ') || 'none'}</span>
                <span>·</span>
                <span>sim {frame.transition.sim_time_s.toFixed(2)} s</span>
              </div>
              {Object.keys(frame.transition.reward_terms).length > 0 ? (
                <details className="mt-3">
                  <summary className="cursor-pointer text-xs font-medium text-secondary">Reward terms</summary>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {Object.entries(frame.transition.reward_terms).map(([name, value]) => (
                      <span key={name} className="rounded border border-border px-2 py-1 font-mono text-[10px] text-muted">{name} {value >= 0 ? '+' : ''}{value.toFixed(4)}/s</span>
                    ))}
                  </div>
                </details>
              ) : null}
              <p className="mt-2 text-[10px] text-muted">*Balance margin is the existing state-derived diagnostic heuristic, not policy confidence.</p>
            </div>
          ) : null}

          <div className="mt-5 space-y-5">
            <ObservationInspector
              schema={schema.actor}
              frame={frame}
              selectedActionIndex={selectedActionIndex}
              selectedFeatureIndex={selectedFeatureIndex}
            />
            <ActionPipelinePanel frame={frame} />
            <PolicyJacobianHeatmap
              schema={schema}
              frame={frame}
              selectedActionIndex={selectedActionIndex}
              selectedFeatureIndex={selectedFeatureIndex}
              onActionSelect={setSelectedActionIndex}
              onFeatureSelect={setSelectedFeatureIndex}
            />
            <ExplanationDrawer
              viewer={viewer}
              schema={schema}
              frame={frame}
              capture={capture}
              selectedCaptureId={selectedCaptureId}
              onFeatureSelect={setSelectedFeatureIndex}
            />
          </div>
          <p className="mt-4 text-[11px] text-muted">
            Checkpoint <span className="font-mono">{frame.checkpoint}</span> · sequence {frame.sequence_id} · captured {new Date(frame.captured_at * 1000).toLocaleTimeString()}
          </p>
        </>
      )}
    </section>
  )
}
