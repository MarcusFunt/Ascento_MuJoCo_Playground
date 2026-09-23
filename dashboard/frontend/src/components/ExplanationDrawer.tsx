import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { api } from '../api'
import type { IntrospectionCapture, IntrospectionExplanation, IntrospectionSchema, PolicyIntrospectionFrame, ViewerState } from '../types'

type BaselineKind = 'normalizer_mean' | 'episode_start' | 'selected_frame'

export function ExplanationDrawer({
  viewer,
  schema,
  frame,
  capture,
  selectedCaptureId,
  onFeatureSelect,
}: {
  viewer: ViewerState | null
  schema: IntrospectionSchema | null
  frame: PolicyIntrospectionFrame
  capture: IntrospectionCapture | undefined
  selectedCaptureId: string | null
  onFeatureSelect: (index: number) => void
}) {
  const [actionIndex, setActionIndex] = useState(0)
  const [baselineKind, setBaselineKind] = useState<BaselineKind>('normalizer_mean')
  const [baselineFrameSequence, setBaselineFrameSequence] = useState<number | null>(null)
  const [nSteps, setNSteps] = useState(32)
  const [explanationId, setExplanationId] = useState<string | null>(null)
  const stalePolicy = Boolean(viewer?.checkpoint && viewer.checkpoint !== frame.checkpoint)
  const replayBaselineFrames = useMemo(
    () => capture?.frames.filter((item) => item.sequence_id !== frame.sequence_id) || [],
    [capture, frame.sequence_id],
  )
  const selectedReplayBaseline = replayBaselineFrames.find((item) => item.sequence_id === baselineFrameSequence)
    || replayBaselineFrames[0]

  useEffect(() => {
    setExplanationId(null)
  }, [frame.sequence_id, frame.checkpoint, selectedCaptureId])

  const request = useMutation({
    mutationFn: () => {
      const selectedBaseline = baselineKind === 'selected_frame' ? selectedReplayBaseline : undefined
      const target = frame.action_pipeline?.targets[actionIndex]?.channel || `action_${actionIndex}`
      return api.requestIntrospectionExplanation(viewer!.id, {
        checkpoint: frame.checkpoint,
        input_raw: frame.raw_observation,
        action_index: actionIndex,
        action_name: target,
        baseline_kind: baselineKind,
        baseline_raw: selectedBaseline?.raw_observation,
        baseline_description: selectedBaseline
          ? `Replay frame sequence ${selectedBaseline.sequence_id} from the same capture.`
          : '',
        n_steps: nSteps,
        live_sequence: frame.sequence_id,
        episode_id: frame.transition?.episode_id,
        event_id: selectedCaptureId,
      })
    },
    onSuccess: (value) => setExplanationId(value.explanation_id),
  })
  const result = useQuery({
    queryKey: ['introspection-explanation', viewer?.id, explanationId],
    queryFn: () => api.introspectionExplanation(viewer!.id, explanationId!),
    enabled: Boolean(viewer?.id && explanationId),
    refetchInterval: (query) => query.state.data?.status === 'pending' ? 500 : false,
    retry: false,
  })
  const explanation = result.data as IntrospectionExplanation | undefined
  const rankedFeatures = useMemo(() => {
    if (!explanation?.attribution || !schema) return []
    return explanation.attribution
      .map((value, index) => ({
        index,
        label: schema.actor.features[index]?.label || `observation_${index}`,
        value,
        fraction: explanation.absolute_fraction?.[index] || 0,
      }))
      .sort((left, right) => Math.abs(right.value) - Math.abs(left.value))
      .slice(0, 12)
  }, [explanation, schema])

  return (
    <details className="rounded-lg border border-border bg-background/30 p-4">
      <summary className="cursor-pointer text-sm font-semibold">Explain this action · Integrated Gradients</summary>
      <p className="mt-2 text-xs leading-relaxed text-muted">
        On-demand gradients run only in the isolated viewer, never in PPO training. Attribution is relative to the explicitly selected baseline; it is not causal proof or policy confidence.
      </p>
      <div className="mt-3 flex flex-wrap items-end gap-3">
        <label className="grid gap-1 text-xs text-secondary">
          Action output
          <select className="min-w-40 rounded-md border border-border bg-panel px-2 py-2" value={actionIndex} onChange={(event) => setActionIndex(Number(event.target.value))}>
            {frame.actor_output.map((_, index) => (
              <option key={index} value={index}>{frame.action_pipeline?.targets[index]?.channel || `action_${index}`}</option>
            ))}
          </select>
        </label>
        <label className="grid gap-1 text-xs text-secondary">
          Baseline
          <select className="min-w-56 rounded-md border border-border bg-panel px-2 py-2" value={baselineKind} onChange={(event) => setBaselineKind(event.target.value as BaselineKind)}>
          <option value="normalizer_mean">Normalizer running mean (normalized zero)</option>
          <option value="episode_start">First observation in this episode</option>
            {replayBaselineFrames.length > 0 ? <option value="selected_frame">Another frame in this replay</option> : null}
          </select>
        </label>
        {baselineKind === 'selected_frame' && selectedReplayBaseline ? (
          <label className="grid gap-1 text-xs text-secondary">
            Reference frame
            <select
              className="min-w-44 rounded-md border border-border bg-panel px-2 py-2"
              value={selectedReplayBaseline.sequence_id}
              onChange={(event) => setBaselineFrameSequence(Number(event.target.value))}
            >
              {replayBaselineFrames.map((item) => (
                <option key={item.sequence_id} value={item.sequence_id}>
                  seq {item.sequence_id}{item.transition ? ` · step ${item.transition.episode_step}` : ''}
                </option>
              ))}
            </select>
          </label>
        ) : null}
        <label className="grid gap-1 text-xs text-secondary">
          Integration steps
          <select className="rounded-md border border-border bg-panel px-2 py-2" value={nSteps} onChange={(event) => setNSteps(Number(event.target.value))}>
            {[16, 32, 64, 128].map((steps) => <option key={steps} value={steps}>{steps}</option>)}
          </select>
        </label>
        <button
          type="button"
          className="rounded-md border border-border-strong bg-raised px-3 py-2 text-xs font-semibold hover:bg-hover disabled:opacity-50"
          disabled={!viewer || stalePolicy || request.isPending || (baselineKind === 'selected_frame' && !selectedReplayBaseline)}
          onClick={() => request.mutate()}
        >
          {request.isPending || explanation?.status === 'pending' ? 'Explaining…' : 'Explain selected output'}
        </button>
        {stalePolicy ? <span className="text-xs text-warning">This replay uses {frame.checkpoint}; viewer currently has {viewer?.checkpoint}. Load that checkpoint before explaining.</span> : null}
      </div>
      {request.error ? <p className="mt-3 text-xs text-danger">Could not queue explanation: {request.error.message}</p> : null}
      {explanation?.status === 'error' ? <p className="mt-3 rounded border border-danger/30 bg-danger/10 p-3 text-xs text-danger">Explanation unavailable: {explanation.error}</p> : null}
      {explanation?.status === 'complete' ? (
        <div className="mt-4">
          <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-secondary">
            <span>Baseline: {explanation.baseline_description}</span>
            <span>{explanation.n_steps} steps</span>
            <span>Convergence delta: {explanation.convergence_delta?.toExponential(3) ?? '—'}</span>
            <span>{explanation.duration_ms?.toFixed(1) ?? '—'} ms</span>
          </div>
          <div className="mt-3 space-y-1.5">
            {rankedFeatures.map((item) => (
              <button
                key={item.index}
                type="button"
                className="grid w-full grid-cols-[minmax(8rem,1fr)_5rem_3.5rem] items-center gap-2 rounded px-1 py-0.5 text-left text-xs hover:bg-hover"
                onClick={() => onFeatureSelect(item.index)}
                title={`Highlight ${item.label} in the observation inspector`}
              >
                <span className="truncate font-mono text-secondary">{item.label}</span>
                <span className="text-right font-mono">{item.value >= 0 ? '+' : ''}{item.value.toFixed(5)}</span>
                <span className="text-right text-muted">{(item.fraction * 100).toFixed(1)}%</span>
              </button>
            ))}
          </div>
          <p className="mt-2 text-[10px] text-muted">Signed attribution · share of total absolute attribution. Positive values support the selected deterministic action output relative to the stated baseline.</p>
        </div>
      ) : null}
    </details>
  )
}
